"""Mạng cho Diffusion Policy (theo real-stanford/diffusion_policy).

- ResNet18 train từ đầu, BatchNorm -> GroupNorm (ổn định với EMA / batch nhỏ),
  avgpool+fc -> SpatialSoftmax (32 keypoint) -> Linear 64.
- ConditionalUnet1D: U-Net 1D theo trục thời gian của action chunk, điều kiện FiLM
  từ (embedding bước khuếch tán + đặc trưng quan sát).
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Bộ mã hóa ảnh
# ============================================================================
def replace_bn_with_gn(module: nn.Module, features_per_group: int = 16) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            setattr(module, name, nn.GroupNorm(c // features_per_group, c))
        else:
            replace_bn_with_gn(child, features_per_group)
    return module


class SpatialSoftmax(nn.Module):
    """Biến feature map [B,C,H,W] thành tọa độ kỳ vọng (x,y) của num_kp keypoint."""

    def __init__(self, in_c: int, num_kp: int = 32):
        super().__init__()
        self.kp = nn.Conv2d(in_c, num_kp, 1)
        self.num_kp = num_kp

    def forward(self, x):
        x = self.kp(x)
        B, K, H, W = x.shape
        ys = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        attn = F.softmax(x.flatten(2).float(), dim=-1).to(x.dtype)  # [B,K,HW]
        ex = (attn * gx.flatten()).sum(-1)
        ey = (attn * gy.flatten()).sum(-1)
        return torch.stack([ex, ey], -1).flatten(1)  # [B, 2K]


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim: int = 64, num_kp: int = 32, crop: int = 84):
        super().__init__()
        import torchvision

        rn = torchvision.models.resnet18(weights=None)
        rn = replace_bn_with_gn(rn)
        self.trunk = nn.Sequential(*list(rn.children())[:-2])  # bỏ avgpool, fc
        self.pool = SpatialSoftmax(512, num_kp)
        self.out = nn.Linear(2 * num_kp, feature_dim)
        self.crop = crop
        self.feature_dim = feature_dim

    def _crop(self, x):
        """Train: random crop (mỗi ảnh một vị trí); eval: center crop."""
        H, W = x.shape[-2:]
        c = self.crop
        if c >= H:
            return x
        if self.training:
            B = x.shape[0]
            ar = torch.arange(c, device=x.device)
            rows = torch.randint(0, H - c + 1, (B, 1), device=x.device) + ar  # [B, c]
            cols = torch.randint(0, W - c + 1, (B, 1), device=x.device) + ar
            b = torch.arange(B, device=x.device)[:, None, None]
            # chỉ số nâng cao tách bởi ':' -> kết quả [B, c, c, C]
            return x[b, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
        o = (H - c) // 2
        return x[..., o:o + c, o:o + c]

    def forward(self, img):  # img [B,3,H,W] float [0,1]
        x = self._crop(img)
        return F.relu(self.out(self.pool(self.trunk(x))))


class ObsEncoder(nn.Module):
    """(ảnh [B,n_obs,3,H,W], state [B,n_obs,2]) -> vector điều kiện [B, n_obs*(feat+2)]."""

    def __init__(self, n_obs: int = 2, state_dim: int = 2, feature_dim: int = 64, crop: int = 84):
        super().__init__()
        self.image = ImageEncoder(feature_dim, crop=crop)
        self.n_obs = n_obs
        self.out_dim = n_obs * (feature_dim + state_dim)

    def forward(self, image, state):
        B, n = image.shape[:2]
        f = self.image(image.flatten(0, 1)).view(B, n, -1)
        return torch.cat([f, state], -1).flatten(1)


# ============================================================================
# U-Net 1D có điều kiện
# ============================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        e = t.float()[:, None] * f[None]
        return torch.cat([e.sin(), e.cos()], -1)


class Conv1dBlock(nn.Sequential):
    def __init__(self, cin, cout, k, n_groups=8):
        super().__init__(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(n_groups, cout), nn.Mish())


class CondResBlock1D(nn.Module):
    """Hai Conv1dBlock, ở giữa điều biến FiLM: h = scale * h + bias."""

    def __init__(self, cin, cout, cond_dim, k=5, n_groups=8):
        super().__init__()
        self.b1 = Conv1dBlock(cin, cout, k, n_groups)
        self.b2 = Conv1dBlock(cout, cout, k, n_groups)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.cout = cout

    def forward(self, x, cond):
        h = self.b1(x)
        scale, bias = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        h = scale * h + bias
        return self.b2(h) + self.res(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim: int, global_cond_dim: int, step_embed_dim: int = 128,
                 down_dims=(256, 512, 1024), kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.step_emb = nn.Sequential(
            SinusoidalPosEmb(step_embed_dim), nn.Linear(step_embed_dim, step_embed_dim * 4),
            nn.Mish(), nn.Linear(step_embed_dim * 4, step_embed_dim))
        cond_dim = step_embed_dim + global_cond_dim
        dims = [input_dim, *down_dims]
        pairs = list(zip(dims[:-1], dims[1:]))
        mk = lambda a, b: CondResBlock1D(a, b, cond_dim, kernel_size, n_groups)  # noqa: E731

        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(pairs):
            last = i == len(pairs) - 1
            self.down.append(nn.ModuleList([
                mk(a, b), mk(b, b), nn.Conv1d(b, b, 3, 2, 1) if not last else nn.Identity()]))
        mid = down_dims[-1]
        self.mid = nn.ModuleList([mk(mid, mid), mk(mid, mid)])
        self.up = nn.ModuleList()
        # như repo gốc: mọi tầng up đều upsample (16 -> 8 -> 4 khi down, 4 -> 8 -> 16 khi up)
        for a, b in reversed(pairs[1:]):
            self.up.append(nn.ModuleList([mk(b * 2, a), mk(a, a), nn.ConvTranspose1d(a, a, 4, 2, 1)]))
        self.final = nn.Sequential(Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
                                   nn.Conv1d(down_dims[0], input_dim, 1))

    def forward(self, x, t, global_cond):
        """x [B, T, D] (D = chiều action), t [B], global_cond [B, G] -> eps [B, T, D]."""
        x = x.transpose(1, 2)
        cond = torch.cat([self.step_emb(t), global_cond], -1)
        skips = []
        for r1, r2, down in self.down:
            x = r2(r1(x, cond), cond)
            skips.append(x)
            x = down(x)
        for r in self.mid:
            x = r(x, cond)
        for r1, r2, up in self.up:
            x = torch.cat([x, skips.pop()], 1)
            x = up(r2(r1(x, cond), cond))
        return self.final(x).transpose(1, 2)


class EMA:
    """EMA trọng số như repo gốc: decay = 1 - (1 + step)^-0.75, tối đa 0.9999 (chép từ policy.py cũ)."""

    def __init__(self, model: nn.Module, power: float = 0.75, max_decay: float = 0.9999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.power, self.max_decay, self.step = power, max_decay, 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        d = min(self.max_decay, max(0.0, 1 - (1 + self.step) ** -self.power))
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)
