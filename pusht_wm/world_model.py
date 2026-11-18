"""world model trong không gian latent (kiểu DINO-WM, rút gọn).

Đầu vào: W frame latent gần nhất z [B, F<=W, P, D] (đã chuẩn hóa) + action tại mỗi frame a [B, F, 2].
Token = Linear(z) + nhúng vị trí không gian (P) + nhúng thời gian (F) + nhúng action của frame.
Transformer với mặt nạ NHÂN QUẢ THEO FRAME: token ở frame t chỉ nhìn token của frame <= t.
Đầu ra tại frame t:
  z_next[t] = z[t] + Δ   (dự đoán latent frame t+1; dạng residual -> khởi đầu gần "copy frame")
  r[t]                   (coverage sau khi thực thi a[t])
Rollout: tự hồi quy, đưa dự đoán vào lại, cửa sổ trượt W frame.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F


@dataclass
class WMConfig:
    P: int = 64            # số token mỗi frame (theo encoder)
    D: int = 128           # số chiều mỗi token (theo encoder)
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    window: int = 4        # số frame tối đa model nhìn cùng lúc
    action_dim: int = 2
    dropout: float = 0.0
    grad_ckpt: bool = True  # gradient checkpointing khi train: rollout có BPTT tốn rất nhiều VRAM nếu tắt
    pose_head: bool = False  # đầu đọc tư thế (chấm tròn + khối T) từ latent, train CHUNG với WM -> dùng cho loss FDE


class Block(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h, self.drop = h, dropout

    def forward(self, x, mask):
        B, N, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(y.transpose(1, 2).reshape(B, N, d))
        return x + self.mlp(self.ln2(x))


class LatentWorldModel(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.inp = nn.Linear(cfg.D, d)
        self.act = nn.Sequential(nn.Linear(cfg.action_dim, d), nn.GELU(), nn.Linear(d, d))
        self.pos_s = nn.Parameter(torch.randn(cfg.P, d) * 0.02)
        self.pos_t = nn.Parameter(torch.randn(cfg.window, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.D)
        nn.init.zeros_(self.out.weight)   # khởi tạo Δ = 0 -> ban đầu = "copy frame hiện tại"
        nn.init.zeros_(self.out.bias)
        self.rew = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.pose = None
        if cfg.pose_head:
            from .wm_planner import PoseHead
            self.pose = PoseHead(cfg.P, cfg.D)

    def _mask(self, n_frames, device):
        f = torch.arange(n_frames, device=device).repeat_interleave(self.cfg.P)
        return f[None, :] <= f[:, None]  # [N, N] True = được nhìn

    def hidden(self, z, a):
        """Biểu diễn bên trong (sau transformer + LayerNorm) [B, F, P, d_model].
        Dùng cho policy "WM -> Diffusion": policy đọc trạng thái qua hiểu biết động lực học của world model."""
        B, Fr, P, D = z.shape
        assert Fr <= self.cfg.window and P == self.cfg.P
        x = self.inp(z) + self.pos_s[None, None] + self.pos_t[None, :Fr, None] + self.act(a)[:, :, None]
        x = x.reshape(B, Fr * P, -1)
        mask = self._mask(Fr, z.device)
        for blk in self.blocks:
            if self.cfg.grad_ckpt and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(blk, x, mask, use_reentrant=False)
            else:
                x = blk(x, mask)
        return self.ln(x).view(B, Fr, P, -1)

    def forward(self, z, a):
        """z [B,F,P,D], a [B,F,2] -> z_next [B,F,P,D], r [B,F]."""
        x = self.hidden(z, a)
        z_next = z + self.out(x)
        r = self.rew(x.mean(2)).squeeze(-1)
        return z_next, r

    def rollout(self, z_ctx, a_ctx, a_future):
        """Tự hồi quy K bước.

        z_ctx [B,C,P,D]: latent các frame ngữ cảnh t-C+1..t (thật)
        a_ctx [B,C-1,2]: action đã thực thi tại các frame ngữ cảnh trừ frame cuối
        a_future [B,K,2]: action sẽ thực thi tại t, t+1, ..., t+K-1
        -> z_pred [B,K,P,D] (frame t+1..t+K), r_pred [B,K] (coverage sau mỗi action tương lai)
        """
        zs = list(z_ctx.unbind(1))
        acts = list(a_ctx.unbind(1))
        W = self.cfg.window
        z_out, r_out = [], []
        for k in range(a_future.shape[1]):
            acts.append(a_future[:, k])
            zw = torch.stack(zs[-W:], 1)
            aw = torch.stack(acts[-W:], 1)
            zn, r = self(zw, aw)
            zs.append(zn[:, -1])
            z_out.append(zn[:, -1])
            r_out.append(r[:, -1])
        return torch.stack(z_out, 1), torch.stack(r_out, 1)


def _rmse(pred, target, w_hi: float):
    """MSE reward, trọng số (1 + w_hi) ở vùng coverage thật > 0.8 (nơi cần phân biệt path tinh nhất)."""
    w = 1.0 + w_hi * (target > 0.8).float()
    return (w * (pred.float() - target) ** 2).sum() / w.sum()


def pose_fde(pred, true, r: float = 60.0):
    """FDE (final displacement error) của KHỐI T, đơn vị sim: sqrt(dx^2 + dy^2) + r * |dθ|.
    pred, true [..., 6] = (chấm tròn x, y, khối T x, y) / 512, sin góc, cos góc. r: đơn vị sim cho mỗi radian."""
    pred, true = pred.float(), true.float()
    d = (pred[..., 2:4] - true[..., 2:4]) * 512.0
    pos = torch.sqrt((d ** 2).sum(-1) + 1e-6)
    sp, cp, st, ct = pred[..., 4], pred[..., 5], true[..., 4], true[..., 5]
    ang = torch.atan2(sp * ct - cp * st, cp * ct + sp * st + 1e-6).abs()   # |góc dự đoán - góc thật| trong [0, π]
    return pos + r * ang


def wm_losses(model: LatentWorldModel, z, a, r, n_ctx: int, n_roll: int, w_reward: float = 10.0,
              w_hi: float = 0.0, pose=None, has_pose=None, w_fde: float = 0.0, w_pose: float = 0.0,
              fde_r: float = 60.0):
    """z [B,L,P,D], a [B,L,2], r [B,L] với L >= max(window+1, n_ctx+n_roll).

    teacher forcing: cửa sổ W frame đầu, dự đoán frame kế tiếp tại mọi vị trí
    rollout: từ n_ctx frame thật, tự hồi quy n_roll bước (gradient đi qua cả chuỗi)
    FDE (khi model có đầu đọc tư thế và có pose): tư thế khối T đọc từ latent DỰ ĐOÁN ở bước CUỐI rollout so với
    tư thế thật (Δx, Δy, Δθ). pose [B,L,6] tư thế thật từng frame, has_pose [B,L] (demo lerobot/pusht không có
    tư thế khối T -> False -> chỉ góp vào phần loss latent).
    """
    W = model.cfg.window
    zp, rp = model(z[:, :W], a[:, :W])
    tf_z = F.mse_loss(zp, z[:, 1:W + 1])
    tf_r = _rmse(rp, r[:, :W], w_hi)
    C, K = n_ctx, n_roll
    zr, rr = model.rollout(z[:, :C], a[:, :C - 1], a[:, C - 1:C - 1 + K])
    ro_z = F.mse_loss(zr, z[:, C:C + K])
    ro_r = _rmse(rr, r[:, C - 1:C - 1 + K], w_hi)
    loss = tf_z + ro_z + w_reward * (tf_r + ro_r)
    parts = {"tf_z": tf_z.item(), "tf_r": tf_r.item(), "ro_z": ro_z.item(), "ro_r": ro_r.item()}
    if model.pose is not None and pose is not None and (w_fde > 0 or w_pose > 0):
        m = has_pose[:, :W].float()
        read = ((model.pose(z[:, :W]).float() - pose[:, :W]).abs().mean(-1) * m).sum() / m.sum().clamp_min(1)
        mf = has_pose[:, C + K - 1].float()
        fde = (pose_fde(model.pose(zr[:, -1]), pose[:, C + K - 1], fde_r) * mf).sum() / mf.sum().clamp_min(1)
        loss = loss + w_pose * read + w_fde * fde / 256.0
        parts.update(pose_read=read.item(), fde=fde.item())
    return loss, parts


def wm_checkpoint(model: LatentWorldModel, stats, extra: dict | None = None) -> dict:
    return {"wm_config": asdict(model.cfg), "state_dict": model.state_dict(),
            "latent_stats": stats.state_dict(), **(extra or {})}


def load_wm(ckpt, device="cpu"):
    """-> (model ở chế độ eval, dict checkpoint). ckpt: đường dẫn hoặc dict đã nạp."""
    c = ckpt if isinstance(ckpt, dict) else torch.load(ckpt, map_location="cpu", weights_only=False)
    m = LatentWorldModel(WMConfig(**c["wm_config"])).to(device).eval()
    m.load_state_dict(c["state_dict"])
    return m, c
