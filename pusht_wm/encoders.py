"""Phần đọc ảnh của world model: DINOv2 ViT-S/14 đóng băng + PCA.

encoder.encode(frames uint8 [B, 1, 96, 96, 3]) -> latent float [B, 64, 128]
  ảnh 96 -> 224, 16x16 patch x 384 -> avg-pool 8x8 = 64 token -> PCA 128 chiều.
Pool + PCA để vừa RAM/GPU. Tham số PCA lưu cùng encoder (và nhúng trong checkpoint world model) để mã hóa
ảnh mới lúc chạy dùng đúng phép chiếu đã dùng khi cache.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PCAProjector:
    """Chiếu token D_in -> k chiều (fit bằng SVD trên mẫu token)."""

    def __init__(self, mean=None, comps=None):
        self.mean, self.comps = mean, comps  # [D_in], [D_in, k]

    @torch.no_grad()
    def fit(self, tokens: torch.Tensor, k: int) -> float:
        x = tokens.reshape(-1, tokens.shape[-1]).double()
        self.mean = x.mean(0)
        _, S, Vh = torch.linalg.svd(x - self.mean, full_matrices=False)
        self.comps = Vh[:k].T.contiguous()
        var = S ** 2
        self.mean, self.comps = self.mean.float(), self.comps.float()
        return float(var[:k].sum() / var.sum())  # tỉ lệ phương sai giữ lại

    def __call__(self, tokens):
        return (tokens - self.mean.to(tokens.device)) @ self.comps.to(tokens.device)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "comps": self.comps.cpu()}


class BaseEncoder:
    name = "base"
    n_frames = 1

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.pca: PCAProjector | None = None

    def _raw_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """frames uint8 [B, n_frames, H, W, 3] trên device -> token chưa chiếu [B, P, D_in]."""
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, frames) -> torch.Tensor:
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        tok = self._raw_tokens(frames.to(self.device))
        return self.pca(tok) if self.pca is not None else tok

    def fit_pca(self, frames, k: int, chunk: int = 256) -> float:
        toks = torch.cat([self._raw_tokens(torch.from_numpy(frames[i:i + chunk]).to(self.device)).float().cpu()
                          for i in range(0, len(frames), chunk)])
        self.pca = PCAProjector()
        return self.pca.fit(toks, k)

    # lưu/tải phần trạng thái cần để mã hóa lại y hệt (cấu hình + PCA)
    def config(self) -> dict:
        return {"name": self.name}

    def save(self, path):
        torch.save({"config": self.config(), "pca": self.pca.state_dict() if self.pca else None}, path)


def _pool(tok: torch.Tensor, grid: int, out: int) -> torch.Tensor:
    """[B, grid*grid, D] -> avg-pool -> [B, out*out, D]."""
    B, _, D = tok.shape
    x = tok.transpose(1, 2).reshape(B, D, grid, grid)
    x = F.adaptive_avg_pool2d(x, out)
    return x.flatten(2).transpose(1, 2)


class DinoV2Encoder(BaseEncoder):
    name = "dinov2"

    def __init__(self, model_id="facebook/dinov2-small", res=224, pool=8, device="cuda"):
        super().__init__(device)
        from transformers import AutoModel

        self.model_id, self.res, self.pool = model_id, res, pool
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval().requires_grad_(False)
        self.grid = res // 14

    def _raw_tokens(self, frames):
        x = frames[:, -1].permute(0, 3, 1, 2).float() / 255  # chỉ frame hiện tại
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        h = self.model(pixel_values=x).last_hidden_state[:, 1:]  # bỏ CLS -> [B, 256, 384]
        h = h[:, -self.grid * self.grid:]                      # (phòng khi model có register token)
        return _pool(h, self.grid, self.pool) if self.pool < self.grid else h

    def config(self):
        return {"name": self.name, "model_id": self.model_id, "res": self.res, "pool": self.pool}


def build_encoder(name: str, device="cuda", **kw) -> BaseEncoder:
    return {"dinov2": DinoV2Encoder}[name](device=device, **kw)


def load_encoder(path, device="cuda") -> BaseEncoder:
    """Tải encoder đã dùng khi cache (cấu hình + PCA)."""
    # path: file encoder.pt, hoặc dict đã nạp (cấu hình + PCA nhúng sẵn trong checkpoint world model)
    s = path if isinstance(path, dict) else torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(s["config"])
    enc = build_encoder(cfg.pop("name"), device=device, **cfg)
    if s["pca"] is not None:
        enc.pca = PCAProjector(s["pca"]["mean"], s["pca"]["comps"])
    return enc


def frame_stack(raw, idx: np.ndarray, n_frames: int) -> np.ndarray:
    """Lấy n_frames frame kết thúc tại idx (kẹp ở đầu episode) -> [B, n_frames, H, W, 3]."""
    starts = np.r_[0, raw.episode_ends[:-1]][raw.episode_index[idx]]
    offs = np.arange(-n_frames + 1, 1)
    j = np.maximum(idx[:, None] + offs[None], starts[:, None])
    return raw.images[j]


def cache_latents(enc: BaseEncoder, raw, out_path, batch: int = 128, log=print) -> np.ndarray:
    """Mã hóa toàn bộ frame của raw -> memmap fp16 [N, P, D] tại out_path (.npy)."""
    N = len(raw.images)
    probe = enc.encode(frame_stack(raw, np.arange(2), enc.n_frames))
    P, D = probe.shape[1:]
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(N, P, D))
    import time
    t0 = time.time()
    for i in range(0, N, batch):
        idx = np.arange(i, min(N, i + batch))
        out[idx] = enc.encode(frame_stack(raw, idx, enc.n_frames)).cpu().numpy().astype(np.float16)
        if (i // batch) % 100 == 0:
            log(f"    {i + len(idx)}/{N} frame ({(i + len(idx)) / (time.time() - t0):.0f} frame/s)")
    out.flush()
    return out
