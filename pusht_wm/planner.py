"""World model ẢNH lúc chạy online: mã hóa frame thành latent s_t và "tưởng tượng" hậu quả của các chuỗi action.

  frame --(DINOv2 đóng băng + PCA, y hệt lúc cache)--> token --> chuẩn hóa = s_t (latent, KHÔNG phải tọa độ)
  s_t + chuỗi action --(world model rollout)--> latent tương lai --> coverage dự đoán + tư thế đọc từ latent

Điểm chọn path (ưu tiên 1: tới đích; 2: tới đích sớm): xem wm_planner.dense_cost.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .encoders import load_encoder
from .latent_data import norm_action, pool_tokens
from .world_model import load_wm


class WMScorer:
    def __init__(self, wm_ckpt: str, device="cuda", amp=torch.bfloat16):
        c = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
        self.dev = torch.device(device)
        self.model, _ = load_wm(c, self.dev)
        self.model.requires_grad_(False)
        self.mean = c["latent_stats"]["mean"].to(self.dev)
        self.std = c["latent_stats"]["std"].to(self.dev)
        self.pool = c.get("pool", 0)
        # encoder (cấu hình + PCA): nhúng trong checkpoint (bản mới) hoặc file riêng (bản cũ)
        self.encoder = load_encoder(c["encoder_state"] if "encoder_state" in c else c["encoder_path"], device=device)
        # đầu đọc tư thế: train chung với WM (bản mới) hoặc pose_head.pt cạnh checkpoint (bản cũ)
        self.pose_head = self.model.pose
        ph = Path(wm_ckpt).parent / "pose_head.pt"
        if self.pose_head is None and ph.exists():
            from .wm_planner import PoseHead
            pc = torch.load(ph, map_location="cpu", weights_only=False)
            self.pose_head = PoseHead(pc["P"], pc["D"]).to(self.dev).eval()
            self.pose_head.load_state_dict(pc["state_dict"])
        # không có GPU, hoặc GPU không có bf16 (GTX 16xx) -> chạy fp32
        if self.dev.type != "cuda" or (amp is torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False)):
            amp = None
        self.amp = amp
        self.step = c.get("step")

    def _autocast(self):
        return torch.autocast(self.dev.type, dtype=self.amp or torch.bfloat16, enabled=self.amp is not None)

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """frames uint8 [B, T, H, W, 3] -> latent chuẩn hóa [B, T, P, D] (mỗi frame mã hóa độc lập)."""
        assert self.encoder.n_frames == 1, "WMScorer chỉ hỗ trợ encoder 1 frame (dinov2/vae)"
        B, T = frames.shape[:2]
        flat = frames.reshape(B * T, 1, *frames.shape[2:])
        # mã hóa theo nhóm 64 ảnh: nhiều env cùng lúc làm DINOv2 tràn VRAM
        z = torch.cat([self.encoder.encode(flat[i:i + 64]).float() for i in range(0, len(flat), 64)])
        z = pool_tokens(z, self.pool)
        z = (z - self.mean) / self.std
        return z.view(B, T, *z.shape[1:])

    def _ctx(self, z, prev_action, paths):
        B, N, K, _ = paths.shape
        a_ctx = torch.from_numpy(norm_action(np.asarray(prev_action, np.float32))).to(self.dev)
        a_ctx = a_ctx.repeat_interleave(N, 0)[:, None]               # [B*N, 1, 2]
        a_fut = torch.from_numpy(norm_action(np.asarray(paths, np.float32).reshape(B * N, K, 2))).to(self.dev)
        return z.repeat_interleave(N, 0), a_ctx, a_fut

    @torch.no_grad()
    def imagine(self, frames, prev_action, paths, pose_head=None, thr: float = 0.95, z=None):
        """World model "tưởng tượng" kết quả của từng chuỗi action.
        paths [B, N, K, 2] -> (pose [B, N, K, 6] (chấm tròn x,y, khối T x,y /512, sin, cos góc) đọc từ latent dự đoán,
                                coverage [B, N, K], điểm [B, N] = -chi phí dày, cao = tốt).
        z: latent [B, 2, P, D] đã mã hóa sẵn (khỏi mã hóa lại frame)."""
        from .wm_planner import dense_cost
        head = pose_head if pose_head is not None else self.pose_head
        B, N, K, _ = paths.shape
        zc, a_ctx, a_fut = self._ctx(self.encode(frames) if z is None else z, prev_action, paths)
        with self._autocast():
            zp, r = self.model.rollout(zc, a_ctx, a_fut)
            pose = head(zp.float())
        pose, cov = pose.float(), r.float().clamp(0, 1)
        score = -dense_cost(pose, cov, thr)
        return (pose.view(B, N, K, 6).cpu().numpy(), cov.view(B, N, K).cpu().numpy(),
                score.view(B, N).cpu().numpy())
