"""World model DẠNG TRẠNG THÁI — bám sát câu chữ đề bài:
"từ trạng thái hiện tại và hành động của robot, dự đoán sự thay đổi vị trí vật thể".

Trạng thái 6 chiều: (chấm tròn x, y, khối T x, y) chuẩn hóa x/256-1, cùng (sin, cos) góc khối T.
Đầu vào: trạng thái t-1 và t (2 bước -> suy ra được vận tốc, vì trạng thái không chứa vận tốc) + action t.
Đầu ra: Δ trạng thái -> trạng thái t+1 (sin/cos chuẩn hóa lại về đường tròn đơn vị).
Rollout tự hồi quy nhiều bước giống world model ảnh.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class StateWMConfig:
    hidden: int = 512
    n_layers: int = 4
    dropout: float = 0.0


def encode_state(sf: np.ndarray) -> np.ndarray:
    """state_full [..., 5] (agent x,y, khối x,y, góc rad) -> [..., 6]."""
    return np.concatenate([sf[..., :4] / 256.0 - 1.0, np.sin(sf[..., 4:5]), np.cos(sf[..., 4:5])], -1).astype(np.float32)


def decode_block_pose(s: np.ndarray) -> np.ndarray:
    """[..., 6] -> tư thế khối T (x, y đơn vị sim, góc rad)."""
    return np.stack([(s[..., 2] + 1) * 256, (s[..., 3] + 1) * 256, np.arctan2(s[..., 4], s[..., 5])], -1)


class StateWorldModel(nn.Module):
    def __init__(self, cfg: StateWMConfig = StateWMConfig()):
        super().__init__()
        self.cfg = cfg
        layers, d = [], 6 * 2 + 2
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(d, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
            d = cfg.hidden
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(d, 6)
        nn.init.zeros_(self.out.weight)   # khởi tạo Δ = 0 -> ban đầu = "không đổi"
        nn.init.zeros_(self.out.bias)

    def hidden(self, s_prev, s_cur, a):
        """Biểu diễn bên trong world model [..., hidden] (chứa hiểu biết động lực học) — dùng cho WM -> Diffusion."""
        return self.net(torch.cat([s_prev, s_cur, a], -1))

    def step(self, s_prev, s_cur, a):
        """s_* [B, 6], a [B, 2] (chuẩn hóa a/256-1) -> s_next [B, 6]."""
        nxt = s_cur + self.out(self.net(torch.cat([s_prev, s_cur, a], -1)))
        sc = nxt[:, 4:6] / nxt[:, 4:6].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.cat([nxt[:, :4], sc], -1)

    def rollout(self, s_prev, s_cur, actions):
        """actions [B, K, 2] -> trạng thái dự đoán [B, K, 6] ở t+1..t+K."""
        out = []
        for k in range(actions.shape[1]):
            s_next = self.step(s_prev, s_cur, actions[:, k])
            out.append(s_next)
            s_prev, s_cur = s_cur, s_next
        return torch.stack(out, 1)

    def checkpoint(self, extra=None):
        return {"state_wm_config": asdict(self.cfg), "state_dict": self.state_dict(), **(extra or {})}

    @classmethod
    def from_checkpoint(cls, ckpt):
        if not isinstance(ckpt, dict):
            from .paths import resolve
            ckpt = torch.load(resolve(ckpt), map_location="cpu", weights_only=False)
        m = cls(StateWMConfig(**ckpt["state_wm_config"]))
        m.load_state_dict(ckpt["state_dict"])
        return m
