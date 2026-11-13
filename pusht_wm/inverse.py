"""Mô hình NGHỊCH (inverse dynamics): từ đường đi trạng thái -> action.

Dùng cho sơ đồ  frame -> WM -> s_t -> Diffusion -> PATH -> action:
Diffusion sinh path (các trạng thái tương lai); mô hình nghịch trả lời "phải ra lệnh gì để đi từ s_k tới s_{k+1}".
Đầu vào: trạng thái mã hóa 6 chiều (encode_state) ở k-1, k, k+1 (k-1 cho biết vận tốc hiện tại của chấm tròn,
vì chấm tròn được điều khiển PD về điểm đích = action). Đầu ra: action chuẩn hóa a/256 - 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class InverseConfig:
    hidden: int = 512
    layers: int = 3


class InverseModel(nn.Module):
    def __init__(self, cfg: InverseConfig = InverseConfig()):
        super().__init__()
        self.cfg = cfg
        L, d = [], 18
        for _ in range(cfg.layers):
            L += [nn.Linear(d, cfg.hidden), nn.Mish()]
            d = cfg.hidden
        self.net = nn.Sequential(*L, nn.Linear(d, 2))

    def forward(self, s_prev, s_cur, s_next):
        """s_* [..., 6] -> action chuẩn hóa [..., 2]. Dự đoán ĐỘ LỆCH so với vị trí chấm tròn hiện tại (dễ học hơn)."""
        return s_cur[..., :2] + self.net(torch.cat([s_prev, s_cur, s_next], -1))

    def checkpoint(self, extra=None):
        return {"inverse_config": asdict(self.cfg), "state_dict": self.state_dict(), **(extra or {})}

    @classmethod
    def from_checkpoint(cls, ckpt):
        if not isinstance(ckpt, dict):
            from .paths import resolve
            ckpt = torch.load(resolve(ckpt), map_location="cpu", weights_only=False)
        m = cls(InverseConfig(**ckpt["inverse_config"]))
        m.load_state_dict(ckpt["state_dict"])
        return m
