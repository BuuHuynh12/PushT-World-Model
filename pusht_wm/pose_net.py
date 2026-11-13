"""ENCODER của world model có cấu trúc: ảnh 96x96 -> tư thế khối T (x, y, góc).

Vì sao: bước 11 cho thấy khâu yếu nhất của world model ảnh là ĐỌC tư thế từ latent DINOv2 đã gộp/nén
(sai số ~2.3 đơn vị ngay trên frame thật), còn khâu dự đoán động lực học thì tốt.
-> Tách world model thành  encoder (ảnh -> trạng thái) + dynamics (world model trạng thái, bước 11).
Vị trí chấm tròn KHÔNG cần đoán: env và dataset đều cho sẵn agent_pos.

Mạng: ResNet18 (bỏ maxpool để giữ độ phân giải không gian, bản đồ đặc trưng 6x6) -> 4 số:
(khối x/256-1, khối y/256-1, sin góc, cos góc). Train trên ảnh sim có nhãn thật (sim_v1 + sim_v2).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torchvision


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)
        r.maxpool = nn.Identity()
        r.fc = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 4))
        self.net = r

    def forward(self, img_u8):
        """img_u8 uint8 [B, 96, 96, 3] (tensor) -> [B, 4] (x, y chuẩn hóa, sin, cos chuẩn hóa về đường tròn)."""
        x = img_u8.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        o = self.net(x)
        sc = o[:, 2:4] / o[:, 2:4].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.cat([o[:, :2], sc], -1)

    @torch.no_grad()
    def block_pose(self, img_u8: np.ndarray, dev, batch: int = 1024) -> np.ndarray:
        """ảnh uint8 [N, 96, 96, 3] (numpy) -> tư thế khối T [N, 3] (x, y đơn vị sim, góc rad)."""
        out = []
        for i in range(0, len(img_u8), batch):
            o = self(torch.from_numpy(np.ascontiguousarray(img_u8[i:i + batch])).to(dev)).float().cpu().numpy()
            out.append(np.stack([(o[:, 0] + 1) * 256, (o[:, 1] + 1) * 256, np.arctan2(o[:, 2], o[:, 3])], 1))
        return np.concatenate(out) if out else np.zeros((0, 3))


def pose_target(state_full: np.ndarray) -> np.ndarray:
    """state_full [N, 5] -> nhãn [N, 4]."""
    return np.stack([state_full[:, 2] / 256 - 1, state_full[:, 3] / 256 - 1,
                     np.sin(state_full[:, 4]), np.cos(state_full[:, 4])], 1).astype(np.float32)


def pose_errors(pred3: np.ndarray, true_sf: np.ndarray):
    """-> (sai số vị trí [N] đơn vị sim, sai số góc [N] độ)."""
    e_pos = np.hypot(pred3[:, 0] - true_sf[:, 2], pred3[:, 1] - true_sf[:, 3])
    e_ang = np.degrees(np.abs((pred3[:, 2] - true_sf[:, 4] + math.pi) % (2 * math.pi) - math.pi))
    return e_pos, e_ang
