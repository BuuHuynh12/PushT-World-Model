"""Phần CHẤM PATH của world model.

PoseHead: latent chuẩn hóa [P, D] -> (chấm tròn x, y, khối T x, y, sin góc, cos góc), x/y chia 512. Train chung với
world model (loss đọc tư thế + FDE), dùng để đọc tư thế từ latent world model TƯỞNG TƯỢNG -> chấm điểm path.
dense_cost: chi phí của một path (thấp = tốt):
    tới đích (coverage > 0.95) ở bước k -> -2 + k/100 (ưu tiên tới đích, càng sớm càng tốt); ngược lại
    (1 - coverage dự đoán) + |khối T - đích|/256 + |Δgóc|/π + 0.3 |chấm tròn - khối T|/256   (ở bước cuối).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


GOAL = (256.0, 256.0, math.pi / 4)  # goal_pose của gym-pusht


class PoseHead(nn.Module):
    def __init__(self, P: int, D: int, hidden: int = 512):
        super().__init__()
        self.P, self.D = P, D
        self.net = nn.Sequential(nn.Flatten(1), nn.Linear(P * D, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, 6))

    def forward(self, z):  # z [..., P, D] -> [..., 6]
        lead = z.shape[:-2]
        return self.net(z.reshape(-1, self.P, self.D)).view(*lead, 6)


def dense_cost(pose, cov, thr: float = 0.95):
    """pose [..., K, 6] (đầu ra PoseHead), cov [..., K] coverage dự đoán -> chi phí [...] (thấp = tốt)."""
    hit = cov > thr
    k_first = torch.where(hit.any(-1), hit.float().argmax(-1) + 1, torch.zeros_like(cov[..., 0]).long())
    p = pose[..., -1, :]
    ax, ay, bx, by = (p[..., i] * 512 for i in range(4))
    ang = torch.atan2(p[..., 4], p[..., 5])
    dang = torch.remainder(ang - GOAL[2] + math.pi, 2 * math.pi) - math.pi
    c = (1 - cov[..., -1]) + torch.hypot(bx - GOAL[0], by - GOAL[1]) / 256 + dang.abs() / math.pi \
        + 0.3 * torch.hypot(ax - bx, ay - by) / 256
    return torch.where(k_first > 0, -2.0 + k_first.float() / 100.0, c)
