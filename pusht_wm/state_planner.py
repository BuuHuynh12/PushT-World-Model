"""Lập kế hoạch KHÔNG POLICY bằng WORLD MODEL TRẠNG THÁI (chạy trên GPU).

Ý tưởng: giống oracle "dense" (bộ mô phỏng hoàn hảo + CEM: 94% success, 89 bước, 200 seed),
nhưng thay bộ mô phỏng bằng world model trạng thái (bước 11: sai số khối T sau 16 bước ~2 đơn vị lúc bị đẩy).
Mỗi lần quyết định: S chuỗi action × I vòng CEM, world model "tưởng tượng" trạng thái tương lai của TẤT CẢ
chuỗi cùng lúc (batch GPU), chấm bằng chi phí dày, giữ E chuỗi tốt nhất, cập nhật phân phối.

Coverage (tỉ lệ khối T nằm trong vùng đích) là hàm HÌNH HỌC của tư thế khối T -> tính trực tiếp từ tư thế dự đoán:
lấy lưới điểm đều bên trong khối T ở đích, đếm tỉ lệ điểm nằm trong khối T dự đoán (= diện tích giao / diện tích đích,
đúng định nghĩa của gym-pusht `_get_coverage`). Kiểm tra khớp với env: scripts/check_state_planner.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .state_wm import StateWorldModel

GOAL = (256.0, 256.0, math.pi / 4)   # goal_pose của gym-pusht
SCALE, LENGTH = 30.0, 4.0            # add_tee(scale=30), length=4
# 2 hình chữ nhật của khối T trong hệ tọa độ thân (gốc = body.position): (x_min, x_max, y_min, y_max)
TEE_RECTS = ((-LENGTH * SCALE / 2, LENGTH * SCALE / 2, 0.0, SCALE),
             (-SCALE / 2, SCALE / 2, SCALE, LENGTH * SCALE))


def _rot(x, y, ang):
    c, s = math.cos(ang), math.sin(ang)
    return c * x - s * y, s * x + c * y


def goal_points(step: float = 1.5) -> np.ndarray:
    """Lưới điểm (tâm ô) phủ đều khối T ở đích, tọa độ thế giới [M, 2]."""
    pts = []
    for x0, x1, y0, y1 in TEE_RECTS:
        xs = np.arange(x0 + step / 2, x1, step)
        ys = np.arange(y0 + step / 2, y1, step)
        gx, gy = np.meshgrid(xs, ys)
        pts.append(np.stack([gx.ravel(), gy.ravel()], 1))
    p = np.concatenate(pts)
    wx, wy = _rot(p[:, 0], p[:, 1], GOAL[2])
    return np.stack([wx + GOAL[0], wy + GOAL[1]], 1).astype(np.float32)


class TeeCoverage:
    """coverage(bx, by, ang) trên GPU cho nhiều tư thế cùng lúc."""

    def __init__(self, dev, step: float = 1.5):
        self.pts = torch.from_numpy(goal_points(step)).to(dev)            # [M, 2]

    def __call__(self, bx, by, ang):
        """bx, by, ang [...] -> coverage [...] trong [0, 1]."""
        px = self.pts[:, 0] - bx[..., None]                              # [..., M]
        py = self.pts[:, 1] - by[..., None]
        c, s = torch.cos(ang)[..., None], torch.sin(ang)[..., None]
        lx, ly = c * px + s * py, -s * px + c * py                        # quay ngược về hệ thân khối T
        inside = torch.zeros_like(lx, dtype=torch.bool)
        for x0, x1, y0, y1 in TEE_RECTS:
            inside |= (lx >= x0) & (lx <= x1) & (ly >= y0) & (ly <= y1)
        return inside.float().mean(-1)


def state_vec(s5: np.ndarray) -> np.ndarray:
    """trạng thái env [5] (agent x, y, khối x, y, góc) -> vector 6 chiều của world model trạng thái."""
    return np.concatenate([s5[:4] / 256.0 - 1.0, [math.sin(s5[4]), math.cos(s5[4])]]).astype(np.float32)


def dense_cost_state(traj, cover: TeeCoverage, thr: float = 0.95, w_agent: float = 0.3):
    """traj [S, K, 6] trạng thái dự đoán -> (chi phí [S], coverage [S, K]).
    Giống oracle dense: tới đích ở bước k -> -2 + k/100; ngược lại
    (1 - coverage) + |khối T - đích|/256 + |Δgóc|/π + w_agent·|chấm tròn - khối T|/256 (ở bước cuối)."""
    ax, ay, bx, by = ((traj[..., i] + 1) * 256 for i in range(4))
    ang = torch.atan2(traj[..., 4], traj[..., 5])
    cov = cover(bx, by, ang)                                              # [S, K]
    hit = cov > thr
    k_first = torch.where(hit.any(-1), hit.float().argmax(-1) + 1, torch.zeros_like(cov[:, 0]).long())
    dang = torch.remainder(ang[:, -1] - GOAL[2] + math.pi, 2 * math.pi) - math.pi
    c = (1 - cov[:, -1]) + torch.hypot(bx[:, -1] - GOAL[0], by[:, -1] - GOAL[1]) / 256 + dang.abs() / math.pi \
        + w_agent * torch.hypot(ax[:, -1] - bx[:, -1], ay[:, -1] - by[:, -1]) / 256
    return torch.where(k_first > 0, -2.0 + k_first.float() / 100.0, c), cov


class StateCEMPlanner:
    def __init__(self, wm: StateWorldModel, dev, horizon=16, samples=256, iters=4, elites=24, std0=60.0, seed=0,
                 cov_step: float = 1.5):
        self.wm, self.dev = wm.to(dev).eval(), dev
        self.H, self.S, self.I, self.E, self.std0 = horizon, samples, iters, elites, std0
        self.cover = TeeCoverage(dev, cov_step)
        self.g = torch.Generator(device=dev).manual_seed(seed)

    @torch.no_grad()
    def plan(self, s_prev5: np.ndarray, s_cur5: np.ndarray, mean: np.ndarray | None):
        """s_prev5, s_cur5: trạng thái env [5] ở t-1, t; mean [H, 2] kế hoạch cũ đã dịch (hoặc None).
        -> (kế hoạch tốt nhất [H, 2] đơn vị sim, chi phí dự đoán, quỹ đạo tưởng tượng [H, 6] của kế hoạch đó)."""
        dev, H, S = self.dev, self.H, self.S
        sp = torch.from_numpy(state_vec(s_prev5)).to(dev)[None].expand(S, 6)
        sc = torch.from_numpy(state_vec(s_cur5)).to(dev)[None].expand(S, 6)
        mu = torch.from_numpy(mean if mean is not None else np.repeat(s_cur5[None, :2], H, 0)).float().to(dev)
        std = torch.full((H, 2), self.std0, device=dev)
        best, best_c, best_traj = mu.clone(), float("inf"), None
        for _ in range(self.I):
            cand = mu[None] + torch.randn((S, H, 2), generator=self.g, device=dev) * std[None]
            cand[0] = mu                                                  # luôn giữ kế hoạch hiện tại
            cand = cand.clamp(0, 512)
            traj = self.wm.rollout(sp, sc, cand / 256.0 - 1.0)            # [S, H, 6]
            cost, _ = dense_cost_state(traj, self.cover)
            idx = torch.argsort(cost)[:self.E]
            el = cand[idx]
            mu, std = el.mean(0), el.std(0) + 2.0
            if cost[idx[0]] < best_c:
                best_c, best, best_traj = float(cost[idx[0]]), cand[idx[0]].clone(), traj[idx[0]].clone()
        return best.cpu().numpy(), best_c, best_traj.cpu().numpy()
