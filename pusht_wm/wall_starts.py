"""Sinh vị trí ban đầu với khối T SÁT TƯỜNG / SÁT GÓC (ngoài phân phối chuẩn của gym-pusht: tâm khối T 100..400).

Mỗi seed -> trạng thái [agent x, y, khối x, y, góc] (đúng thứ tự reset_to_state của gym-pusht), chỉ phụ thuộc seed:
  - 50% "góc": khối T cách HAI tường kề nhau một khe 2..30 đơn vị; 50% "cạnh": cách MỘT tường, chiều kia ngẫu nhiên.
  - Góc khối T ngẫu nhiên; cả khối nằm TRONG bàn (tường ở 5..506, dày 2).
  - Chấm tròn ngẫu nhiên, cách khối T > 25 đơn vị (bán kính chấm tròn 15).
"""
from __future__ import annotations

import numpy as np

# 2 hình chữ nhật của chữ T trong toạ độ thân (scale 30, length 4), trọng tâm (0, 45) — y hệt gym_pusht.add_tee
_BAR = np.array([(-60, 30), (60, 30), (60, 0), (-60, 0)], float)
_STEM = np.array([(-15, 30), (-15, 120), (15, 120), (15, 30)], float)
_COG = np.array([0.0, 45.0])
LO, HI = 9.0, 502.0          # mặt trong tường (5 + 2 + 2 lề, 506 - 2 - 2 lề)


def tee_vertices(x, y, ang):
    """-> ([4, 2], [4, 2]) toạ độ thế giới 2 phần của chữ T có TRỌNG TÂM tại (x, y), góc ang.
    Lưu ý quy ước gym-pusht: reset_to_state [bx, by, θ] đặt GỐC thân tại (bx, by) rồi xoay quanh trọng tâm
    -> trọng tâm = (bx, by + 45); body.position (get_full_state) sau đó là gốc thân ĐÃ dời."""
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return [(v - _COG) @ R.T + np.array([x, y]) for v in (_BAR, _STEM)]


def _dist_point_poly(p, poly):
    """Khoảng cách từ điểm tới đa giác lồi (0 nếu nằm trong)."""
    n = len(poly)
    inside = all(np.cross(poly[(i + 1) % n] - poly[i], p - poly[i]) >= 0 for i in range(n)) or \
        all(np.cross(poly[(i + 1) % n] - poly[i], p - poly[i]) <= 0 for i in range(n))
    if inside:
        return 0.0
    d = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
        d.append(np.linalg.norm(p - (a + t * (b - a))))
    return float(min(d))


def wall_start(seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    ang = float(rng.uniform(-np.pi, np.pi))
    rel = np.concatenate(tee_vertices(0.0, 0.0, ang))           # đỉnh so với trọng tâm
    mn, mx = rel.min(0), rel.max(0)
    xy = np.zeros(2)
    corner = rng.random() < 0.5
    axes = [0, 1] if corner else [int(rng.integers(2))]
    for k in range(2):
        lo, hi = LO - mn[k], HI - mx[k]                         # khoảng trọng tâm để cả khối nằm trong bàn
        if k in axes:
            gap = rng.uniform(2.0, 30.0)
            xy[k] = lo + gap if rng.random() < 0.5 else hi - gap  # sát tường thấp hoặc cao
        else:
            xy[k] = rng.uniform(lo, hi)
    polys = tee_vertices(xy[0], xy[1], ang)
    for _ in range(1000):
        a = rng.uniform(30.0, 482.0, 2)
        if min(_dist_point_poly(a, P) for P in polys) > 25.0:
            break
    return np.array([a[0], a[1], xy[0], xy[1] - _COG[1], ang], np.float64)   # trọng tâm -> tham số reset_to_state
