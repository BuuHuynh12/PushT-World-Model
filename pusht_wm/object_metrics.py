"""Thước đo chính của đề bài: world model dự đoán SỰ THAY ĐỔI VỊ TRÍ VẬT THỂ (khối T) đúng tới đâu.

Đầu vào cho mỗi cửa sổ (N cửa sổ, nhìn trước K bước):
  pred [N, K, 3]  tư thế khối T dự đoán ở t+1..t+K: (x, y, góc rad), x/y đơn vị sim (khung 512)
  true [N, K, 3]  tư thế thật
  cur  [N, 3]     tư thế thật ở thời điểm hiện tại t
Báo theo k:
  - sai số vị trí (đơn vị sim) và góc (độ) của dự đoán, so với MỐC "đoán không đổi" (pred = cur)
  - riêng các cửa sổ khối T THẬT SỰ BỊ ĐẨY (dịch > 2 đơn vị hoặc xoay > 1° sau k bước) — nơi
    quy luật tương tác robot-vật thể thể hiện; phần lớn thời gian khối T đứng yên nên mốc
    "không đổi" rất mạnh ở tập tổng.
  - hướng chuyển động: cosine giữa Δ vị trí dự đoán và Δ thật; tỉ lệ độ lớn |Δ dự đoán| / |Δ thật|
"""
from __future__ import annotations

import numpy as np

MOVE_POS, MOVE_DEG = 2.0, 1.0  # ngưỡng coi là "khối T bị đẩy"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def pose_from_vec6(v: np.ndarray) -> np.ndarray:
    """(agent x, y, khối x, y /512, sin, cos) -> tư thế khối T (x, y đơn vị sim, góc rad)."""
    return np.stack([v[..., 2] * 512, v[..., 3] * 512, np.arctan2(v[..., 4], v[..., 5])], -1)


def object_metrics(pred, true, cur, ks=(1, 4, 8, 16)) -> dict:
    out = {}
    for k in ks:
        j = k - 1
        p, t = pred[:, j], true[:, j]
        e_pos = np.hypot(p[:, 0] - t[:, 0], p[:, 1] - t[:, 1])
        e_ang = np.degrees(np.abs(wrap(p[:, 2] - t[:, 2])))
        d_true = t[:, :2] - cur[:, :2]
        d_pred = p[:, :2] - cur[:, :2]
        mag_true = np.hypot(*d_true.T)
        rot_true = np.degrees(np.abs(wrap(t[:, 2] - cur[:, 2])))
        b_pos, b_ang = mag_true, rot_true                         # mốc "đoán không đổi"
        mv = (mag_true > MOVE_POS) | (rot_true > MOVE_DEG)
        r = {"n": int(len(p)), "n_moving": int(mv.sum()),
             "pos_err_mean": float(e_pos.mean()), "pos_err_median": float(np.median(e_pos)),
             "ang_err_median": float(np.median(e_ang)),
             "baseline_pos_err_mean": float(b_pos.mean()), "baseline_ang_err_median": float(np.median(b_ang))}
        if mv.sum() >= 5:
            m = mv
            mag_pred = np.hypot(*d_pred[m].T)
            cos = (d_pred[m] * d_true[m]).sum(1) / (mag_pred * mag_true[m] + 1e-9)
            r.update({
                "moving_pos_err_mean": float(e_pos[m].mean()),
                "moving_baseline_pos_err_mean": float(b_pos[m].mean()),
                "moving_ang_err_median": float(np.median(e_ang[m])),
                "moving_baseline_ang_err_median": float(np.median(b_ang[m])),
                # % sai số giảm so với "đoán không đổi" (100% = hoàn hảo, 0% = không hơn mốc, âm = tệ hơn)
                "moving_pos_gain_pct": float(100 * (1 - e_pos[m].mean() / max(b_pos[m].mean(), 1e-9))),
                "moving_dir_cos_median": float(np.median(cos[mag_true[m] > MOVE_POS])) if (mag_true[m] > MOVE_POS).any() else None,
                "moving_mag_ratio_median": float(np.median(mag_pred[mag_true[m] > MOVE_POS] /
                                                           mag_true[m][mag_true[m] > MOVE_POS]))
                if (mag_true[m] > MOVE_POS).any() else None})
        out[f"k{k}"] = r
    return out


def print_table(name: str, m: dict):
    print(f"\n### {name}")
    print(f"{'k':>3} | {'vị trí TB (dự đoán / không đổi)':>32} | {'góc trung vị °':>16} || "
          f"{'LÚC BỊ ĐẨY: vị trí (dự đoán / không đổi)':>40} | {'giảm sai số':>11} | {'cos hướng':>9} | n đẩy")
    for k, r in m.items():
        if "moving_pos_err_mean" in r:
            cos = r.get("moving_dir_cos_median")
            mv = (f"{r['moving_pos_err_mean']:8.2f} / {r['moving_baseline_pos_err_mean']:8.2f}            | "
                  f"{r['moving_pos_gain_pct']:10.1f}% | {('%.3f' % cos) if cos is not None else '-':>9}")
        else:
            mv = "(quá ít cửa sổ bị đẩy)"
        print(f"{k[1:]:>3} | {r['pos_err_mean']:13.2f} / {r['baseline_pos_err_mean']:13.2f}    | "
              f"{r['ang_err_median']:6.2f} / {r['baseline_ang_err_median']:6.2f}    || {mv} | {r['n_moving']}")


def self_test():
    """Kiểm tra thước đo: dự đoán đúng sự thật -> sai số 0; dự đoán 'không đổi' -> đúng bằng mốc."""
    rng = np.random.default_rng(0)
    cur = rng.uniform(100, 400, (200, 3))
    true = cur[:, None] + np.cumsum(rng.normal(0, 2, (200, 16, 3)), 1)
    m = object_metrics(true.copy(), true, cur)
    assert all(abs(r["pos_err_mean"]) < 1e-9 and r["ang_err_median"] < 1e-9 for r in m.values()), "dự đoán hoàn hảo phải sai số 0"
    m = object_metrics(np.repeat(cur[:, None], 16, 1), true, cur)
    assert all(abs(r["pos_err_mean"] - r["baseline_pos_err_mean"]) < 1e-9 for r in m.values()), "'không đổi' phải = mốc"
    assert all(abs(r.get("moving_pos_gain_pct", 0)) < 1e-6 for r in m.values()), "'không đổi' phải giảm 0%"
    return True


def eval_windows(episode_ends: np.ndarray, length: int, n: int, seed: int = 0) -> np.ndarray:
    """Chỉ số bắt đầu các cửa sổ [s, s+length) nằm trọn trong 1 episode, lấy ngẫu nhiên n cửa sổ (cố định seed).
    DÙNG CHUNG cho world model ảnh và world model trạng thái -> chấm trên cùng các cửa sổ."""
    starts = np.r_[0, episode_ends[:-1]]
    allw = np.concatenate([np.arange(s, e - length + 1) for s, e in zip(starts, episode_ends) if e - s >= length])
    return np.sort(np.random.default_rng(seed).choice(allw, min(n, len(allw)), replace=False))
