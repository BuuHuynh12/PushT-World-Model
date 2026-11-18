"""Kiểm tra bước PATH -> ACTION (thuần toán) với simulator thật.

    python scripts/check_path_math.py

1) Mô hình tuyến tính của bộ điều khiển PD: từ (p_{k-1}, p_k, a_{k-1}, a_k) đoán p_{k+1}, so với simulator
   (action ngẫu nhiên, có cả lúc đang đẩy khối T).
2) Giải ngược: từ quỹ đạo robot đã ghi, khôi phục lại action đã dùng.
3) Bám path: lấy quỹ đạo robot của một màn làm PATH, đổi ra action bằng path_to_actions (mỗi lần 16 điểm, thực thi 8,
   như lúc chạy thật) rồi cho robot đi lại trong simulator -> robot lệch path bao nhiêu.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from pusht_wm.latent_path import PD_G, PD_M, path_to_actions
from pusht_wm.sim import make_env


def run_random(env, seed, n=200):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    P, A = [obs["agent_pos"].astype(np.float64)], []
    goal = P[0]
    for t in range(n):
        if t % 6 == 0:      # đổi đích mỗi 6 bước; một nửa số lần nhắm vào khối T để có va chạm
            blk = np.array(env.unwrapped.block.position)
            goal = blk + rng.normal(0, 30, 2) if rng.random() < 0.5 else rng.uniform(30, 480, 2)
        a = np.clip(goal + rng.normal(0, 5, 2), 0, 512).astype(np.float32)
        obs, *_ = env.step(a)
        A.append(a.astype(np.float64)), P.append(obs["agent_pos"].astype(np.float64))
    return np.array(P), np.array(A)


def main():
    (m00, m01), (m10, m11) = PD_M
    g0, g1 = PD_G
    print(f"hệ số một action: p' = {m00:.4f} p + {m01:.4f} v + {g0:.4f} a ; v' = {m10:.4f} p + {m11:.4f} v + {g1:.4f} a")
    env = make_env()
    e_fwd, e_inv = [], []
    for seed in range(20):
        P, A = run_random(env, seed)                       # P[k] trước A[k]; P[k+1] sau A[k]
        for k in range(1, len(A)):
            v0 = (P[k] - m00 * P[k - 1] - g0 * A[k - 1]) / m01
            v = m10 * P[k - 1] + m11 * v0 + g1 * A[k - 1]
            e_fwd.append(np.linalg.norm(m00 * P[k] + m01 * v + g0 * A[k] - P[k + 1]))
            e_inv.append(np.linalg.norm((P[k + 1] - m00 * P[k] - m01 * v) / g0 - A[k]))
    e_fwd, e_inv = np.array(e_fwd), np.array(e_inv)
    print(f"1) đoán vị trí kế tiếp: sai số TB {e_fwd.mean():.4f}, lớn nhất {e_fwd.max():.4f} (đơn vị sim, {len(e_fwd)} bước)")
    print(f"2) khôi phục action:    sai số TB {e_inv.mean():.4f}, lớn nhất {e_inv.max():.4f}")

    # 3) bám path theo kiểu chạy thật: nghĩ 16 điểm, thực thi 8, nghĩ lại
    errs = []
    for seed in range(20, 30):
        P, _ = run_random(env, seed)                       # quỹ đạo tham chiếu = PATH
        obs, _ = env.reset(seed=seed)
        pos = [obs["agent_pos"].astype(np.float64)] * 2
        prev = pos[-1].copy()
        t = 0
        while t + 16 < len(P):
            path = torch.from_numpy(P[t + 1:t + 17])[None]
            acts = path_to_actions(torch.from_numpy(np.stack(pos))[None], torch.from_numpy(prev)[None], path)[0].numpy()
            for a in acts[:8]:
                obs, *_ = env.step(a.astype(np.float32))
                t += 1
                prev = a
                pos = [pos[-1], obs["agent_pos"].astype(np.float64)]
                errs.append(np.linalg.norm(pos[-1] - P[t]))
    errs = np.array(errs)
    print(f"3) bám path (nghĩ 16, thực thi 8): robot lệch path TB {errs.mean():.4f}, p95 {np.quantile(errs, 0.95):.4f}, "
          f"lớn nhất {errs.max():.4f} (đơn vị sim, {len(errs)} bước)")


if __name__ == "__main__":
    main()
