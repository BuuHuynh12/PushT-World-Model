"""Bước 15: WM có SUY LUẬN đúng quan hệ nhân quả không? (thử phản thực tế, không phải đo độ chính xác trên dữ liệu)

    python scripts/wm_counterfactual.py --n 2000

Mỗi kịch bản: khối T ở tư thế ngẫu nhiên, chấm tròn đứng yên cách khối 50-90 đơn vị, rồi đi thẳng 12 bước
(~8 đơn vị/bước) về một điểm nhắm. Điểm nhắm = tâm khối T + lệch ngẫu nhiên (đẩy lệch tâm -> xoay), hoặc
cố tình nhắm RA XA khối T (đẩy hụt -> khối phải đứng yên). Các kịch bản này KHÔNG có trong dữ liệu train
(bắt đầu đứng yên, đường thẳng, góc tiếp cận tùy ý).
So "tưởng tượng" của WM với bộ mô phỏng thật:
  - đẩy trúng (khối T thật dịch > 3 hoặc xoay > 3°): chiều XOAY có đúng không, HƯỚNG dịch chuyển (cos), độ lớn
  - đẩy hụt (khối T thật gần như đứng yên): WM có dự đoán đứng yên không
  - "ai thắng": với 2 cách đẩy cùng tư thế, WM có xếp đúng cách nào đưa khối T gần đích hơn không
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from pusht_wm.sim import make_env
from pusht_wm.state_wm import StateWorldModel, encode_state

STEPS, SPEED = 12, 8.0


def scenario(rng):
    bx, by, ang = rng.uniform(170, 340), rng.uniform(170, 340), rng.uniform(-math.pi, math.pi)
    th = rng.uniform(-math.pi, math.pi)
    r0 = rng.uniform(50, 90)
    ax, ay = bx + r0 * math.cos(th), by + r0 * math.sin(th)
    miss = rng.random() < 0.3
    if miss:   # nhắm ra xa: đi theo hướng ngược với khối T, lệch tối đa 60°
        d = th + rng.uniform(-1, 1) * math.radians(60)
        tx, ty = ax + 100 * math.cos(d), ay + 100 * math.sin(d)
    else:      # nhắm vào khối T, lệch tâm ngẫu nhiên tới 35 đơn vị
        tx, ty = bx + rng.uniform(-35, 35), by + rng.uniform(-35, 35)
    v = np.array([tx - ax, ty - ay])
    v = v / (np.linalg.norm(v) + 1e-9)
    acts = np.array([[ax, ay]]) + SPEED * np.arange(1, STEPS + 1)[:, None] * v[None]
    return np.array([ax, ay, bx, by, ang]), np.clip(acts, 5, 507).astype(np.float32), miss


def simulate(env, s0, acts):
    u = env.unwrapped
    u.agent.velocity, u.block.velocity, u.block.angular_velocity = (0, 0), (0, 0), 0
    u.block.angle = float(s0[4])                      # góc TRƯỚC vị trí (khối T có tâm khối lệch)
    u.block.position = (float(s0[2]), float(s0[3]))
    u.agent.position = (float(s0[0]), float(s0[1]))
    traj = []
    for a in acts:
        env.step(a)
        traj.append([*u.agent.position, *u.block.position, u.block.angle])
    return np.array(traj)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--wm", nargs="+", default=["models/state_wm.pt"])
    ap.add_argument("--out", default="outputs/counterfactual.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    env = make_env(obs_type="state")
    env.reset(seed=0)
    S0, A, M, T = [], [], [], []
    for _ in range(args.n):
        s0, acts, miss = scenario(rng)
        tr = simulate(env, s0, acts)
        S0.append(s0), A.append(acts), M.append(miss), T.append(tr)
    S0, A, M, T = np.array(S0), np.array(A), np.array(M), np.array(T)
    true_d = T[:, -1, 2:4] - S0[:, 2:4]
    true_r = wrap(T[:, -1, 4] - S0[:, 4])
    moved = (np.linalg.norm(true_d, axis=1) > 3) | (np.abs(np.degrees(true_r)) > 3)
    still = (np.linalg.norm(true_d, axis=1) < 0.5) & (np.abs(np.degrees(true_r)) < 0.5)
    rot = moved & (np.abs(np.degrees(true_r)) > 5)
    print(f"{args.n} kịch bản: khối T bị đẩy {moved.sum()} | đứng yên {still.sum()} (trong đó nhắm hụt {(still & M).sum()})")

    res = {"n": args.n, "n_moved": int(moved.sum()), "n_still": int(still.sum()), "n_rot": int(rot.sum())}
    for path in args.wm:
        wm = StateWorldModel.from_checkpoint(path).eval()
        s = torch.from_numpy(encode_state(S0).astype(np.float32))
        with torch.no_grad():
            pr = wm.rollout(s, s, torch.from_numpy(A / 256.0 - 1.0).float()).numpy()
        pb = (pr[:, -1, 2:4] + 1) * 256
        pa = np.arctan2(pr[:, -1, 4], pr[:, -1, 5])
        pd, prr = pb - S0[:, 2:4], wrap(pa - S0[:, 4])
        cos = (pd * true_d).sum(1) / (np.linalg.norm(pd, axis=1) * np.linalg.norm(true_d, axis=1) + 1e-9)
        sign_ok = np.sign(prr) == np.sign(true_r)
        mag_err = np.linalg.norm(pd - true_d, axis=1)
        nochange = np.linalg.norm(true_d, axis=1)
        pred_still = (np.linalg.norm(pd, axis=1) < 2) & (np.abs(np.degrees(prr)) < 2)
        # "ai thắng": ghép cặp kịch bản đẩy trúng, so khoảng cách khối T tới tư thế ban đầu + đích (256,256)
        goal = np.array([256.0, 256.0])
        dt = np.linalg.norm(T[:, -1, 2:4] - goal, axis=1)
        dp = np.linalg.norm(pb - goal, axis=1)
        idx = np.flatnonzero(moved)
        pairs = rng.choice(idx, (2000, 2))
        diff = np.abs(dt[pairs[:, 0]] - dt[pairs[:, 1]]) > 5
        rank_ok = (np.sign(dt[pairs[:, 0]] - dt[pairs[:, 1]]) == np.sign(dp[pairs[:, 0]] - dp[pairs[:, 1]]))[diff]
        r = {"rot_sign_acc": float(sign_ok[rot].mean()),
             "move_dir_cos_median": float(np.median(cos[moved])), "move_dir_cos_gt0.9": float(np.mean(cos[moved] > 0.9)),
             "pos_err_moved_mean": float(mag_err[moved].mean()), "nochange_err_moved_mean": float(nochange[moved].mean()),
             "rot_err_moved_deg_mean": float(np.degrees(np.abs(wrap(prr - true_r)))[moved].mean()),
             "still_pred_still_rate": float(pred_still[still].mean()),
             "rank_acc": float(rank_ok.mean()), "rank_n": int(diff.sum())}
        res[Path(path).parent.name] = r
        print(f"\n== {Path(path).parent.name}")
        print(f"  chiều xoay đúng (xoay > 5°, n={rot.sum()}): {r['rot_sign_acc']:.1%}")
        print(f"  hướng dịch chuyển: cos trung vị {r['move_dir_cos_median']:.3f} | cos > 0.9: {r['move_dir_cos_gt0.9']:.1%}")
        print(f"  sai số vị trí sau {STEPS} bước: {r['pos_err_moved_mean']:.2f} (đoán 'đứng yên': {r['nochange_err_moved_mean']:.2f})"
              f" | sai số góc {r['rot_err_moved_deg_mean']:.2f}°")
        print(f"  đẩy hụt -> WM đoán đứng yên: {r['still_pred_still_rate']:.1%} (n={still.sum()})")
        print(f"  so sánh 2 cách đẩy, cách nào đưa khối T gần đích hơn: đúng {r['rank_acc']:.1%} (n={r['rank_n']})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
