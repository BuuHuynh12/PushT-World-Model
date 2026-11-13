"""Thu dữ liệu TỰ CHƠI bằng mô hình chính (chạy từ ảnh, như lúc đánh giá) — để tự bắt chước có lọc.

    python scripts/collect_selfplay.py --starts wall --seed-start 810000 --n-seeds 1000 --out outputs/wall/self_0.npz
    python scripts/collect_selfplay.py --merge "outputs/wall/self_*.npz" --out data/wall_self.npz      # ghép các phần

--starts wall: khối T bắt đầu SÁT TƯỜNG/GÓC (pusht_wm.wall_starts). Seed tự chơi KHÔNG trùng tập chọn checkpoint
(100000..), test (600000.., 800000..) hay tinh chỉnh (700000.., 820000..).
Lưu (cùng định dạng khóa với sim_v2, KHÔNG có ảnh vì train policy/WM chỉ cần trạng thái):
  state_full [N,5] trạng thái THẬT của bộ mô phỏng trước mỗi action, action [N,2], success [N] (bool, sau action),
  reward [N] (coverage sau action), episode_ends, source (=4: episode chính), episode_seed.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from pusht_wm.collect import get_full_state
from pusht_wm.sim import make_env
from pusht_wm.state_dp import StateDiffusionPolicy


def merge(paths, out):
    ds = [np.load(p) for p in sorted(paths)]
    keys = ("state_full", "action", "success", "reward", "source")
    o = {k: np.concatenate([d[k] for d in ds]) for k in keys}
    lens = np.concatenate([np.diff(np.r_[0, d["episode_ends"]]) for d in ds])
    o["episode_ends"] = np.cumsum(lens).astype(np.int64)
    o["episode_seed"] = np.concatenate([d["episode_seed"] for d in ds])
    np.savez_compressed(out, **o)
    st = np.r_[0, o["episode_ends"][:-1]]
    suc = np.array([o["success"][s:e].any() for s, e in zip(st, o["episode_ends"])])
    print(f"ghép {len(ds)} phần -> {out}: {len(lens)} episode, {len(o['action'])} frame, success {suc.mean():.3f}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/path_policy.pt")
    ap.add_argument("--seed-start", type=int, default=810000)
    ap.add_argument("--n-seeds", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--feas-w", type=float, default=0.5, help="phạt path phi thực tế khi WM chọn")
    ap.add_argument("--edge-w", type=float, default=0.0, help="phạt WM tưởng tượng khối T bị đẩy về phía tường")
    ap.add_argument("--starts", default="wall", choices=["standard", "wall"])
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.merge:
        paths = [p for g in args.merge for p in glob.glob(g)]
        return merge(paths, args.out)
    dev = torch.device("cuda")
    p = StateDiffusionPolicy.from_checkpoint(args.ckpt).to(dev).eval()
    p.cfg.feas_w, p.cfg.edge_w = args.feas_w, args.edge_w
    gen = torch.Generator(device=dev).manual_seed(args.seed_start)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    n = len(seeds)
    envs = [make_env() for _ in seeds]
    rec = [{"full": [], "act": [], "suc": [], "rew": []} for _ in seeds]
    img, pos, prev = [], [], np.zeros((n, 2), np.float32)
    for i, (env, s) in enumerate(zip(envs, seeds)):
        if args.starts == "wall":
            from pusht_wm.wall_starts import wall_start
            obs, _ = env.reset(seed=s, options={"reset_to_state": wall_start(s)})
        else:
            obs, _ = env.reset(seed=s)
        img.append([obs["pixels"]] * 2), pos.append([obs["agent_pos"]] * 2)
        prev[i] = obs["agent_pos"]
    done = np.zeros(n, bool)
    t0 = time.time()
    while not done.all():
        act = np.flatnonzero(~done)
        acts = p.act_from_images(np.stack([np.stack(img[i]) for i in act]),
                                 np.stack([np.stack(pos[i]) for i in act]).astype(np.float32), prev[act],
                                 generator=gen, n_samples=args.n_samples)
        for j, i in enumerate(act):
            R = rec[i]
            for a in acts[j]:
                R["full"].append(get_full_state(envs[i]))
                obs, _, term, _, info = envs[i].step(a.astype(np.float32))
                R["act"].append(a), R["suc"].append(bool(info["is_success"])), R["rew"].append(float(info["coverage"]))
                prev[i] = a
                img[i].append(obs["pixels"]), pos[i].append(obs["agent_pos"])
                if term or len(R["act"]) >= args.max_steps:
                    done[i] = True
                    break
            img[i], pos[i] = img[i][-2:], pos[i][-2:]
    for e in envs:
        e.close()
    lens = np.array([len(R["act"]) for R in rec])
    cat = lambda k, dt: np.concatenate([np.asarray(R[k], dt) for R in rec])  # noqa: E731
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, state_full=cat("full", np.float64), action=cat("act", np.float32),
                        success=cat("suc", bool), reward=cat("rew", np.float32),
                        source=np.full(lens.sum(), 4, np.int8), episode_ends=np.cumsum(lens).astype(np.int64),
                        episode_seed=np.array(seeds, np.int64))
    suc = np.array([any(R["suc"]) for R in rec])
    st = [l for l, s in zip(lens, suc) if s]
    meta = {"args": vars(args), "success": float(suc.mean()), "steps_median": float(np.median(st)) if st else None,
            "frames": int(lens.sum()), "time_s": time.time() - t0}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"{out}: {n} episode, success {suc.mean():.3f}, trung vị {meta['steps_median']}, {lens.sum()} frame, "
          f"{meta['time_s']:.0f}s")


if __name__ == "__main__":
    main()
