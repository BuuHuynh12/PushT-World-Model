"""Đánh giá mô hình (frame -> WM ảnh -> s_t latent -> Diffusion -> path -> action) trong simulator,
hoặc thu dữ liệu TỰ CHƠI cho vòng lặp train lại world model.

    python scripts/eval_latent_path.py --ckpt models/diffusion_path.pt                        # test 500 seed, 16 path + WM chọn
    python scripts/eval_latent_path.py --ckpt ... --n-samples 1                                       # đối chứng: 1 path, WM không chọn
    python scripts/eval_latent_path.py --ckpt ... --seed-start 910000 --n-seeds 1000 --record data/self_r1.npz   # tự chơi

Seed: 100000.. chọn checkpoint | 600000.. TEST | 9x0000.. tự chơi (không trùng nhau).
--record: lưu ảnh + vị trí chấm tròn + action + coverage + trạng thái thật của simulator (cùng định dạng sim_v1/sim_v2).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import torch

from pusht_wm.latent_path import LatentPathPolicy, rollout, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/diffusion_path.pt")
    ap.add_argument("--wm", default=None, help="world model dùng để đọc s_t và chọn path (mặc định: bản ghi trong checkpoint)")
    ap.add_argument("--seed-start", type=int, default=600000)
    ap.add_argument("--n-seeds", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--batch-envs", type=int, default=100)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--record", default=None, help="file .npz để lưu dữ liệu tự chơi")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = LatentPathPolicy.from_checkpoint(args.ckpt, wm_ckpt=args.wm, device=dev)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    t0 = time.time()
    per, raw = rollout(p, seeds, n_samples=args.n_samples, max_steps=args.max_steps, batch_envs=args.batch_envs,
                       sample_seed=args.sample_seed, record=args.record is not None)
    r = summarize(per)
    tag = args.tag or f"latent_path_N{args.n_samples}" + (f"_max{args.max_steps}" if args.max_steps != 300 else "")
    res = {"ckpt": args.ckpt, "wm": args.wm, "n_samples": args.n_samples, "max_steps": args.max_steps,
           "n_action_steps": p.cfg.n_action_steps, "ddim_steps": p.cfg.num_inference_steps, "seeds": [seeds[0], seeds[-1]],
           **r, "time_s": time.time() - t0, "per_seed": per}
    if raw is not None:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        raw.save(args.record)
        Path(args.record).with_suffix(".json").write_text(json.dumps({k: v for k, v in res.items() if k != "per_seed"}, indent=1))
        print(f"đã lưu {args.record}: {raw.n_episodes} episode, {len(raw.action)} frame")
    else:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"result_{tag}_{seeds[0]}_{len(seeds)}.json").write_text(json.dumps(res))
    print(f"[{tag}] {len(seeds)} seed từ {seeds[0]}: success {r['success_rate']:.3f} ±{r['ci95']:.3f} | thời gian tới đích "
          f"(trung vị) {r['steps_to_success_median']} | max_cov TB {r['mean_max_coverage']:.3f} | {res['time_s']:.0f}s")


if __name__ == "__main__":
    main()
