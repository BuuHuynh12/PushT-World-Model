"""Đánh giá MÔ HÌNH CHÍNH trên tập seed mới, chạy TỪ ẢNH:
frame -> WM encoder đọc s_t -> Diffusion sinh 16 PATH -> WM dynamics chọn (+ phạt path phi thực tế) -> mô hình nghịch -> action.

    python scripts/eval_state_dp.py                                   # test 500 seed 600000..600499 (kết quả chốt: 89.8% / 143)
    python scripts/eval_state_dp.py --seed-start 700000               # tập TINH CHỈNH (88.8% / 142)
    python scripts/eval_state_dp.py --n-samples 1                     # đối chứng: 1 path, KHÔNG để WM chọn (86.0% / 175.5)
    python scripts/eval_state_dp.py --n-seeds 4 --n-video 4           # chạy thử + lưu video
    python scripts/eval_state_dp.py --starts wall --seed-start 800000 # khối T bắt đầu SÁT TƯỜNG/GÓC (ngoài phân phối)
    python scripts/eval_state_dp.py --max-steps 1000                 # cho chạy tới 1000 bước; in CẢ tỉ lệ thắng trong 300 bước

Seed: 100000.. chọn checkpoint lúc train, 600000.. TEST, 700000.. tinh chỉnh; sát tường: 800000.. TEST, 820000.. tinh chỉnh,
810000.. tự chơi. Thành công = coverage > 0.95 trong <= 300 bước.
Lưu json (tỉ lệ thành công, trung vị số bước, kết quả từng seed) vào --out.
(Bản đầy đủ có tùy chọn gộp s_t encoder + dynamics: archive/scripts/eval_state_dp_full.py)
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

import numpy as np
import torch

from pusht_wm.evaluate import evaluate, save_video
from pusht_wm.paths import PATH_POLICY
from pusht_wm.state_dp import StateDiffusionPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=PATH_POLICY)
    ap.add_argument("--seed-start", type=int, default=600000)
    ap.add_argument("--n-seeds", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=16, help="số path Diffusion đề xuất; >1: WM dynamics chọn path tốt nhất")
    ap.add_argument("--feas-w", type=float, default=0.5, help="phạt lệch giữa khối T trong path và khối T WM tưởng tượng")
    ap.add_argument("--edge-w", type=float, default=0.0, help="phạt WM tưởng tượng khối T bị đẩy về phía tường")
    ap.add_argument("--starts", default="standard", choices=["standard", "wall"],
                    help="vị trí ban đầu: chuẩn gym-pusht hoặc khối T sát tường/góc (pusht_wm.wall_starts)")
    ap.add_argument("--select-wm", default=None, help="checkpoint WM dynamics khác, CHỈ dùng để chọn path")
    ap.add_argument("--n-action-steps", type=int, default=None, help="số action thực thi mỗi lần nghĩ (mặc định 8)")
    ap.add_argument("--ddim-steps", type=int, default=None, help="số bước khử nhiễu DDIM (mặc định 10)")
    ap.add_argument("--sample-seed", type=int, default=0, help="seed bộ sinh nhiễu của Diffusion")
    ap.add_argument("--max-steps", type=int, default=300,
                    help="giới hạn số bước mỗi màn (300 = chuẩn gym-pusht/LeRobot; app không giới hạn). 300 bước đầu của "
                         "mỗi màn y hệt nhau dù giới hạn dài hơn -> từ 1 lần chạy tính được thắng trong 300 và trong max-steps")
    ap.add_argument("--n-video", type=int, default=0)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = StateDiffusionPolicy.from_checkpoint(args.ckpt).to(dev).eval()
    if args.n_action_steps:
        p.cfg.n_action_steps = args.n_action_steps
    if args.ddim_steps:
        p.cfg.num_inference_steps = args.ddim_steps
    p.cfg.feas_w, p.cfg.edge_w = args.feas_w, args.edge_w
    if args.select_wm:
        from pusht_wm.state_wm import StateWorldModel
        object.__setattr__(p, "select_wm", StateWorldModel.from_checkpoint(args.select_wm).to(dev).eval())
    gen = torch.Generator(device=dev).manual_seed(args.sample_seed)

    def act(images, states, prev):
        return p.act_from_images(images, states, prev, generator=gen, n_samples=args.n_samples)
    act.wants_prev = True

    t0 = time.time()
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    init = None
    if args.starts == "wall":
        from pusht_wm.wall_starts import wall_start
        init = [wall_start(s) for s in seeds]
    r = evaluate(act_fn=act, seeds=seeds, n_obs=2, n_video=args.n_video, init_states=init, max_steps=args.max_steps)
    succ = np.array([x["success"] for x in r["per_seed"]])
    st = [x["steps"] for x in r["per_seed"] if x["success"]]
    ci = 1.96 * np.sqrt(succ.mean() * (1 - succ.mean()) / len(succ))
    tag = args.tag or (f"{Path(args.ckpt).stem}_N{args.n_samples}_feas{args.feas_w:g}"
                       + (f"_edge{args.edge_w:g}" if args.edge_w else "")
                       + (f"_max{args.max_steps}" if args.max_steps != 300 else "") + ("_wall" if args.starts == "wall" else "")
                       + (f"_sel-{Path(args.select_wm).stem}" if args.select_wm else ""))
    res = {"ckpt": args.ckpt, "n_action_steps": p.cfg.n_action_steps, "ddim_steps": p.cfg.num_inference_steps,
           "max_steps": args.max_steps, "n_samples": args.n_samples, "feas_w": args.feas_w, "edge_w": args.edge_w, "starts": args.starts, "select_wm": args.select_wm, "seeds": [seeds[0], seeds[-1]],
           "success_rate": float(succ.mean()), "ci95": float(ci),
           "steps_to_success_median": float(np.median(st)) if st else None,
           "mean_max_coverage": r["mean_max_coverage"], "time_s": time.time() - t0, "per_seed": r["per_seed"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"result_{tag}_{seeds[0]}_{len(seeds)}.json").write_text(json.dumps(res))
    for i, v in enumerate(r["videos"]):
        save_video(v, out / f"video_{tag}_{seeds[i]}.mp4")
    print(f"[{tag}] {len(seeds)} seed từ {seeds[0]}: success {succ.mean():.3f} ±{ci:.3f} | thời gian tới đích "
          f"(trung vị) {res['steps_to_success_median']} | max_cov TB {r['mean_max_coverage']:.3f} | {res['time_s']:.0f}s")
    if args.max_steps > 300:      # cùng lần chạy: thắng TRONG 300 bước (chuẩn) để so với các kết quả khác
        s300 = np.array([x["success"] and x["steps"] <= 300 for x in r["per_seed"]])
        st300 = [x["steps"] for x in r["per_seed"] if x["success"] and x["steps"] <= 300]
        print(f"  trong 300 bước: success {s300.mean():.3f} (trung vị {np.median(st300) if st300 else None}) | "
              f"trong {args.max_steps} bước: success {succ.mean():.3f} (trung vị {res['steps_to_success_median']})")


if __name__ == "__main__":
    main()
