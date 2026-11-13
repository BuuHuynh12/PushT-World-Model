"""Bước 18: PHÂN TÍCH MÀN THẤT BẠI của mô hình chính (Diffusion sinh path + WM chọn).

    python scripts/analyze_failures.py --result results/policy/test_600000_main.json

Lấy các seed THUA trong file kết quả test, chạy lại chúng (cùng cấu hình) + cùng số seed THẮNG làm đối chứng, ghi ở MỖI lần
ra quyết định: trạng thái thật, coverage, sai số encoder đọc s_t. Rồi phân loại vì sao thua (chỉ dựa trên trạng thái thật):
  - "sát đích, không chốt"   : max coverage >= 0.90
  - "tụt sau khi gần xong"   : max coverage - coverage cuối >= 0.20 (đẩy lệch khối khỏi đích)
  - "sai góc"                : khối đúng chỗ (<= 20 đơn vị) nhưng lệch góc > 15°
  - "sai vị trí"             : góc đúng (<= 15°) nhưng khối cách đích > 20
  - "xa đích cả hai"         : cả vị trí lẫn góc đều sai
  phụ: "kẹt" = 100 bước cuối khối gần như đứng yên (dịch < 5 đơn vị, xoay < 3°); "chấm tròn xa khối" = tỉ lệ thời gian
  chấm tròn cách khối > 80 đơn vị.
Lưu: npz log, json tóm tắt, video vài màn thua (cùng thư mục --out).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from pusht_wm.collect import get_full_state
from pusht_wm.evaluate import evaluate, save_video
from pusht_wm.state_dp import StateDiffusionPolicy

GOAL = np.array([256.0, 256.0, np.pi / 4])


def ang_err(a):
    """|góc - góc đích| trong [0, π]."""
    return np.abs((a - GOAL[2] + np.pi) % (2 * np.pi) - np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="results/policy/test_600000_main.json", help="json kết quả test (để lấy seed thua/thắng)")
    ap.add_argument("--ckpt", default="models/path_policy.pt")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--feas-w", type=float, default=0.5)
    ap.add_argument("--n-control", type=int, default=None, help="số seed thắng làm đối chứng (mặc định = số seed thua)")
    ap.add_argument("--n-video", type=int, default=10)
    ap.add_argument("--sample-seed", type=int, default=0, help="seed của bộ sinh nhiễu Diffusion")
    ap.add_argument("--out", default="outputs/failures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per = json.load(open(args.result))["per_seed"]
    fail = [x["seed"] for x in per if not x["success"]]
    win = [x["seed"] for x in per if x["success"]]
    nc = len(fail) if args.n_control is None else args.n_control
    ctrl = list(np.random.default_rng(0).choice(win, nc, replace=False))
    seeds = fail + [int(s) for s in ctrl]                      # màn thua đứng trước -> video là màn thua
    print(f"{len(fail)} seed thua + {nc} seed thắng đối chứng")

    dev = torch.device("cuda")
    p = StateDiffusionPolicy.from_checkpoint(args.ckpt).to(dev).eval()
    p.cfg.feas_w = args.feas_w
    gen = torch.Generator(device=dev).manual_seed(args.sample_seed)
    LOG = {}                                                   # seed -> list các dòng [t, true5, cov, enc_err_xy, enc_err_deg]
    tcount = {}

    def act(images, states, prev, ids=None, envs=None):
        enc = p.read_state(images, states)                     # [B, 2, 5] encoder đọc
        a = p.act_from_state(enc, prev, gen, args.n_samples)
        for j, (i, e) in enumerate(zip(ids, envs)):
            s = seeds[int(i)]
            tru = get_full_state(e)
            cov = float(e.unwrapped._get_coverage())
            exy = float(np.linalg.norm(enc[j, -1, 2:4] - tru[2:4]))
            eth = float(np.degrees(np.abs((enc[j, -1, 4] - tru[4] + np.pi) % (2 * np.pi) - np.pi)))
            t = tcount.get(s, 0)
            LOG.setdefault(s, []).append([t, *tru, cov, exy, eth])
            tcount[s] = t + a.shape[1]
        return a
    act.wants_prev = True
    act.wants_ids = True

    t0 = time.time()
    r = evaluate(act_fn=act, seeds=seeds, n_obs=2, n_video=args.n_video)
    res = {x["seed"]: x for x in r["per_seed"]}
    print(f"chạy lại xong {time.time() - t0:.0f}s | thua tái hiện: {sum(not res[s]['success'] for s in fail)}/{len(fail)}"
          f" | thắng tái hiện: {sum(res[int(s)]['success'] for s in ctrl)}/{nc}")

    rows = []
    for s in seeds:
        L = np.array(LOG[s])                                   # cột: t, ax, ay, bx, by, th, cov, exy, eth
        t, bxy, th, cov = L[:, 0], L[:, 3:5], L[:, 5], L[:, 6]
        pe = np.linalg.norm(bxy - GOAL[:2], axis=1)
        ae = np.degrees(ang_err(th))
        last = t >= t[-1] - 100
        k0 = np.flatnonzero(last)[0]
        stuck = (np.linalg.norm(bxy[-1] - bxy[k0]) < 5) and (np.degrees(np.abs(
            (th[-1] - th[k0] + np.pi) % (2 * np.pi) - np.pi)) < 3)
        far = float(np.mean(np.linalg.norm(L[:, 1:3] - bxy, axis=1) > 80))
        mx = max(float(cov.max()), res[s]["max_coverage"])
        cf = float(cov[-1])
        if mx >= 0.90:
            cat = "sát đích, không chốt"
        elif mx - cf >= 0.20:
            cat = "tụt sau khi gần xong"
        elif pe[-1] <= 20 and ae[-1] > 15:
            cat = "sai góc"
        elif ae[-1] <= 15 and pe[-1] > 20:
            cat = "sai vị trí"
        elif pe[-1] <= 20 and ae[-1] <= 15:
            cat = "gần đích (coverage thấp)"
        else:
            cat = "xa đích cả hai"
        rows.append({"seed": int(s), "group": "thua" if s in fail else "thắng", "success_rerun": res[s]["success"],
                     "steps": res[s]["steps"], "max_cov": mx, "final_cov": cf, "final_pos_err": float(pe[-1]),
                     "final_ang_err_deg": float(ae[-1]), "stuck_last100": bool(stuck), "agent_far_frac": far,
                     "enc_err_xy_mean": float(L[:, 7].mean()), "enc_err_deg_mean": float(L[:, 8].mean()),
                     "enc_err_xy_max": float(L[:, 7].max()), "category": cat})

    F = [x for x in rows if x["group"] == "thua" and not x["success_rerun"]]
    W = [x for x in rows if x["group"] == "thắng" and x["success_rerun"]]
    print(f"\n== {len(F)} màn THUA (tái hiện) — phân loại:")
    cats = {}
    for x in F:
        cats.setdefault(x["category"], []).append(x["seed"])
    for c, v in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        print(f"  {c:28s} {len(v):3d} ({len(v) / max(len(F), 1):.0%})  vd seed {v[:5]}")

    def m(G, k):
        return np.mean([x[k] for x in G]) if G else float("nan")
    print("\n== so sánh THUA vs THẮNG (đối chứng):")
    for k, name in [("max_cov", "max coverage"), ("final_pos_err", "lệch vị trí cuối"),
                    ("final_ang_err_deg", "lệch góc cuối (°)"), ("agent_far_frac", "tỉ lệ t/g chấm tròn xa khối"),
                    ("enc_err_xy_mean", "encoder sai vị trí TB"), ("enc_err_deg_mean", "encoder sai góc TB (°)"),
                    ("enc_err_xy_max", "encoder sai vị trí MAX")]:
        print(f"  {name:30s} thua {m(F, k):8.3f} | thắng {m(W, k):8.3f}")
    print(f"  {'kẹt 100 bước cuối':30s} thua {np.mean([x['stuck_last100'] for x in F]):.0%}")

    (out / "failures.json").write_text(json.dumps({"rows": rows, "categories": cats}, indent=1, ensure_ascii=False))
    np.savez_compressed(out / "failure_logs.npz", **{str(s): np.array(v) for s, v in LOG.items()})
    for i, v in enumerate(r["videos"]):
        x = rows[i]
        kq = 'THUA_' + x['category'].replace(' ', '_').replace(',', '') if not x['success_rerun'] else 'lan_nay_THANG'
        save_video(v, out / f"fail_{seeds[i]}_{kq}.mp4")
    print(f"đã lưu {out}/failures.json, failure_logs.npz, {len(r['videos'])} video")


if __name__ == "__main__":
    main()
