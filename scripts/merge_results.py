"""Gộp các file kết quả của eval_state_dp.py chạy CHIA NHỎ theo khoảng seed (chạy song song) thành một kết quả.

    python scripts/merge_results.py outputs/wall/eval/result_path_policy_N16_feas0.5_wall_820000_150.json \
        outputs/wall/eval/result_path_policy_N16_feas0.5_wall_820150_150.json --out outputs/wall/eval/merged_x.json
"""
import argparse
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    R = [json.load(open(f)) for f in a.files]
    per = sorted([x for r in R for x in r["per_seed"]], key=lambda x: x["seed"])
    seeds = [x["seed"] for x in per]
    assert len(set(seeds)) == len(seeds), "trùng seed giữa các phần"
    succ = np.array([x["success"] for x in per])
    st = [x["steps"] for x in per if x["success"]]
    ci = 1.96 * np.sqrt(succ.mean() * (1 - succ.mean()) / len(succ))
    out = {k: v for k, v in R[0].items() if k not in ("per_seed", "seeds", "success_rate", "ci95", "time_s",
                                                       "steps_to_success_median", "mean_max_coverage")}
    out.update(parts=a.files, seeds=[seeds[0], seeds[-1]], success_rate=float(succ.mean()), ci95=float(ci),
               steps_to_success_median=float(np.median(st)) if st else None,
               mean_max_coverage=float(np.mean([x["max_coverage"] for x in per])), per_seed=per)
    msg = f"[{a.name or a.out}] {len(per)} seed {seeds[0]}..{seeds[-1]}: success {succ.mean():.3f} ±{ci:.3f} | " \
          f"trung vị {out['steps_to_success_median']} | max_cov TB {out['mean_max_coverage']:.3f}"
    ms = R[0].get("max_steps", 300)
    if ms > 300:
        s300 = np.array([x["success"] and x["steps"] <= 300 for x in per])
        st3 = [x["steps"] for x in per if x["success"] and x["steps"] <= 300]
        msg += f"\n  trong 300 bước: success {s300.mean():.3f} (trung vị {np.median(st3) if st3 else None}) | " \
               f"trong {ms} bước: success {succ.mean():.3f}"
    json.dump(out, open(a.out, "w"))
    print(msg)


if __name__ == "__main__":
    main()
