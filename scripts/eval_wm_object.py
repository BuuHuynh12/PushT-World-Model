"""Đánh giá WORLD MODEL ẢNH theo đúng đề bài: dự đoán sự thay đổi vị trí khối T.

    python scripts/eval_wm_object.py --wm-ckpt models/wm.pt --out outputs/wm_eval

Tập test: sim_v2 (world model CHƯA từng thấy; nhiều đoạn va chạm gần đích), CÙNG các cửa sổ với world model trạng thái.
Ngữ cảnh: 2 frame thật (t-1, t) + action đã làm ở t-1; rollout 16 action thật tiếp theo trong latent.
Tư thế khối T dự đoán = đầu đọc tư thế trên latent dự đoán. Tư thế thật = trạng thái của bộ mô phỏng.
Cũng báo "sàn" của đầu đọc: sai số đọc tư thế ngay trên latent THẬT của frame hiện tại (sai số khâu nhìn).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from pusht_wm.latent_data import LatentSeqData, LatentStats, data_file
from pusht_wm.object_metrics import eval_windows, object_metrics, pose_from_vec6, print_table, self_test, wrap
from pusht_wm.world_model import load_wm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm-ckpt", default="models/wm.pt")
    ap.add_argument("--dataset", default="sim_v2")
    ap.add_argument("--n-windows", type=int, default=1000000)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--out", default="outputs/wm_eval")
    args = ap.parse_args()
    assert self_test()
    dev = torch.device("cuda")
    wm, c = load_wm(args.wm_ckpt, dev)
    stats = LatentStats(c["latent_stats"]["mean"].to(dev), c["latent_stats"]["std"].to(dev))
    head = wm.pose
    if head is None:   # world model cũ: đầu đọc train riêng, lưu cạnh checkpoint
        from pusht_wm.wm_planner import PoseHead
        pc = torch.load(Path(args.wm_ckpt).parent / "pose_head.pt", map_location="cpu", weights_only=False)
        head = PoseHead(pc["P"], pc["D"]).to(dev).eval()
        head.load_state_dict(pc["state_dict"])

    data = LatentSeqData(c["full_config"]["encoder"], (args.dataset,), pool=c.get("pool", 0))
    sf = np.load(data_file(args.dataset))["state_full"]              # [N, 5] (chấm tròn x,y, khối x,y, góc)
    assert len(sf) == data.n_frames
    H, C = args.horizon, 2
    starts = eval_windows(data.episode_ends, C + H, args.n_windows)   # cùng cửa sổ với world model trạng thái
    print(f"world model bước {c.get('step')} | test {args.dataset}: {len(starts)} cửa sổ, nhìn trước {H} bước")

    preds, floor = [], []
    with torch.no_grad():
        for i in range(0, len(starts), 256):
            st = starts[i:i + 256]
            z, a, _ = data.get(st, C + H, stats)
            zp, _ = wm.rollout(z[:, :C], a[:, :C - 1], a[:, C - 1:C - 1 + H])
            preds.append(pose_from_vec6(head(zp.float()).cpu().numpy()))
            floor.append(pose_from_vec6(head(z[:, C - 1]).cpu().numpy()))   # đọc tư thế frame hiện tại (thật)
    pred, floor = np.concatenate(preds), np.concatenate(floor)
    idx = starts[:, None] + C - 1 + np.arange(1, H + 1)[None]           # frame t+1..t+H
    true = np.stack([sf[idx, 2], sf[idx, 3], sf[idx, 4]], -1)
    cur_i = starts + C - 1
    cur = np.stack([sf[cur_i, 2], sf[cur_i, 3], sf[cur_i, 4]], -1)
    fe_pos = np.hypot(floor[:, 0] - cur[:, 0], floor[:, 1] - cur[:, 1])
    fe_ang = np.degrees(np.abs(wrap(floor[:, 2] - cur[:, 2])))
    print(f"sàn đầu đọc (đọc tư thế trên latent thật): vị trí TB {fe_pos.mean():.2f} | góc trung vị {np.median(fe_ang):.2f}°")
    m = object_metrics(pred, true, cur, ks=[k for k in (1, 4, 8, 16) if k <= H])
    m["pose_head_floor"] = {"pos_err_mean": float(fe_pos.mean()), "ang_err_median": float(np.median(fe_ang))}
    print_table(f"World model ẢNH ({args.wm_ckpt}) trên {args.dataset}", {k: v for k, v in m.items() if k.startswith("k")})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"object_wm_image_{args.dataset}_H{H}.json").write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
