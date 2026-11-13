"""Train + đánh giá WORLD MODEL TRẠNG THÁI (so với world model ảnh, cùng thước đo, cùng cửa sổ test).

    python scripts/train_state_wm.py
    python scripts/train_state_wm.py --steps 500 --out outputs/debug     # chạy thử

Train: sim_v1 (có trạng thái 5D thật), tách val theo episode. Test: sim_v2 (chưa thấy), giống world model ảnh.
Loss: rollout K bước (MSE trạng thái), cửa sổ khối T bị đẩy được nhân trọng số w_move (tập trung vào tương tác).
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

from pusht_wm.object_metrics import eval_windows, object_metrics, print_table, self_test
from pusht_wm.state_dp import path_fde
from pusht_wm.state_wm import StateWMConfig, StateWorldModel, decode_block_pose, encode_state


def load(name):
    d = np.load(f"data/{name}.npz")
    return encode_state(d["state_full"]), (d["action"].astype(np.float32) / 256.0 - 1.0), d["episode_ends"], d["state_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k-train", type=int, default=8)
    ap.add_argument("--w-move", type=float, default=3.0, help="trọng số cửa sổ khối T bị đẩy")
    ap.add_argument("--fde-w", type=float, default=0.0,
                    help="thêm FDE khối T ở bước cuối: sqrt(dx^2+dy^2) + 60|dθ| (đơn vị sim, chia 256) vào loss")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=6000)
    ap.add_argument("--data", nargs="+", default=["sim_v1"], help="dữ liệu train (có trạng thái thật), vd: sim_v1 self_r1")
    ap.add_argument("--compare", default=None, help="checkpoint WM cũ: chấm trên CÙNG cửa sổ val để so sánh")
    ap.add_argument("--out", default="outputs/state_wm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert self_test()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ghép nhiều nguồn; nhớ episode nào thuộc nguồn nào (để báo sai số val riêng từng nguồn)
    S, A, ends, src_ep = [], [], [], []
    off = 0
    for di, name in enumerate(args.data):
        s_, a_, e_, _ = load(name)
        S.append(s_), A.append(a_), ends.append(e_ + off), src_ep.append(np.full(len(e_), di))
        off += len(s_)
    S, A, ends, src_ep = np.concatenate(S), np.concatenate(A), np.concatenate(ends), np.concatenate(src_ep)
    S_t, A_t = torch.from_numpy(S).to(dev), torch.from_numpy(A).to(dev)
    rng = np.random.default_rng(args.seed)
    n_ep = len(ends)
    va = set(rng.permutation(n_ep)[: n_ep // 10].tolist())
    K, L = args.k_train, 2 + args.k_train
    st_ep = np.r_[0, ends[:-1]]
    tr_w = np.concatenate([np.arange(s, e - L + 1) for i, (s, e) in enumerate(zip(st_ep, ends)) if i not in va and e - s >= L])
    va_w = np.concatenate([np.arange(s, e - L + 1) for i, (s, e) in enumerate(zip(st_ep, ends)) if i in va and e - s >= L])
    va_src = np.concatenate([np.full(max(0, e - s - L + 1), src_ep[i]) for i, (s, e) in enumerate(zip(st_ep, ends)) if i in va and e - s >= L])
    print(f"{'+'.join(args.data)}: {len(tr_w)} cửa sổ train / {len(va_w)} val | thiết bị {dev}")

    def batch(w):
        w = torch.from_numpy(w).to(dev)
        idx = w[:, None] + torch.arange(L, device=dev)[None]
        s = S_t[idx]                               # [B, L, 6]
        return s[:, 0], s[:, 1], A_t[idx[:, 1:1 + K]], s[:, 2:2 + K]

    model = StateWorldModel(StateWMConfig(args.hidden, args.layers)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    t0 = time.time()
    val_err = None
    for step in range(1, args.steps + 1):
        sp, sc, a, tgt = batch(rng.choice(tr_w, args.batch))
        pred = model.rollout(sp, sc, a)
        err = ((pred - tgt) ** 2).mean((1, 2))
        moving = ((tgt[:, -1, 2:4] - sc[:, 2:4]).norm(dim=-1) * 256 > 2) | \
                 ((tgt[:, -1, 4:6] - sc[:, 4:6]).norm(dim=-1) > np.radians(1))
        w = 1 + (args.w_move - 1) * moving.float()
        loss = (err * w).mean()
        if args.fde_w > 0:
            loss = loss + args.fde_w * (path_fde(pred, tgt) * w).mean() / 256.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if step % 2000 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                sp, sc, a, tgt = batch(rng.choice(va_w, 4096))
                p = model.rollout(sp, sc, a)
                e_blk = ((p[:, -1, 2:4] - tgt[:, -1, 2:4]).norm(dim=-1) * 256).mean().item()
                e_none = ((sc[:, 2:4] - tgt[:, -1, 2:4]).norm(dim=-1) * 256).mean().item()
                val_err = e_blk
                fde_v = path_fde(p, tgt).mean().item()
            model.train()
            print(f"step {step:6d} loss {loss.item():.2e} | val khối T sau {K} bước: {e_blk:.2f} "
                  f"(không đổi: {e_none:.2f}) | FDE {fde_v:.2f} | {time.time()-t0:.0f}s", flush=True)
    torch.save(model.checkpoint({"args": vars(args)}), out / "state_wm.pt")

    # ---------------- val cố định (toàn bộ cửa sổ val), riêng từng nguồn; so với WM cũ nếu có
    def val_full(m):
        m.eval()
        r = {}
        with torch.no_grad():
            for di, name in enumerate(args.data):
                w = va_w[va_src == di]
                if len(w) == 0:
                    continue
                e_all, e_mv, e_none_mv = [], [], []
                for i in range(0, len(w), 8192):
                    sp, sc, a, tgt = batch(w[i:i + 8192])
                    pr = m.rollout(sp, sc, a)
                    e = (pr[:, -1, 2:4] - tgt[:, -1, 2:4]).norm(dim=-1) * 256
                    e0 = (sc[:, 2:4] - tgt[:, -1, 2:4]).norm(dim=-1) * 256
                    mv = e0 > 2
                    e_all.append(e), e_mv.append(e[mv]), e_none_mv.append(e0[mv])
                e_all, e_mv, e_none_mv = (torch.cat(x).cpu().numpy() for x in (e_all, e_mv, e_none_mv))
                r[name] = {"n": int(len(w)), "block_err_mean": float(e_all.mean()), "moving_n": int(len(e_mv)),
                           "moving_block_err_mean": float(e_mv.mean()) if len(e_mv) else None,
                           "moving_nochange_err_mean": float(e_none_mv.mean()) if len(e_mv) else None}
        return r
    vr = {"new": val_full(model)}
    if args.compare:
        vr["old"] = val_full(StateWorldModel.from_checkpoint(args.compare).to(dev))
    for k_, r in vr.items():
        for name, x in r.items():
            print(f"[val {k_}] {name}: khối T sau {K} bước TB {x['block_err_mean']:.2f} | lúc bị đẩy "
                  f"{x['moving_block_err_mean']} (không đổi: {x['moving_nochange_err_mean']}) | {x['n']} cửa sổ", flush=True)
    (out / "val_by_source.json").write_text(json.dumps(vr, indent=1))
    # dùng để CHỌN cấu hình (theo val của sim_v1, không nhìn tập test sim_v2)
    (out / "val.json").write_text(json.dumps({"val_block_err_k": val_err, "k": K, "args": vars(args)}, indent=1))

    # ---------------- test trên sim_v2, CÙNG cửa sổ với world model ảnh
    model.eval()
    S2, A2, ends2, sf2 = load("sim_v2")
    S2_t, A2_t = torch.from_numpy(S2).to(dev), torch.from_numpy(A2).to(dev)
    res = {}
    for H in (16, 8):
        starts = eval_windows(ends2, 2 + H, args.n_test)
        with torch.no_grad():
            w = torch.from_numpy(starts).to(dev)
            p = model.rollout(S2_t[w], S2_t[w + 1], A2_t[w[:, None] + 1 + torch.arange(H, device=dev)[None]])
        pred = decode_block_pose(p.cpu().numpy())
        idx = starts[:, None] + 1 + np.arange(1, H + 1)[None]
        true = np.stack([sf2[idx, 2], sf2[idx, 3], sf2[idx, 4]], -1)
        cur = np.stack([sf2[starts + 1, 2], sf2[starts + 1, 3], sf2[starts + 1, 4]], -1)
        m = object_metrics(pred, true, cur, ks=[k for k in (1, 4, 8, 16) if k <= H])
        print_table(f"World model TRẠNG THÁI trên sim_v2 (nhìn trước {H}, {len(starts)} cửa sổ)", m)
        res[f"H{H}"] = m
    (out / "object_state_wm_sim_v2.json").write_text(json.dumps(res, indent=1))
    print(f"đã lưu {out}")


if __name__ == "__main__":
    main()
