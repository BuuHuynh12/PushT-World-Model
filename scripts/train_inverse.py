"""Bước 14a: train MÔ HÌNH NGHỊCH (s_{k-1}, s_k, s_{k+1}) -> a_k (dùng để biến PATH của Diffusion thành action).

    python scripts/train_inverse.py
    python scripts/train_inverse.py --steps 500 --out outputs/debug     # chạy thử

Dữ liệu có action thật: sim_v1 + sim_v2 (trạng thái thật) + 206 demo (chấm tròn thật, khối T do encoder gán nhãn).
Tách val theo episode (10%). Báo sai số action theo đơn vị sim, và sai số "tái tạo": đưa action dự đoán vào
bộ mô phỏng thật, chấm tròn có tới đúng s_{k+1} không (đó mới là điều quan trọng khi bám path).
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

from pusht_wm.inverse import InverseConfig, InverseModel
from pusht_wm.state_wm import encode_state


def load(name):
    if name == "demo":
        d, r = np.load("data/demo_state.npz"), np.load("data/pusht_raw.npz")
        return d["state_full"], r["action"], d["episode_ends"]
    d = np.load(f"data/{name}.npz")
    return d["state_full"], d["action"], d["episode_ends"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["sim_v1", "sim_v2", "demo"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="outputs/inverse")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    S, A, tr_k, va_k, va_src = [], [], [], [], []
    off = 0
    for di, name in enumerate(args.data):
        sf, act, ends = load(name)
        st = np.r_[0, ends[:-1]]
        va = set(rng.permutation(len(ends))[: max(1, len(ends) // 10)].tolist())
        for i, (s, e) in enumerate(zip(st, ends)):
            if e - s < 3:
                continue
            k = np.arange(s + 1, e - 1) + off          # cần k-1, k, k+1 trong cùng episode
            (va_k if i in va else tr_k).append(k)
            if i in va:
                va_src.append(np.full(len(k), di))
        S.append(encode_state(sf).astype(np.float32)), A.append((act / 256.0 - 1.0).astype(np.float32))
        off += len(sf)
        print(f"{name}: {len(ends)} episode, {len(sf)} frame", flush=True)
    S_t, A_t = torch.from_numpy(np.concatenate(S)).to(dev), torch.from_numpy(np.concatenate(A)).to(dev)
    tr_k, va_k, va_src = np.concatenate(tr_k), np.concatenate(va_k), np.concatenate(va_src)
    print(f"train {len(tr_k)} / val {len(va_k)} mẫu", flush=True)

    model = InverseModel(InverseConfig()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)

    def err(k):
        """sai số action (đơn vị sim) trên các chỉ số k."""
        k = torch.from_numpy(k).to(dev)
        with torch.no_grad():
            p = model(S_t[k - 1], S_t[k], S_t[k + 1])
        return ((p - A_t[k]).norm(dim=-1) * 256).cpu().numpy()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        k = torch.from_numpy(rng.choice(tr_k, args.batch)).to(dev)
        loss = (model(S_t[k - 1], S_t[k], S_t[k + 1]) - A_t[k]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if step % 2000 == 0 or step == args.steps:
            model.eval()
            e = err(rng.choice(va_k, 20000))
            model.train()
            print(f"step {step:6d} loss {loss.item():.4f} | val sai số action TB {e.mean():.2f} trung vị "
                  f"{np.median(e):.2f} p90 {np.percentile(e, 90):.2f} | {time.time()-t0:.0f}s", flush=True)
    model.eval()
    torch.save(model.checkpoint({"args": vars(args)}), out / "inverse.pt")

    rep = {}
    for di, name in enumerate(args.data):
        e = np.concatenate([err(c) for c in np.array_split(va_k[va_src == di], 20)])
        rep[name] = {"n": int(len(e)), "mean": float(e.mean()), "median": float(np.median(e)),
                     "p90": float(np.percentile(e, 90))}
        print(f"[val {name}] sai số action: TB {e.mean():.2f} | trung vị {np.median(e):.2f} | p90 "
              f"{np.percentile(e, 90):.2f} (đơn vị sim, sân 512)", flush=True)

    # ---- kiểm tra trong bộ mô phỏng: đặt env về đúng trạng thái + vận tốc ở k, ra lệnh action dự đoán,
    # so vị trí chấm tròn sau 1 bước với s_{k+1} thật (chỉ dữ liệu sim có trạng thái thật)
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from pusht_wm.sim import make_env
    env = make_env(obs_type="state")
    env.reset(seed=0)
    u = env.unwrapped
    sf_all = np.concatenate([load(n)[0] for n in args.data])
    pick = rng.choice(va_k[va_src == 0], 300, replace=False)
    pos_err_pred, pos_err_true = [], []
    for k in pick:
        for use_pred in (True, False):
            s_prev, s_cur, s_next = sf_all[k - 1], sf_all[k], sf_all[k + 1]
            u.block.angle = float(s_cur[4])
            u.block.position = (float(s_cur[2]), float(s_cur[3]))
            u.block.velocity, u.block.angular_velocity = (0, 0), 0
            u.agent.position = (float(s_cur[0]), float(s_cur[1]))
            u.agent.velocity = tuple((s_cur[:2] - s_prev[:2]) * 10.0)     # 10 Hz
            if use_pred:
                with torch.no_grad():
                    kk = torch.tensor([k], device=dev)
                    a = ((model(S_t[kk - 1], S_t[kk], S_t[kk + 1])[0].cpu().numpy() + 1) * 256).astype(np.float32)
            else:
                a = ((A_t[k].cpu().numpy() + 1) * 256).astype(np.float32)
            env.step(a)
            d = float(np.hypot(*(np.array(u.agent.position) - s_next[:2])))
            (pos_err_pred if use_pred else pos_err_true).append(d)
    rep["sim_check"] = {"agent_err_pred_action_mean": float(np.mean(pos_err_pred)),
                        "agent_err_true_action_mean": float(np.mean(pos_err_true))}
    print(f"[mô phỏng] chấm tròn sau 1 bước lệch s_k+1: action dự đoán {np.mean(pos_err_pred):.2f} | "
          f"action thật {np.mean(pos_err_true):.2f} (mức nền do vận tốc ước lượng)", flush=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"đã lưu {out}")


if __name__ == "__main__":
    main()
