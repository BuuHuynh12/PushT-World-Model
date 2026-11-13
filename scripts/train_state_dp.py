"""Train Diffusion của MÔ HÌNH CHÍNH: s_t (WM encoder đọc từ ảnh) -> 16 PATH trạng thái tương lai.
Mặc định = đúng cấu hình models/path_policy.pt (feature state, output path, + sim_v2, 20k bước, batch 1024, lr 2e-4).
WM encoder + WM dynamics + mô hình nghịch ĐÓNG BĂNG; loss MSE nhiễu (DDPM), không FDE.

    python scripts/train_state_dp.py                                                   # train lại mô hình chính
    python scripts/train_state_dp.py --steps 300 --eval-every 300 --eval-seeds 5 --out outputs/debug  # chạy thử
    (các biến thể cũ: --feature wm/wm_pred, --output action, --pred x0 --fde-w ... — xem archive/README_nhat_ky_day_du.md)

Dữ liệu: 206 demo, trạng thái khối T do encoder gán nhãn (data/demo_state.npz, từ train_pose_net.py);
tách train/val theo episode như baseline (seed 42, 5%).
--extra sim_v2: thêm các episode CHÍNH (không phải nhánh) THÀNH CÔNG của sim_v2 (trạng thái thật).
Đánh giá trong lúc train: chạy TỪ ẢNH (encoder đọc khối T), 50 seed 100000.. -> chọn checkpoint.
Lưu: ckpt_best.pt (theo success, hòa thì nhanh hơn), ckpt_latest.pt, log.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from pusht_wm.data import MinMaxNormalizer, train_val_split
from pusht_wm.evaluate import EVAL_SEEDS, evaluate
from pusht_wm.models import EMA
from pusht_wm.pose_net import PoseNet
from pusht_wm.state_dp import StateDiffusionPolicy, StateDPConfig, add_state_noise, enc_state_t
from pusht_wm.state_wm import StateWorldModel


def windows(ends: np.ndarray, episodes, n_obs: int, horizon: int, offset: int = 0):
    """Giống PushTWindowDataset: mỗi frame t một mẫu; obs t-n_obs+1..t, action chunk bắt đầu ở frame obs cũ nhất.
    -> obs_idx [M, n_obs], act_idx [M, H], first [M] (True nếu frame t-1 nằm ngoài episode)."""
    st = np.r_[0, ends[:-1]]
    O, A, F = [], [], []
    for ep in episodes:
        s, e = int(st[ep]), int(ends[ep])
        t = np.arange(s, e)
        t0 = t - n_obs + 1
        O.append(np.clip(t0[:, None] + np.arange(n_obs)[None], s, e - 1))
        A.append(np.clip(t0[:, None] + np.arange(horizon)[None], s, e - 1))
        F.append(t0 < s)
    return np.concatenate(O) + offset, np.concatenate(A) + offset, np.concatenate(F)


def make_act_fn(policy, gen):
    def act(images, states, prev):
        return policy.act_from_images(images, states, prev, generator=gen)
    act.wants_prev = True
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", default="state", choices=["state", "wm", "wm_pred"])
    ap.add_argument("--output", default="path", choices=["action", "path"],
                    help="path: Diffusion sinh đường đi trạng thái, mô hình nghịch (--inverse) suy ra action")
    ap.add_argument("--inverse", default="models/inverse.pt")
    ap.add_argument("--pred", default="eps", choices=["eps", "x0"], help="x0: U-Net đoán thẳng path, loss = MSE + fde_w*FDE")
    ap.add_argument("--fde-w", type=float, default=0.0)
    ap.add_argument("--extra", nargs="*", default=["sim_v2"], help="thêm episode chính thành công của: sim_v2 / sim_v1")
    ap.add_argument("--extra-max-steps", nargs="*", type=int, default=[],
                    help="LỌC theo từng nguồn --extra (cùng thứ tự): chỉ giữ episode thành công TRONG <= N bước "
                         "(0 = không lọc). Dùng cho dữ liệu tự chơi: chỉ bắt chước các lần đẩy NHANH (tự bắt chước có lọc)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--eval-seeds", type=int, default=50)
    ap.add_argument("--state-noise-pos", type=float, default=1.0)
    ap.add_argument("--state-noise-deg", type=float, default=0.5)
    ap.add_argument("--state-wm", default="models/state_wm.pt")
    ap.add_argument("--pose-net", default="models/pose_net.pt")
    ap.add_argument("--out", default="outputs/path_policy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", type=int, default=1, help="1 = train bằng bf16 (tensor core, nhanh ~2-3x); chạy/đánh giá vẫn fp32")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    # ---------------- dữ liệu
    raw = np.load("data/pusht_raw.npz")             # chỉ đọc action + episode_ends (KHÔNG nạp ảnh: tiết kiệm RAM)
    raw_action, raw_ends = raw["action"], raw["episode_ends"]
    lab = np.load("data/demo_state.npz")
    assert len(lab["state_full"]) == len(raw_action), "nhãn demo không khớp số frame"
    tr, va = train_val_split(len(raw_ends), 0.05, seed=args.seed)
    SF, ACT = [lab["state_full"].astype(np.float32)], [raw_action.astype(np.float32)]
    O, A, F = windows(raw_ends, tr, 2, 16)
    Ov, Av, Fv = windows(raw_ends, va, 2, 16)          # 5% demo để riêng: theo dõi loss val (phát hiện học thuộc)
    O, A, F = [O], [A], [F]
    off = len(raw_action)
    for j, name in enumerate(args.extra):
        d = np.load(f"data/{name}.npz")
        ends, src, suc = d["episode_ends"], d["source"], d["success"]
        st = np.r_[0, ends[:-1]]
        keep = [i for i in range(len(ends)) if src[st[i]] != 5 and suc[st[i]:ends[i]].any()]   # 5 = nhánh
        mx = args.extra_max_steps[j] if j < len(args.extra_max_steps) else 0
        if mx > 0:      # số bước tới lần thành công ĐẦU TIÊN <= mx
            n_all = len(keep)
            keep = [i for i in keep if int(np.argmax(suc[st[i]:ends[i]])) + 1 <= mx]
            print(f"  lọc {name}: giữ {len(keep)}/{n_all} episode thành công trong <= {mx} bước", flush=True)
        o, a, f = windows(ends, keep, 2, 16, offset=off)
        O.append(o), A.append(a), F.append(f)
        SF.append(d["state_full"].astype(np.float32)), ACT.append(d["action"].astype(np.float32))
        off += len(d["action"])
        print(f"+ {name}: {len(keep)} episode thành công, {len(o)} mẫu", flush=True)
    SF, ACT = np.concatenate(SF), np.concatenate(ACT)
    O, A, F = np.concatenate(O), np.concatenate(A), np.concatenate(F)
    action_norm = MinMaxNormalizer.fit(ACT[np.unique(A)])
    sf_t = torch.from_numpy(SF).to(dev)
    act_t = torch.from_numpy(ACT).to(dev)
    act_n = torch.from_numpy(action_norm.normalize(ACT).astype(np.float32)).to(dev)
    O_t, A_t, F_t = (torch.from_numpy(x).to(dev) for x in (O, A, F))
    vsel = np.random.default_rng(0).permutation(len(Ov))[:4096]
    Ov_t, Av_t, Fv_t = (torch.from_numpy(x[vsel]).to(dev) for x in (Ov, Av, Fv))
    print(f"train {len(tr)} demo + extra = {len(O)} mẫu | feature={args.feature}", flush=True)

    # ---------------- mô hình
    wm = StateWorldModel.from_checkpoint(args.state_wm).to(dev)
    pn = PoseNet()
    pn.load_state_dict(torch.load(args.pose_net, map_location="cpu", weights_only=False)["state_dict"])
    cfg = StateDPConfig(feature=args.feature, state_wm_ckpt=args.state_wm, pose_net_ckpt=args.pose_net,
                        wm_hidden=wm.cfg.hidden, state_noise_pos=args.state_noise_pos,
                        state_noise_deg=args.state_noise_deg, output=args.output, inverse_ckpt=args.inverse,
                        pred=args.pred, fde_w=args.fde_w)
    inv = None
    if args.output == "path":
        from pusht_wm.inverse import InverseModel
        inv = InverseModel.from_checkpoint(args.inverse).to(dev)
    policy = StateDiffusionPolicy(cfg, action_norm).to(dev).attach(wm, pn.to(dev), inv)
    ema = EMA(policy)
    ema.model.attach(wm, pn, inv)
    print(f"tham số policy: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / args.warmup if s < args.warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - args.warmup) /
                                                                  max(1, args.steps - args.warmup))))

    def batch(ids):
        oi, ai, first = O_t[ids], A_t[ids], F_t[ids]
        sf = add_state_noise(sf_t[oi], args.state_noise_pos, args.state_noise_deg)
        prev = torch.where(first[:, None], sf_t[oi[:, 0], :2], act_t[oi[:, 0]])   # đầu episode: đứng yên
        tgt = enc_state_t(sf_t[ai]) if args.output == "path" else act_n[ai]     # path: đường đi trạng thái THẬT/nhãn
        return sf, prev, tgt

    @torch.no_grad()
    def val_loss(model, n_rep=4):
        """loss khuếch tán trên demo val, nhiễu + bước t CỐ ĐỊNH (seed 0) -> so sánh được giữa các lần đo."""
        st = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.manual_seed(0)
        sf = sf_t[Ov_t]                                   # KHÔNG thêm nhiễu trạng thái
        prev = torch.where(Fv_t[:, None], sf_t[Ov_t[:, 0], :2], act_t[Ov_t[:, 0]])
        tgt = enc_state_t(sf_t[Av_t]) if args.output == "path" else act_n[Av_t]
        tot = sum(model.compute_loss(sf, prev, tgt).item() for _ in range(n_rep)) / n_rep
        torch.set_rng_state(st[0]), torch.cuda.set_rng_state(st[1])
        return tot

    @torch.no_grad()
    def val_fde(model, n=1024, n_multi=256, k=16):
        """(chỉ output=path) sinh path cho cửa sổ demo val, so với path thật (nhãn):
        FDE khối T ở bước cuối (đơn vị sim, gồm R*|dθ|), ADE vị trí khối T, minFDE trong k mẫu."""
        from pusht_wm.state_dp import path_fde
        g = torch.Generator(device=dev).manual_seed(0)
        sf = sf_t[Ov_t[:n]]
        prev = torch.where(Fv_t[:n, None], sf_t[Ov_t[:n, 0], :2], act_t[Ov_t[:n, 0]])
        true = enc_state_t(sf_t[Av_t[:n]])
        x = model.sample_chunk(sf, prev, generator=g)
        fde = path_fde(x, true, model.cfg.fde_r)
        ade = ((x[:, 2:, 2:4] - true[:, 2:, 2:4]).norm(dim=-1) * 256).mean(-1)
        m = n_multi
        xm = model.sample_chunk(sf[:m], prev[:m], n_samples=k, generator=g)
        fm = path_fde(xm, true[:m].repeat_interleave(k, 0), model.cfg.fde_r).view(m, k).min(1).values
        return {"fde": fde.mean().item(), "ade": ade.mean().item(), f"min{k}fde": fm.mean().item()}

    def save(name, step, best):
        torch.save(ema.model.checkpoint({"step": step, "best": best, "args": vars(args)}), out / name)

    best, hist, t_log, log = (-1.0, 0.0), [], time.time(), []
    for step in range(1, args.steps + 1):
        sf, prev, ac = batch(torch.from_numpy(rng.integers(0, len(O), args.batch)).to(dev))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.bf16)):
            loss = policy.compute_loss(sf, prev, ac)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        ema.update(policy)
        hist.append(loss.item())
        if step % 1000 == 0:
            print(f"step {step:6d} loss {np.mean(hist):.4f} lr {sched.get_last_lr()[0]:.1e} "
                  f"{1000/(time.time()-t_log):.1f} it/s", flush=True)
            hist, t_log = [], time.time()
        if step % args.eval_every == 0 or step == args.steps:
            t0 = time.time()
            r = evaluate(act_fn=make_act_fn(ema.model, torch.Generator(device=dev).manual_seed(0)),
                         seeds=EVAL_SEEDS[:args.eval_seeds], n_obs=2)
            st_ = [p["steps"] for p in r["per_seed"] if p["success"]]
            med = float(np.median(st_)) if st_ else 999.0
            vl = val_loss(ema.model)
            vf = val_fde(ema.model) if args.output == "path" else {}
            log.append({"step": step, "success": r["success_rate"], "steps_median": med,
                        "max_cov": r["mean_max_coverage"], "val_loss": vl, **vf})
            if vf:
                print(f"[val path {step}] FDE {vf['fde']:.2f} | ADE {vf['ade']:.2f} | minFDE(16) {vf['min16fde']:.2f} "
                      f"(đơn vị sim; FDE = vị trí cuối + 60·|Δθ|)", flush=True)
            print(f"[eval {step}] success {r['success_rate']:.3f} | thời gian {med} | max_cov "
                  f"{r['mean_max_coverage']:.3f} | val_loss {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if (r["success_rate"], -med) > (best[0], -best[1]):
                best = (r["success_rate"], med)
                save("ckpt_best.pt", step, best)
            (out / "eval_log.json").write_text(json.dumps(log, indent=1))
            t_log = time.time()
    save("ckpt_latest.pt", args.steps, best)
    print(f"xong. best (50 seed): success {best[0]:.3f}, thời gian {best[1]}")


if __name__ == "__main__":
    main()
