"""Train DIFFUSION đọc s_t latent của world model ảnh -> sinh PATH (quỹ đạo robot cần đi).

    python scripts/train_latent_path.py --wm models/wm.pt --extra sim_v2 self_r1 self_r2
    python scripts/train_latent_path.py --steps 300 --eval-every 300 --eval-seeds 4 --out outputs/debug     # chạy thử

World model (encoder DINOv2 + transformer) ĐÓNG BĂNG: chỉ bộ đọc token + U-Net được cập nhật. Loss DDPM chuẩn.
Dữ liệu: 206 demo lerobot/pusht + episode chính THÀNH CÔNG của các tập --extra (mặc định sim_v2). Chỉ cần latent đã
cache, vị trí chấm tròn và action -> KHÔNG dùng tọa độ khối T, không cần gán nhãn.
Chọn checkpoint theo success trên seed 100000.. (1 path, không world model chọn). Lưu: ckpt_best.pt, ckpt_latest.pt.
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

from pusht_wm.evaluate import EVAL_SEEDS
from pusht_wm.latent_data import data_file, pool_tokens
from pusht_wm.latent_path import LatentPathConfig, LatentPathPolicy, norm_pos, rollout, summarize
from pusht_wm.models import EMA
from pusht_wm.planner import WMScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="models/wm.pt")
    ap.add_argument("--feature", default="both", choices=["latent", "wm_hidden", "both"])
    ap.add_argument("--extra", nargs="*", default=["sim_v2"], help="thêm episode chính thành công của các tập sim")
    ap.add_argument("--no-agent-pos", action="store_true", help="không đưa vị trí chấm tròn vào điều kiện")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-seeds", type=int, default=50)
    ap.add_argument("--out", default="outputs/latent_path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    # ---------------- world model ảnh (đóng băng) + encoder online
    scorer = WMScorer(args.wm, device="cuda", amp=None)        # fp32: s_t lúc train và lúc chạy khớp nhau
    wm = scorer.model
    enc_name = torch.load(args.wm, map_location="cpu", weights_only=False)["full_config"]["encoder"]

    # ---------------- dữ liệu: latent đã cache (pool + chuẩn hóa như world model), vị trí chấm tròn, action
    H, rng = 16, np.random.default_rng(args.seed)
    Z, POS, ACT, OBS, PATH, off = [], [], [], [], [], 0
    for name in ["demo", *args.extra]:
        d = np.load(data_file(name))
        ends = d["episode_ends"]
        st = np.r_[0, ends[:-1]]
        if name == "demo":
            keep = list(range(len(ends)))
        else:   # chỉ episode chính (không phải nhánh "nếu chọn path khác", source = 5) và có thành công
            src, suc = d["source"], d["success"]
            keep = [i for i in range(len(ends)) if src[st[i]] != 5 and suc[st[i]:ends[i]].any()]
        lat = np.load(f"data/latents/{enc_name}/{name}.npy", mmap_mode="r")
        assert len(lat) == len(d["action"]), f"{name}: latent không khớp số frame"
        z = torch.from_numpy(pool_tokens(np.asarray(lat), scorer.pool))
        zn = torch.cat([((z[i:i + 16384].to(dev).float() - scorer.mean) / scorer.std).half() for i in range(0, len(z), 16384)])
        Z.append(zn), POS.append(d["state"].astype(np.float32)), ACT.append(d["action"].astype(np.float32))
        n0 = sum(len(o) for o in OBS)
        for i in keep:
            s, e = int(st[i]), int(ends[i])
            t = np.arange(s, e)
            OBS.append(np.stack([np.maximum(t - 1, s), t], 1) + off)
            PATH.append(np.minimum(t[:, None] + np.arange(1, H + 1)[None], e - 1) + off)   # cuối episode: đứng yên
        off += len(d["action"])
        print(f"{name}: {len(keep)}/{len(ends)} episode, {sum(len(o) for o in OBS) - n0} mẫu", flush=True)
    Z = torch.cat(Z)                                                         # [N, P, D] fp16 trên GPU
    POS_t = torch.from_numpy(np.concatenate(POS)).to(dev)
    ACT_t = torch.from_numpy(np.concatenate(ACT)).to(dev)
    OBS_t, PATH_t = torch.from_numpy(np.concatenate(OBS)).to(dev), torch.from_numpy(np.concatenate(PATH)).to(dev)
    n = len(OBS_t)

    P, D = Z.shape[1:]
    cfg = LatentPathConfig(feature=args.feature, P=P, D_in=LatentPathPolicy.feature_dim(args.feature, D, wm.cfg.d_model),
                           use_agent_pos=not args.no_agent_pos)
    policy = LatentPathPolicy(cfg).to(dev)
    ema = EMA(policy)
    print(f"s_t = {args.feature}: {P} token x {cfg.D_in} | {n} mẫu | tham số train "
          f"{sum(p.numel() for p in policy.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)

    def batch(ids):
        oi = OBS_t[ids]                                                      # [B, 2]
        z = Z[oi].float()                                                    # [B, 2, P, D]
        # action đã làm ở t-1; đầu episode (chưa có action nào) = vị trí hiện tại (đứng yên), như lúc chạy
        prev = torch.where((oi[:, 0] == oi[:, 1])[:, None], POS_t[oi[:, 1]], ACT_t[oi[:, 0]])
        feat = LatentPathPolicy.features(wm, cfg.feature, z, prev)
        return feat, norm_pos(POS_t[oi]), norm_pos(POS_t[PATH_t[ids]])

    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / args.warmup if s < args.warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, args.steps - args.warmup))))

    def save(name, step, best):
        torch.save(ema.model.checkpoint({"step": step, "best_success": best, "wm_ckpt": args.wm, "args": vars(args)}),
                   out / name)

    best, hist, t_log, log = -1.0, [], time.time(), []
    policy.train()
    for step in range(1, args.steps + 1):
        feat, pos_n, path_n = batch(torch.randint(0, n, (args.batch,), device=dev))
        loss = policy.compute_loss(feat, pos_n, path_n)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        ema.update(policy)
        hist.append(loss.item())
        if step % 200 == 0:
            print(f"step {step:6d} loss {np.mean(hist):.4f} lr {sched.get_last_lr()[0]:.1e} "
                  f"{200/(time.time()-t_log):.1f} it/s", flush=True)
            hist, t_log = [], time.time()
        if step % args.eval_every == 0 or step == args.steps:
            m = ema.model.eval()
            m.attach_scorer(scorer)
            t0 = time.time()
            per, _ = rollout(m, EVAL_SEEDS[:args.eval_seeds], n_samples=1, log=lambda *_: None)
            r = summarize(per)
            log.append({"step": step, **r})
            print(f"[eval {step}] success {r['success_rate']:.3f} | thời gian (trung vị) {r['steps_to_success_median']} "
                  f"| max_cov {r['mean_max_coverage']:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if r["success_rate"] > best:
                best = r["success_rate"]
                save("ckpt_best.pt", step, best)
            save("ckpt_latest.pt", step, best)
            (out / "eval_log.json").write_text(json.dumps(log, indent=1))
            t_log = time.time()
    print(f"xong. success tốt nhất ({args.eval_seeds} seed chọn checkpoint) = {best:.3f}")


if __name__ == "__main__":
    main()
