"""Train WORLD MODEL ẢNH trên latent đã cache (loss latent + coverage + FDE tư thế khối T).

    python scripts/train_wm.py --config configs/wm_image.yaml
    python scripts/train_wm.py --config configs/wm_image.yaml --set train.steps=500 out_dir=outputs/debug

Val (episode riêng, cửa sổ cố định): rollout 16 bước từ 2 frame thật, so với mốc "copy frame cuối".
Tư thế khối T cho FDE lấy từ trạng thái simulator ghi sẵn trong dữ liệu sim (demo lerobot/pusht không có -> chỉ loss latent).
Lưu: config.yaml, wm_best.pt (theo FDE khối T sau 16 bước trên val; không có tư thế thì theo MSE latent), wm_last.pt, tb/.
Checkpoint chứa: trọng số, WMConfig, thống kê chuẩn hóa latent, tên encoder + đường dẫn encoder.pt.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from pusht_wm.config import load_config, save_config
from pusht_wm.data import train_val_split
from pusht_wm.latent_data import LatentSeqData
from pusht_wm.world_model import LatentWorldModel, WMConfig, pose_fde, wm_checkpoint, wm_losses


@torch.no_grad()
def validate(model, data, starts, stats, n_ctx, H, amp_dtype, fde_r=60.0):
    """Rollout H bước -> MSE latent theo từng bước, mốc copy, sai số + tương quan reward."""
    model.eval()
    mse, mse_copy, rp_all, rt_all, fde, fde_copy = [], [], [], [], [], []
    for i in range(0, len(starts), 128):
        z, a, r = data.get(starts[i:i + 128], n_ctx + H, stats)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            zp, rp = model.rollout(z[:, :n_ctx], a[:, :n_ctx - 1], a[:, n_ctx - 1:n_ctx - 1 + H])
            pp = model.pose(zp[:, -1].float()) if model.pose is not None else None
        if pp is not None:   # FDE khối T ở bước cuối (đơn vị sim), so với mốc "đoán không đổi"
            pose, has = data.get_pose(starts[i:i + 128], n_ctx + H)
            m = has[:, -1]
            fde.append(pose_fde(pp, pose[:, -1], fde_r)[m])
            fde_copy.append(pose_fde(pose[:, n_ctx - 1], pose[:, -1], fde_r)[m])
        zt = z[:, n_ctx:n_ctx + H]
        mse.append(((zp.float() - zt) ** 2).mean((2, 3)))
        mse_copy.append(((z[:, n_ctx - 1:n_ctx] - zt) ** 2).mean((2, 3)))
        rp_all.append(rp.float())
        rt_all.append(r[:, n_ctx - 1:n_ctx - 1 + H])
    mse, mse_copy = torch.cat(mse).mean(0), torch.cat(mse_copy).mean(0)
    rp, rt = torch.cat(rp_all), torch.cat(rt_all)
    corr = torch.corrcoef(torch.stack([rp.flatten(), rt.flatten()]))[0, 1].item()
    model.train()
    out = {"mse": mse.cpu().numpy(), "mse_copy": mse_copy.cpu().numpy(),
           "r_mae": (rp - rt).abs().mean(0).cpu().numpy(), "r_corr": corr}
    if fde and sum(len(f) for f in fde) > 0:
        out["fde"], out["fde_copy"] = float(torch.cat(fde).mean()), float(torch.cat(fde_copy).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wm_image.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tc = cfg["train"]
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = torch.device("cuda")
    if tc.get("tf32", True):  # tensor core cho phép nhân fp32 (GPU Ampere trở lên); tắt: --set train.tf32=false
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[tc["amp"]]

    t0 = time.time()
    data = LatentSeqData(cfg["encoder"], cfg["datasets"], pool=cfg.get("pool", 0))
    tr, va = train_val_split(data.n_episodes, cfg["val_ratio"], seed=cfg["seed"])
    init = tc.get("init")
    if init:  # train tiếp: giữ NGUYÊN thống kê chuẩn hóa của checkpoint cũ (latent phải cùng thang đo)
        from pusht_wm.latent_data import LatentStats
        ck0 = torch.load(init, map_location="cpu", weights_only=False)
        stats = LatentStats(ck0["latent_stats"]["mean"].to(dev), ck0["latent_stats"]["std"].to(dev))
    else:
        stats = data.fit_stats(tr)
    print(f"latent {cfg['encoder']}: {data.n_frames} frame, {data.P} token x {data.D} | "
          f"{'GPU' if data.on_gpu else 'CPU pinned'} | train {len(tr)} ep / val {len(va)} ep ({time.time()-t0:.0f}s)")
    C, K, H, W = tc["n_ctx"], tc["n_roll"], tc["eval_horizon"], cfg["model"]["window"]
    L = max(W + 1, C + K)
    tr_starts = data.window_starts(tr, L)
    va_all = data.window_starts(va, C + H)
    va_starts = np.sort(np.random.default_rng(0).choice(va_all, min(tc["val_windows"], len(va_all)), replace=False))

    model = LatentWorldModel(WMConfig(P=data.P, D=data.D, **cfg["model"])).to(dev)
    if init:
        model.load_state_dict(ck0["state_dict"])
        print(f"khởi tạo từ {init} (bước {ck0.get('step')})")
    print(f"tham số world model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | cửa sổ train {L} frame")
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "pos" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tc["lr"], betas=(0.9, 0.95))
    warm, total = tc["warmup_steps"], tc["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 *
                                              (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total - warm)))))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(out / "tb"))
    enc_path = f"data/latents/{cfg['encoder']}/encoder.pt"
    # nhúng luôn cấu hình encoder + PCA vào checkpoint -> chạy online không cần thư mục data/latents
    enc_meta = {"encoder": cfg["encoder"], "encoder_path": enc_path,
                "encoder_state": torch.load(enc_path, map_location="cpu", weights_only=False),
                "pool": cfg.get("pool", 0),  # latent lúc chạy phải pool y hệt
                "full_config": cfg}

    def report(step):
        v = validate(model, data, va_starts, stats, C, H, amp_dtype, tc.get("fde_r", 60.0))
        if "fde" in v:
            writer.add_scalar("val/fde", v["fde"], step)
            print(f"[val {step}] FDE khối T sau {H} bước: {v['fde']:.2f} (đoán không đổi: {v['fde_copy']:.2f})", flush=True)
        ks = [k for k in (1, 4, 8, 16) if k <= H]
        for k in ks:
            writer.add_scalar(f"val/mse_k{k}", v["mse"][k - 1], step)
            writer.add_scalar(f"val/mse_copy_k{k}", v["mse_copy"][k - 1], step)
            writer.add_scalar(f"val/r_mae_k{k}", v["r_mae"][k - 1], step)
        writer.add_scalar("val/r_corr", v["r_corr"], step)
        print(f"[val {step}] MSE latent (WM | copy) " +
              " ".join(f"k{k}: {v['mse'][k-1]:.3f}|{v['mse_copy'][k-1]:.3f}" for k in ks) +
              f" | reward MAE k1 {v['r_mae'][0]:.4f} k{H} {v['r_mae'][-1]:.4f} corr {v['r_corr']:.3f}", flush=True)
        return v

    rng = np.random.default_rng(cfg["seed"])
    best, hist, t_log = float("inf"), [], time.time()
    model.train()
    for step in range(1, total + 1):
        st = rng.choice(tr_starts, tc["batch_size"])
        z, a, r = data.get(st, L, stats)
        pose, has = data.get_pose(st, L)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, parts = wm_losses(model, z, a, r, C, K, tc["w_reward"], tc.get("w_reward_hi", 0.0),
                                    pose=pose, has_pose=has, w_fde=tc.get("w_fde", 0.0), w_pose=tc.get("w_pose", 0.0),
                                    fde_r=tc.get("fde_r", 60.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tc["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        sched.step()
        hist.append(parts)
        if step % tc["log_every"] == 0:
            m = {k: np.mean([h[k] for h in hist]) for k in hist[0]}
            for k, v in m.items():
                writer.add_scalar(f"train/{k}", v, step)
            dt = time.time() - t_log
            print(f"step {step:6d} tf_z {m['tf_z']:.4f} ro_z {m['ro_z']:.4f} tf_r {m['tf_r']:.5f} "
                  f"ro_r {m['ro_r']:.5f} " + (f"fde {m['fde']:.2f} đọc {m['pose_read']:.4f} " if "fde" in m else "") +
                  f"lr {sched.get_last_lr()[0]:.1e} {tc['log_every']/dt:.1f} it/s", flush=True)
            hist, t_log = [], time.time()
        if step % tc["val_every"] == 0 or step == total:
            v = report(step)
            score = v["fde"] if "fde" in v else float(np.mean(v["mse"]))   # chọn checkpoint theo FDE (đúng thước đo đề bài)
            ck = wm_checkpoint(model, stats, {**enc_meta, "step": step, "val": {k: (x.tolist() if hasattr(x, "tolist") else x)
                                                                                  for k, x in v.items()}})
            torch.save(ck, out / "wm_last.pt")
            if score < best:
                best = score
                torch.save(ck, out / "wm_best.pt")
            (out / "val_last.json").write_text(json.dumps(ck["val"], indent=1))
            t_log = time.time()
    writer.close()
    print(f"xong. điểm val tốt nhất (FDE khối T sau {H} bước, hoặc MSE latent nếu không có tư thế) = {best:.4f}")


if __name__ == "__main__":
    main()
