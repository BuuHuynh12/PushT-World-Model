"""Bước 12a: train ENCODER ảnh -> tư thế khối T, rồi GÁN NHÃN trạng thái cho 206 demo.

    python scripts/train_pose_net.py                        # train + đánh giá + gán nhãn demo
    python scripts/train_pose_net.py --steps 300 --n-per-file 4000 --out outputs/debug   # chạy thử

Dữ liệu: ảnh sim_v1 + sim_v2 có nhãn thật (state_full); tách val theo EPISODE.
Augment màu/nhiễu/mờ (KHÔNG dịch/xoay ảnh vì sẽ làm sai nhãn) -> bớt lệch giữa ảnh sim và ảnh demo (video nén).
Kiểm tra:
  - val sim: sai số vị trí / góc khối T
  - demo (không có nhãn): render lại ảnh từ (agent thật, khối T dự đoán) và so mask khối T với ảnh demo (IoU);
    so với cùng chỉ số trên ảnh sim val -> nếu tương đương thì lệch miền không đáng kể.
Lưu: <out>/pose_net.pt, <out>/report.json, data/demo_state.npz (state_full [N, 5] ước lượng cho demo).
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
import torch.nn.functional as F

from pusht_wm.data import get_pusht
from pusht_wm.pose_net import PoseNet, pose_errors, pose_target
from pusht_wm.sim import _block_mask, _iou, make_env


def read_rows(path, key, idx, chunk=2048):
    """Đọc các hàng idx (đã sắp xếp) của mảng `key` trong file .npz NÉN mà không nạp cả mảng vào RAM
    (ảnh sim ~4 GB/file; máy thuê dùng chung có thể chỉ còn vài GB RAM trống)."""
    import zipfile
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        ver = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = rd(f)
        per = int(np.prod(shape[1:])) * dtype.itemsize
        out = np.empty((len(idx), *shape[1:]), dtype)
        j = 0
        for s in range(0, shape[0], chunk):
            n = min(chunk, shape[0] - s)
            buf = np.frombuffer(f.read(n * per), dtype).reshape(n, *shape[1:])
            k = np.searchsorted(idx, s + n)
            out[j:k] = buf[idx[j:k] - s]
            j = k
    assert j == len(idx)
    return out


def load_sim(name, n, rng):
    d = np.load(f"data/{name}.npz")
    ends = d["episode_ends"]
    ep = np.searchsorted(ends, np.arange(ends[-1]), side="right")
    idx = np.sort(rng.choice(len(ep), min(n, len(ep)), replace=False))
    return read_rows(f"data/{name}.npz", "images", idx), d["state_full"][idx], ep[idx]


def augment(x):
    """x float [B, 3, H, W] trong [-1, 1]: sáng/tương phản, nhiễu, mờ nhẹ (mô phỏng nén video)."""
    B = x.shape[0]
    g = 1 + 0.1 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 2
    b = 0.08 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 2
    x = x * g + b
    blur = torch.rand(B, device=x.device) < 0.5
    if blur.any():
        k = torch.tensor([1., 2., 1.], device=x.device)
        k = (k[:, None] * k[None]) / 16
        xb = F.conv2d(F.pad(x[blur], (1, 1, 1, 1), mode="replicate"), k.expand(3, 1, 3, 3), groups=3)
        x[blur] = xb
    return (x + 0.03 * torch.randn_like(x)).clamp(-1, 1)


def render_state(env, agent_xy, pose3):
    """Render ảnh 96x96 đúng trạng thái. KHÔNG dùng reset_to_state của gym-pusht: nó gán vị trí TRƯỚC góc,
    khối T có trọng tâm lệch nên bị dịch (kiểm tra: IoU nhãn thật chỉ 0.22). Gán GÓC trước rồi mới VỊ TRÍ."""
    u = env.unwrapped
    u.agent.position = (float(agent_xy[0]), float(agent_xy[1]))
    u.block.angle = float(pose3[2])
    u.block.position = (float(pose3[0]), float(pose3[1]))
    return u._render(visualize=False)


def render_iou(env, imgs, agent_xy, pose3):
    env.reset(seed=0)
    return np.array([_iou(_block_mask(im), _block_mask(render_state(env, a, p))) for im, a, p in zip(imgs, agent_xy, pose3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-per-file", type=int, default=80000, help="số ảnh lấy từ mỗi file sim (giới hạn RAM)")
    ap.add_argument("--out", default="outputs/pose_net")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imgs, sfs, groups = [], [], []
    for gi, name in enumerate(("sim_v1", "sim_v2")):
        im, sf, ep = load_sim(name, args.n_per_file, rng)
        imgs.append(im), sfs.append(sf), groups.append(ep + gi * 10_000_000)
        print(f"{name}: {len(im)} ảnh", flush=True)
    imgs, sfs, groups = np.concatenate(imgs), np.concatenate(sfs), np.concatenate(groups)
    ug = np.unique(groups)
    va_g = set(rng.choice(ug, len(ug) // 20, replace=False).tolist())
    is_va = np.array([g in va_g for g in groups])
    tr_i, va_i = np.flatnonzero(~is_va), np.flatnonzero(is_va)
    print(f"train {len(tr_i)} / val {len(va_i)} ảnh (tách theo episode)", flush=True)
    imgs_t = torch.from_numpy(imgs)          # không pin: ~6 GB, máy thuê có thể ít RAM
    y_t = torch.from_numpy(pose_target(sfs))

    net = PoseNet().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    t0 = time.time()
    for step in range(1, args.steps + 1):
        b = torch.from_numpy(rng.choice(tr_i, args.batch))
        img = imgs_t[b].to(dev, non_blocking=True)
        y = y_t[b].to(dev)
        x = augment(img.permute(0, 3, 1, 2).float() / 127.5 - 1.0)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            o = net.net(x.contiguous(memory_format=torch.channels_last)).float()
        sc = o[:, 2:4] / o[:, 2:4].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # vị trí: L1 theo đơn vị chuẩn hóa (1 đơn vị sim = 1/256); góc: L1 trên sin/cos
        loss = (o[:, :2] - y[:, :2]).abs().mean() * 10 + (sc - y[:, 2:]).abs().mean() + 0.01 * (o[:, 2:4].norm(dim=-1) - 1).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if step % 1000 == 0 or step == args.steps:
            net.eval()
            vi = va_i[:20000]
            p = net.block_pose(imgs[vi], dev)
            e_pos, e_ang = pose_errors(p, sfs[vi])
            net.train()
            print(f"step {step:6d} loss {loss.item():.4f} | val: vị trí TB {e_pos.mean():.2f} trung vị {np.median(e_pos):.2f} "
                  f"p95 {np.percentile(e_pos, 95):.2f} | góc TB {e_ang.mean():.2f}° trung vị {np.median(e_ang):.2f}° | "
                  f"{time.time()-t0:.0f}s", flush=True)
    net.eval()
    torch.save({"state_dict": net.state_dict(), "args": vars(args)}, out / "pose_net.pt")

    # ---------------- đánh giá cuối trên toàn bộ val
    p = net.block_pose(imgs[va_i], dev)
    e_pos, e_ang = pose_errors(p, sfs[va_i])
    rep = {"val_n": int(len(va_i)), "val_pos_mean": float(e_pos.mean()), "val_pos_median": float(np.median(e_pos)),
           "val_pos_p95": float(np.percentile(e_pos, 95)), "val_ang_mean": float(e_ang.mean()),
           "val_ang_median": float(np.median(e_ang)), "val_ang_p95": float(np.percentile(e_ang, 95))}

    # ---------------- gán nhãn demo + kiểm tra lệch miền bằng render-and-compare
    raw = get_pusht(cache="data/pusht_raw.npz")
    pd = net.block_pose(raw.images, dev)
    demo_sf = np.concatenate([raw.state.astype(np.float64), pd], 1)   # agent thật + khối T ước lượng
    np.savez("data/demo_state.npz", state_full=demo_sf, episode_ends=raw.episode_ends)
    env = make_env()
    k = rng.choice(len(raw.images), 400, replace=False)
    iou_demo = render_iou(env, raw.images[k], raw.state[k], pd[k])
    kv = np.sort(rng.choice(va_i, min(400, len(va_i)), replace=False))
    iou_sim_pred = render_iou(env, imgs[kv], sfs[kv, :2], p[np.searchsorted(va_i, kv)])
    iou_sim_true = render_iou(env, imgs[kv], sfs[kv, :2], sfs[kv, 2:5])
    # độ "giật" nhãn demo: khối T gần như đứng yên giữa 2 frame liền kề khi chấm tròn ở xa
    ends = raw.episode_ends
    same_ep = np.ones(len(pd) - 1, bool)
    same_ep[ends[:-1] - 1] = False
    far = np.hypot(raw.state[:-1, 0] - pd[:-1, 0], raw.state[:-1, 1] - pd[:-1, 1]) > 120
    jit = np.hypot(*(pd[1:, :2] - pd[:-1, :2]).T)[same_ep & far]
    rep.update({"demo_iou_mean": float(iou_demo.mean()), "demo_iou_p10": float(np.percentile(iou_demo, 10)),
                "sim_val_iou_pred_mean": float(iou_sim_pred.mean()), "sim_val_iou_true_mean": float(iou_sim_true.mean()),
                "demo_jitter_static_median": float(np.median(jit)) if len(jit) else None,
                "demo_jitter_static_p95": float(np.percentile(jit, 95)) if len(jit) else None})
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    ok = rep["val_pos_mean"] < 2.0 and rep["demo_iou_mean"] > rep["sim_val_iou_true_mean"] - 0.05
    print("KIỂM TRA:", "OK" if ok else "CẦN XEM LẠI",
          "(val vị trí TB < 2 và IoU demo gần IoU của nhãn thật trên sim)")
    print(f"đã lưu {out}/pose_net.pt và data/demo_state.npz")


if __name__ == "__main__":
    main()
