"""Cache latent DINOv2 (+ PCA) cho các tập dữ liệu (để train world model nhanh).

    python scripts/cache_latents.py --encoder dinov2 --datasets demo sim_v1                   # fit PCA + cache
    python scripts/cache_latents.py --encoder dinov2 --datasets sim_v2 self_r1 --reuse-encoder   # dùng lại PCA

Kết quả trong data/latents/<encoder>/:
  encoder.pt      cấu hình + PCA (để mã hóa ảnh mới lúc chạy online y hệt khi cache)
  <dataset>.npy   latent fp16 [N, P, D], N = số frame của tập (cùng thứ tự với RawPushT)
  meta.json       shape, phương sai PCA giữ lại, thời gian
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

from pusht_wm.data import RawPushT, get_pusht
from pusht_wm.encoders import build_encoder, cache_latents, frame_stack

from pusht_wm.latent_data import data_file


def load_dataset(name):
    return get_pusht(cache=data_file(name)) if name == "demo" else RawPushT.load(data_file(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["dinov2"])
    ap.add_argument("--datasets", nargs="+", default=["demo", "sim_v1"])
    ap.add_argument("--pca-dim", type=int, default=128, help="0 = không PCA")
    ap.add_argument("--pca-frames", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="data/latents")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-encoder", action="store_true",
                    help="dùng lại encoder.pt (PCA) đã có -> latent tập mới tương thích world model cũ")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    out = Path(args.out) / args.encoder
    out.mkdir(parents=True, exist_ok=True)
    if args.reuse_encoder:
        from pusht_wm.encoders import load_encoder
        enc = load_encoder(out / "encoder.pt")
        args.pca_dim = 0  # không fit lại PCA
        print(f"dùng lại {out / 'encoder.pt'} (PCA cũ)")
    else:
        enc = build_encoder(args.encoder)
    raws = {n: load_dataset(n) for n in args.datasets}
    meta = {"encoder": enc.config(), "args": vars(args), "datasets": {}}

    if args.pca_dim:
        # mẫu PCA lấy đều từ mọi tập (tỉ lệ theo số frame)
        rng = np.random.default_rng(args.seed)
        total = sum(len(r.images) for r in raws.values())
        frames = []
        for r in raws.values():
            k = max(1, round(args.pca_frames * len(r.images) / total))
            idx = np.sort(rng.choice(len(r.images), k, replace=False))
            frames.append(frame_stack(r, idx, enc.n_frames))
        t0 = time.time()
        ev = enc.fit_pca(np.concatenate(frames), args.pca_dim)
        meta["pca_explained_variance"] = ev
        print(f"PCA {args.pca_dim} chiều giữ {ev*100:.1f}% phương sai ({time.time()-t0:.0f}s)")
    if not args.reuse_encoder:
        enc.save(out / "encoder.pt")

    for name, r in raws.items():
        t0 = time.time()
        print(f"[{name}] {len(r.images)} frame")
        lat = cache_latents(enc, r, out / f"{name}.npy", args.batch)
        meta["datasets"][name] = {"shape": list(lat.shape), "time_s": time.time() - t0,
                                  "source_file": data_file(name)}
        print(f"  xong {tuple(lat.shape)} trong {time.time()-t0:.0f}s, "
              f"{(out / f'{name}.npy').stat().st_size/2**30:.2f} GB")
    mpath = out / ("meta.json" if not args.reuse_encoder else f"meta_{'_'.join(args.datasets)}.json")
    mpath.write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
