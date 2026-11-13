"""Tạo ảnh động (GIF) cho README từ video đánh giá có sẵn (results/failures/*.mp4) — chỉ đổi định dạng, không chạy mô hình.

    python scripts/make_gifs.py

Ra: assets/demo_grid.gif (4 màn thành công, lưới 2x2), assets/demo_fail_near_goal.gif (1 màn thua kiểu "sát đích không chốt").
Video gốc: 96x96 phóng 3 lần, 10 bước/giây; GIF lấy 1 khung / 2 bước (xem nhanh gấp 2).
"""
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "failures"
OUT = ROOT / "assets"
SKIP = 2            # lấy 1 khung / SKIP bước
FPS = 10            # khung / giây của GIF


def frames(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label(img, text, color=(30, 30, 30), size=16):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = font(size)
    w = d.textlength(text, font=f)
    d.rectangle([4, 4, 12 + w, 10 + size], fill=(255, 255, 255))
    d.text((8, 5), text, fill=color, font=f)
    return np.asarray(im)


def annotate(fr, ok):
    """Ghi số bước lên từng khung; khung cuối ghi kết quả."""
    out = []
    for k, f in enumerate(fr):
        last = k == len(fr) - 1
        txt = f"bước {k}" + ((" — THÀNH CÔNG" if ok else " — hết giờ") if last else "")
        out.append(label(f, txt, (0, 130, 0) if (last and ok) else (200, 0, 0) if last else (30, 30, 30)))
    return out


def save_gif(fr, path, hold=12):
    fr = fr[::SKIP] + [fr[-1]] * hold                       # dừng lại ở khung cuối một chút
    ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=64) for f in fr]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"{path.relative_to(ROOT)}: {len(ims)} khung, {path.stat().st_size / 1e6:.2f} MB")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    wins = [600001, 600005, 600043, 600065]
    eps = []
    for s in wins:
        p = next(SRC.glob(f"fail_{s}_*THANG.mp4"))
        eps.append(annotate(frames(p), True))
    n = max(len(e) for e in eps)
    eps = [e + [e[-1]] * (n - len(e)) for e in eps]          # màn xong sớm giữ khung cuối
    grid = [np.concatenate([np.concatenate([eps[0][t], eps[1][t]], 1),
                            np.concatenate([eps[2][t], eps[3][t]], 1)], 0) for t in range(n)]
    save_gif(grid, OUT / "demo_grid.gif")
    p = next(SRC.glob("fail_600049_*.mp4"))
    save_gif(annotate(frames(p), False), OUT / "demo_fail_near_goal.gif")


if __name__ == "__main__":
    main()
