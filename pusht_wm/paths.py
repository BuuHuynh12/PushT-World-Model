"""Đường dẫn checkpoint của MÔ HÌNH CHÍNH + dò đường dẫn cũ (sau khi dời các thí nghiệm sang archive/).

Checkpoint lưu sẵn đường dẫn tới các thành phần khác (vd path_policy.pt ghi `outputs/step11/.../state_wm.pt`).
resolve() đổi các đường dẫn đó sang chỗ mới: models/ (mô hình chính) hoặc archive/ (thí nghiệm cũ).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

POSE_NET = "models/pose_net.pt"        # WM encoder: ảnh -> x, y, góc khối T
STATE_WM = "models/state_wm.pt"        # WM dynamics: (s_{t-1}, s_t, a_t) -> s_{t+1}
INVERSE = "models/inverse.pt"          # mô hình nghịch: (s_{k-1}, s_k, s_{k+1}) -> a_k
PATH_POLICY = "models/path_policy.pt"  # Diffusion sinh 16 PATH trạng thái

# đường dẫn lúc train (ghi trong checkpoint) -> chỗ mới
_OLD2NEW = {
    "outputs/step12/pose_net/pose_net.pt": POSE_NET,
    "outputs/step11/state_wm_h512_w3/state_wm.pt": STATE_WM,
    "outputs/step14/inverse/inverse.pt": INVERSE,
    "outputs/step14/path_state_extra/ckpt_best.pt": PATH_POLICY,
}


def resolve(path) -> str:
    """Trả đường dẫn tồn tại: nguyên gốc -> bản trong models/ -> bản trong archive/. Không thấy: trả nguyên gốc."""
    p = Path(path)
    cands = [p, ROOT / p]
    key = p.as_posix()
    if key in _OLD2NEW:
        cands.append(ROOT / _OLD2NEW[key])
    if not p.is_absolute():
        cands.append(ROOT / "archive" / p)
    for c in cands:
        if c.exists():
            return str(c)
    return str(path)
