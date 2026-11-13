"""pusht_wm — MÔ HÌNH CHÍNH: frame -> WM đọc s_t -> Diffusion sinh PATH -> WM chọn -> mô hình nghịch -> action.

Module của các thí nghiệm cũ đã dời sang archive/pusht_wm_old/ nhưng VẪN import được dưới tên cũ
(vd `from pusht_wm.policy import DiffusionPolicy`) nhờ thêm thư mục đó vào __path__ của package.
"""
from pathlib import Path as _Path

_OLD = _Path(__file__).resolve().parents[1] / "archive" / "pusht_wm_old"
if _OLD.is_dir():
    __path__.append(str(_OLD))
