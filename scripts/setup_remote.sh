#!/usr/bin/env bash
# Cài môi trường trên máy GPU thuê (Ubuntu 22.04). Chạy trong thư mục project:
#   cd ~/pusht && bash scripts/setup_remote.sh
# Dùng uv để có Python 3.11 (Ubuntu 22.04 mặc định chỉ có 3.10), cài nhanh.
set -euo pipefail

# thư viện hệ thống cho opencv/pygame (gym-pusht) khi chạy không màn hình
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 tmux > /dev/null

if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv venv -p 3.11 .venv
# Bánh xe cu126 chạy được với driver CUDA >= 12.6. RTX 50xx (Blackwell): torch 2.14.1 KHÔNG có bản cu128,
# dùng cu130 (cần driver hỗ trợ CUDA 13): TORCH_INDEX=https://download.pytorch.org/whl/cu130
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
uv pip install -p .venv torch==2.14.1 torchvision==0.29.1 --index-url "$TORCH_INDEX"
uv pip install -p .venv -r requirements-lock.txt

export SDL_VIDEODRIVER=dummy
.venv/bin/python - <<'EOF'
import torch, gym_pusht, av, transformers
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
print("VRAM %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
print("AV1 decoder:", "libdav1d" in av.codecs_available)
EOF
echo "Xong. Kiểm tra tốc độ: .venv/bin/python scripts/check_dp.py"
