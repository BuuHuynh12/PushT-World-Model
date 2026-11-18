#!/usr/bin/env bash
# Train toàn bộ mô hình trên máy GPU Linux (RTX 3090 24 GB: ~6–7 giờ mỗi vòng):
#   frame -> WM (DINOv2 + transformer, loss latent + FDE) -> s_t latent -> Diffusion -> PATH robot -> (toán) -> action
# Quy trình: load data -> setup simulation -> train WM -> train Diffusion (WM đóng băng) -> evaluate
#            -> tự chơi bằng mô hình vừa train -> train lại WM ... (ROUNDS vòng sau vòng 0)
#   tmux new-session -d -s loop "bash scripts/run_loop.sh > outputs/run_loop.log 2>&1"
# Cần sẵn: data/pusht_raw.npz (tự tải từ lerobot/pusht nếu thiếu), data/sim_v1.npz, data/sim_v2.npz.
# Seed: 100000.. chọn checkpoint | 600000.. TEST (chỉ để báo cáo) | 9r0000.. tự chơi vòng r.
# Mô hình trong models/ = vòng 2 của quy trình này.
set -eo pipefail
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
P=${P:-.venv/bin/python}
O=${O:-outputs/loop}
ROUNDS=${ROUNDS:-2}
SELF_N=${SELF_N:-1000}
mkdir -p $O

step() { date; echo "== $*"; }

if [ ! -f data/latents/dinov2/sim_v2.npy ]; then
  step "0) cache latent DINOv2 (fit PCA trên demo + sim_v1; sim_v2 dùng lại PCA đó)"
  $P scripts/cache_latents.py --encoder dinov2 --datasets demo sim_v1
  $P scripts/cache_latents.py --encoder dinov2 --datasets sim_v2 --reuse-encoder
fi
WM_DATA="demo,sim_v1"        # dữ liệu train world model (sim_v2 để riêng làm tập test của world model)
DP_EXTRA="sim_v2"            # dữ liệu thêm cho Diffusion (ngoài 206 demo)
for r in $(seq 0 $ROUNDS); do
  R=$O/r$r
  if [ $r -gt 0 ]; then
    step "vòng $r: tự chơi $SELF_N màn bằng mô hình vòng $((r-1)) -> data/self_r$r.npz"
    $P scripts/eval_latent_path.py --ckpt $O/r$((r-1))/path/ckpt_best.pt --seed-start $((900000 + r*10000)) \
       --n-seeds $SELF_N --record data/self_r$r.npz --out $R/self
    $P scripts/cache_latents.py --encoder dinov2 --datasets self_r$r --reuse-encoder
    WM_DATA="$WM_DATA,self_r$r"; DP_EXTRA="$DP_EXTRA self_r$r"
  fi
  step "vòng $r: train world model trên [$WM_DATA] (loss latent + FDE)"
  $P scripts/train_wm.py --config configs/wm_image.yaml --set "datasets=[$WM_DATA]" out_dir=$R/wm
  step "vòng $r: world model dự đoán thay đổi vị trí khối T (test sim_v2)"
  $P scripts/eval_wm_object.py --wm-ckpt $R/wm/wm_best.pt --out $R/wm
  step "vòng $r: train Diffusion đọc s_t latent -> path (demo + $DP_EXTRA), world model đóng băng"
  $P scripts/train_latent_path.py --wm $R/wm/wm_best.pt --extra $DP_EXTRA --out $R/path
  step "vòng $r: test 500 seed 600000.. (16 path + WM chọn; và 1 path không chọn để đối chứng)"
  $P scripts/eval_latent_path.py --ckpt $R/path/ckpt_best.pt --n-samples 16 --out $R/eval
  $P scripts/eval_latent_path.py --ckpt $R/path/ckpt_best.pt --n-samples 1 --out $R/eval
done
date; echo "XONG. Kết quả từng vòng: $O/r*/wm/object_wm_image_sim_v2_H16.json, $O/r*/eval/result_*.json"
