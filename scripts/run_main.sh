#!/usr/bin/env bash
# Train lại TOÀN BỘ mô hình chính từ dữ liệu có sẵn trong data/ (máy GPU, Linux). Kết quả ra outputs/, KHÔNG ghi đè models/.
#   tmux new-session -d -s main "bash scripts/run_main.sh > outputs/run_main.log 2>&1"
# Thứ tự: WM encoder -> WM dynamics -> mô hình nghịch -> Diffusion sinh PATH (3 phần đầu ĐÓNG BĂNG khi train Diffusion) -> test.
# Lưu ý: bước 1 ghi lại data/demo_state.npz (nhãn khối T của 206 demo do encoder mới gán).
# Thời gian tham khảo (RTX 5060 Ti): encoder ~1 giờ, WM ~30 phút, nghịch ~10 phút, Diffusion ~2 giờ, test ~10 phút.
set -eo pipefail
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
P=${P:-.venv/bin/python}
O=outputs
mkdir -p $O

date; echo "== 1) WM encoder (PoseNet): ảnh -> x, y, góc khối T (sim_v1 + sim_v2 có nhãn thật)"
$P scripts/train_pose_net.py --out $O/pose_net
date; echo "== 2) WM dynamics: (s_{t-1}, s_t, a_t) -> s_{t+1}, MLP 4x512, sim_v1, 40k bước, loss rollout 8 bước"
$P scripts/train_state_wm.py --hidden 512 --w-move 3 --steps 40000 --batch 2048 --n-test 1000000 --out $O/state_wm
date; echo "== 3) mô hình nghịch: (s_{k-1}, s_k, s_{k+1}) -> a_k (sim_v1 + sim_v2 + demo)"
$P scripts/train_inverse.py --out $O/inverse
date; echo "== 4) Diffusion sinh 16 PATH trạng thái từ s_t (demo + sim_v2), 3 phần trên ĐÓNG BĂNG"
$P scripts/train_state_dp.py --pose-net $O/pose_net/pose_net.pt --state-wm $O/state_wm/state_wm.pt \
   --inverse $O/inverse/inverse.pt --out $O/path_policy
date; echo "== 5) test 500 seed 600000.. (16 path + WM chọn, phạt 0.5) — mô hình đã chốt: 89.8% / 143"
$P scripts/eval_state_dp.py --ckpt $O/path_policy/ckpt_best.pt --out $O/eval
date; echo "XONG. Muốn dùng bản mới: chép 4 file vào models/ (pose_net.pt, state_wm.pt, inverse.pt, path_policy.pt)"
