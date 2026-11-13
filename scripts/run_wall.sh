#!/usr/bin/env bash
# Mở rộng mô hình chính ra vị trí KHỐI T SÁT TƯỜNG / GÓC (ngoài phân phối chuẩn gym-pusht).
#   tmux new-session -d -s wall "bash scripts/run_wall.sh > outputs/wall/run.log 2>&1"
# DÙNG HẾT MÁY: mô phỏng chạy trên CPU (1 nhân / tiến trình), mỗi tiến trình ~3-4 GB VRAM -> chia nhỏ mọi việc theo khoảng
# seed và chạy PAR tiến trình song song (24 GB VRAM: 6; 16 GB: 4; 8-12 GB: 2), rồi gộp kết quả (merge_results.py).
# Seed: 800000.. TEST sát tường (chỉ chạy bản cuối, một lần) · 810000.. tự chơi · 820000.. TINH CHỈNH sát tường.
# WM encoder + WM dynamics + mô hình nghịch ĐÓNG BĂNG; chỉ Diffusion train lại; loss MSE thường (không FDE).
set -eo pipefail
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
P=${P:-.venv/bin/python}
O=outputs/wall
PAR=${PAR:-6}                 # số tiến trình song song
NSELF=${NSELF:-2100}          # tổng số màn tự chơi
JOBS=$O/jobs.txt
mkdir -p $O/eval $O/self data
rm -f $JOBS

# add_eval TÊN SEED_ĐẦU SỐ_SEED SỐ_PHẦN [tham số eval_state_dp...] : thêm các phần vào hàng đợi
add_eval() {
  local name=$1 start=$2 n=$3 k=$4; shift 4
  local per=$(( (n + k - 1) / k )) i s m
  for i in $(seq 0 $((k - 1))); do
    s=$((start + i * per)); m=$(( n - i * per < per ? n - i * per : per ))
    echo "$P scripts/eval_state_dp.py --out $O/eval/$name --seed-start $s --n-seeds $m $* > $O/eval/$name.$i.log 2>&1" >> $JOBS
  done
}
run_jobs() { xargs -d '\n' -P $PAR -I CMD bash -c CMD < $JOBS; rm -f $JOBS; }
merge() { $P scripts/merge_results.py $O/eval/$1/result_*.json --out $O/eval/$1.json --name $1; }

date; echo "== 1) MỐC (mô hình chính): 300 màn sát tường (tinh chỉnh 820000) không phạt / phạt sát mép 2;"
echo "      + test CHUẨN 600000 cho chạy tới 1000 bước (đề không bắt buộc 300 bước)"
add_eval base_wall      820000 300 2 --starts wall
add_eval base_wall_edge2 820000 300 2 --starts wall --edge-w 2
add_eval base_std_max1000 600000 500 2 --max-steps 1000
run_jobs
for n in base_wall base_wall_edge2 base_std_max1000; do merge $n; done

date; echo "== 2) mô hình chính TỰ CHƠI từ vị trí sát tường ($NSELF màn, $PAR phần song song)"
per=$(( (NSELF + PAR - 1) / PAR ))
for i in $(seq 0 $((PAR - 1))); do
  echo "$P scripts/collect_selfplay.py --starts wall --seed-start $((810000 + i * per)) --n-seeds $per --out $O/self/self_$i.npz > $O/collect_$i.log 2>&1" >> $JOBS
done
run_jobs
tail -qn 1 $O/collect_*.log
$P scripts/collect_selfplay.py --merge "$O/self/self_*.npz" --out data/wall_self.npz
$P - <<'EOF'
import numpy as np
d = np.load("data/wall_self.npz"); e = d["episode_ends"]; s = np.r_[0, e[:-1]]; suc = d["success"]
k = np.array([np.argmax(suc[a:b]) + 1 for a, b in zip(s, e) if suc[a:b].any()])
print(f"thành công {len(k)}/{len(e)}" + (f" | số bước: p25 {np.percentile(k, 25):.0f} trung vị {np.median(k):.0f} "
      f"p75 {np.percentile(k, 75):.0f} | <= 200 bước: {(k <= 200).sum()}" if len(k) else ""))
EOF

date; echo "== 3) train lại Diffusion (2 bản SONG SONG, 20k bước): demo + sim_v2 + màn sát tường tự chơi THẮNG"
echo "      path_wall: mọi màn thắng | path_wall200: chỉ màn thắng trong <= 200 bước"
$P scripts/train_state_dp.py --extra sim_v2 wall_self --out $O/path_wall > $O/path_wall.log 2>&1 &
$P scripts/train_state_dp.py --extra sim_v2 wall_self --extra-max-steps 0 200 --out $O/path_wall200 > $O/path_wall200.log 2>&1 &
wait
for v in path_wall path_wall200; do echo "$v: $(grep -E '^\+|lọc|^\[eval' $O/$v.log | tr '\n' ' ')"; done

date; echo "== 4) bản mới: sát tường (tinh chỉnh 820000, không phạt / phạt 2) + CHUẨN (tinh chỉnh 700000: bản chính 88.8% / 142)"
for v in path_wall path_wall200; do
  add_eval ${v}_wall       820000 300 2 --ckpt $O/$v/ckpt_best.pt --starts wall
  add_eval ${v}_wall_edge2 820000 300 2 --ckpt $O/$v/ckpt_best.pt --starts wall --edge-w 2
  add_eval ${v}_std        700000 500 2 --ckpt $O/$v/ckpt_best.pt
done
run_jobs
for v in path_wall path_wall200; do for t in wall wall_edge2 std; do merge ${v}_$t; done; done
date; echo "XONG — chọn bản tốt nhất trên tập tinh chỉnh rồi mới chạy test 800000 (sát tường) + 600000 (chuẩn), mỗi tập một lần"
