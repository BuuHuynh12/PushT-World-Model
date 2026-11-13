#!/usr/bin/env bash
# Vòng 2 sát tường: CÂN BẰNG dữ liệu — thêm tự chơi vị trí CHUẨN đã lọc <= 100 bước (self_path, bước 17) để không quên giữa bàn.
set -eo pipefail
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
P=.venv/bin/python; O=outputs/wall; PAR=6; JOBS=$O/jobs.txt; rm -f $JOBS
add_eval() { local name=$1 start=$2 n=$3 k=$4; shift 4; local per=$(( (n + k - 1) / k )) i s m
  for i in $(seq 0 $((k - 1))); do s=$((start + i * per)); m=$(( n - i * per < per ? n - i * per : per ))
    echo "$P scripts/eval_state_dp.py --out $O/eval/$name --seed-start $s --n-seeds $m $* > $O/eval/$name.$i.log 2>&1" >> $JOBS; done; }
run_jobs() { xargs -d '\n' -P $PAR -I CMD bash -c CMD < $JOBS; rm -f $JOBS; }
merge() { $P scripts/merge_results.py $O/eval/$1/result_*.json --out $O/eval/$1.json --name $1; }
date; echo "== 5) train 2 bản cân bằng song song: A sát tường<=200 + chuẩn<=100 | B sát tường<=150 + chuẩn<=100"
$P scripts/train_state_dp.py --extra sim_v2 wall_self self_path --extra-max-steps 0 200 100 --out $O/path_wallA > $O/path_wallA.log 2>&1 &
$P scripts/train_state_dp.py --extra sim_v2 wall_self self_path --extra-max-steps 0 150 100 --out $O/path_wallB > $O/path_wallB.log 2>&1 &
wait
for v in path_wallA path_wallB; do echo "$v: $(grep -E '^\+|lọc|^\[eval' $O/$v.log | tr '\n' ' ')"; done
date; echo "== 6) đánh giá trên tập tinh chỉnh"
for v in path_wallA path_wallB; do
  add_eval ${v}_wall 820000 300 3 --ckpt $O/$v/ckpt_best.pt --starts wall
  add_eval ${v}_std  700000 500 3 --ckpt $O/$v/ckpt_best.pt
done
run_jobs
for v in path_wallA path_wallB; do for t in wall std; do merge ${v}_$t; done; done
date; echo XONG2
