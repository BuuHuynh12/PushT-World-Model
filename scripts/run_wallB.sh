#!/usr/bin/env bash
# Chạy GỌN bản đã chọn trên tập tinh chỉnh (path_wallB: sát tường <= 150 + tự chơi chuẩn <= 100) rồi TEST một lần.
#   tmux new-session -d -s wb "bash scripts/run_wallB.sh > outputs/wall/runB.log 2>&1"
# Cần: data/self_path.npz (tự chơi chuẩn bước 17, lưu ở archive/data/). Máy 24 GB VRAM.
# Test: 800000.. sát tường + 600000.. chuẩn, chạy tới 1000 bước (in cả kết quả trong 300 bước). Mô hình chính chạy CÙNG seed
# để so từng seed (McNemar). WM encoder + WM dynamics + mô hình nghịch ĐÓNG BĂNG; Diffusion loss MSE thường.
set -eo pipefail
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
P=${P:-.venv/bin/python}
O=outputs/wall
JOBS=$O/jobs.txt
mkdir -p $O/eval $O/self data
rm -f $JOBS
add_eval() {
  local name=$1 start=$2 n=$3 k=$4; shift 4
  local per=$(( (n + k - 1) / k )) i s m
  for i in $(seq 0 $((k - 1))); do
    s=$((start + i * per)); m=$(( n - i * per < per ? n - i * per : per ))
    echo "$P scripts/eval_state_dp.py --out $O/eval/$name --seed-start $s --n-seeds $m $* > $O/eval/$name.$i.log 2>&1" >> $JOBS
  done
}
run_jobs() { xargs -d '\n' -P ${1:-6} -I CMD bash -c CMD < $JOBS; rm -f $JOBS; }
merge() { $P scripts/merge_results.py $O/eval/$1/result_*.json --out $O/eval/$1.json --name $1; }

date; echo "== 1) mô hình chính TỰ CHƠI 2100 màn sát tường (6 x 350, seed 810000..)"
for i in $(seq 0 5); do
  echo "$P scripts/collect_selfplay.py --starts wall --seed-start $((810000 + i * 350)) --n-seeds 350 --out $O/self/self_$i.npz > $O/collect_$i.log 2>&1" >> $JOBS
done
run_jobs 6
tail -qn 1 $O/collect_*.log
$P scripts/collect_selfplay.py --merge "$O/self/self_*.npz" --out data/wall_self.npz
echo STEP1_DONE

date; echo "== 2) train path_wallB (20k bước) || SONG SONG: mô hình chính trên TEST (800000 sát tường, 600000 chuẩn), tới 1000 bước"
$P scripts/train_state_dp.py --extra sim_v2 wall_self self_path --extra-max-steps 0 150 100 --out $O/path_wallB > $O/path_wallB.log 2>&1 &
TP=$!
add_eval TEST_main_wall800k 800000 500 3 --starts wall --max-steps 1000
add_eval TEST_main_std600k  600000 500 2 --max-steps 1000
run_jobs 5
for n in TEST_main_wall800k TEST_main_std600k; do merge $n; done
echo STEP2A_DONE
wait $TP
grep -E '^\+|lọc|^\[eval' $O/path_wallB.log | tr '\n' ' '; echo
echo STEP2_DONE

date; echo "== 3) TEST path_wallB (một lần): 800000 sát tường + 600000 chuẩn, tới 1000 bước"
add_eval TEST_B_wall800k 800000 500 3 --ckpt $O/path_wallB/ckpt_best.pt --starts wall --max-steps 1000
add_eval TEST_B_std600k  600000 500 3 --ckpt $O/path_wallB/ckpt_best.pt --max-steps 1000
run_jobs 6
for n in TEST_B_wall800k TEST_B_std600k; do merge $n; done

date; echo "== 4) so từng seed: B vs mô hình chính (trong 300 bước và trong 1000 bước)"
$P - <<'EOF'
import json, math
from fractions import Fraction
import numpy as np
def p(x, y):
    x, y = int(x), int(y); n = x + y; k = min(x, y)
    return float(min(1, Fraction(2 * sum(math.comb(n, i) for i in range(k + 1)), 2 ** n))) if n else 1.0
def load(f, lim):
    return {q["seed"]: (q["steps"] if q["success"] and q["steps"] <= lim else None) for q in json.load(open(f))["per_seed"]}
E = "outputs/wall/eval/"
for tag in ("wall800k", "std600k"):
    for lim in (300, 1000):
        A, B = load(E + f"TEST_main_{tag}.json", lim), load(E + f"TEST_B_{tag}.json", lim)
        oa = sum(A[s] is not None and B[s] is None for s in A); ob = sum(B[s] is not None and A[s] is None for s in A)
        c = [s for s in A if A[s] and B[s]]; d = np.array([B[s] - A[s] for s in c]); fa, sl = (d < 0).sum(), (d > 0).sum()
        sa = np.mean([v is not None for v in A.values()]); sb = np.mean([v is not None for v in B.values()])
        print(f"{tag} <= {lim} bước: chính {sa:.3f} | B {sb:.3f} | chỉ chính thắng {oa} / chỉ B thắng {ob}, p = {p(oa, ob):.3f}"
              f" | số bước trên seed cùng thắng {d.mean():+.1f} (p = {p(fa, sl):.2g})")
EOF
date; echo XONG_B
