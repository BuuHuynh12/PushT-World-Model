# Push-T: World Model latent + Diffusion sinh đường đi

Robot (một chấm tròn 2D) đẩy khối chữ T vào vùng đích, chỉ nhìn qua **ảnh 96 × 96**. Project xây dựng một **world model**
đọc ảnh thành trạng thái ẩn **s_t** (latent, không dùng tọa độ) và **dự đoán khối T sẽ di chuyển thế nào** khi robot
hành động. Một **Diffusion** đọc s_t để đề xuất đường đi cho robot, world model tưởng tượng hậu quả của từng đường đi và
chọn đường tốt nhất; đường đi được đổi ra lệnh điều khiển **bằng toán**.

> Đề bài: *xây dựng World Model — từ trạng thái hiện tại và hành động của robot, dự đoán sự thay đổi vị trí vật thể.*
> Dữ liệu `lerobot/pusht`; tham khảo Diffusion Policy (Chi et al., 2023).

```
frame ─► WM: DINOv2 + transformer ─► s_t latent ─► Diffusion ─► 16 path robot ─► WM tưởng tượng, chọn path ─► toán ─► action
          (đóng băng khi train Diffusion)
```

## Kết quả chính

| | Kết quả |
|---|---|
| **World model** dự đoán vị trí khối T sau 16 bước (1.6 s) khi đang bị đẩy, tập chưa từng thấy | lệch **3.61 đv / 0.75°** — giảm **87.3%** sai số so với "đoán đứng yên" (28.50 đv) |
| **Đẩy khối T**, 500 màn test, chuẩn Push-T (≤ 300 bước, coverage > 95%) | **92.0%** ±2.4 thành công, trung vị **141 bước (14.1 s)** |
| Đối chứng: Diffusion 1 path, không có world model chọn | 84.8% ±3.1, trung vị 173 bước |

Bàn 512 × 512 đv, khối T dài 120 đv; 10 bước = 1 giây. World model chọn path giúp rõ rệt: so cặp từng màn, 16 path thắng
60 màn mà 1 path thua, ngược lại 24 màn (sign test p = 0.0001).

## Mô hình

**1. World model (WM)** — `pusht_wm/world_model.py`, `models/wm.pt` (9.4 M tham số train)
- Đọc ảnh: DINOv2 ViT-S/14 đóng băng, ảnh 96 → 224, 16×16 patch → gộp 8×8 = **64 token**, PCA 384 → 128 chiều.
  s_t = token của 2 frame gần nhất. **Không có tọa độ khối T trong s_t.**
- Dự đoán: transformer nhân quả 6 lớp, d = 256, cửa sổ 4 frame; nhận s_t + action, dự đoán latent của frame kế tiếp
  (phần chênh lệch) và coverage (phần khối T phủ vùng đích). Rollout nhiều bước bằng cách đưa dự đoán vào lại.
- Đầu đọc tư thế (PoseHead) train chung: latent → (chấm tròn x, y, khối T x, y, góc). Chỉ dùng cho loss FDE và để chấm
  path; không đưa vào s_t.

**Loss của world model**

```
L = MSE latent (teacher forcing) + MSE latent (rollout 16 bước) + 10 · MSE coverage
  + L1 đọc tư thế trên latent thật + FDE / 256
FDE = sqrt(Δx² + Δy²) + 60 · |Δθ|      (khối T, ở bước cuối của rollout 16 bước; Δθ theo radian)
```

Tọa độ khối T chỉ xuất hiện trong loss (lấy từ simulator), không phải đầu vào. Dữ liệu demo không có tư thế nên chỉ góp
phần loss latent.

**2. Diffusion sinh path** — `pusht_wm/latent_path.py`, `models/diffusion_path.pt` (69 M tham số)
- Đầu vào: s_t của 2 frame + trạng thái ẩn của transformer WM (WM **đóng băng**) + vị trí chấm tròn.
  Bộ đọc token: transformer 3 lớp + token CLS.
- Đầu ra: **path** = 16 vị trí tương lai của chấm tròn (1.6 s). U-Net 1D có điều kiện, DDPM 100 bước khi train,
  DDIM 10 bước khi chạy. Nhãn path lấy thẳng từ quỹ đạo robot trong dữ liệu → **không cần gán nhãn**.

**3. Chọn path** — Diffusion sinh 16 path; WM tưởng tượng 16 bước tương lai của từng path trong latent, PoseHead đọc tư
thế khối T tưởng tượng; điểm = tới đích sớm nhất, nếu không thì gần đích nhất (vị trí, góc, coverage). Thực hiện 8 bước
đầu của path tốt nhất rồi nghĩ lại.

**4. Path → action bằng toán** — `path_to_actions`. Simulator điều khiển chấm tròn bằng bộ PD (k_p = 100, k_v = 20,
10 bước con 0.01 s mỗi action). Gộp 10 bước con thành một phép biến đổi tuyến tính:

```
p' = 0.7043 p + 0.0299 v + 0.2957 a        v' = -3.7415 p - 0.0066 v + 3.7415 a
```

Vận tốc hiện tại suy từ hai vị trí gần nhất; giải ngược để có action a đưa robot tới đúng điểm kế tiếp của path.
Kiểm tra với simulator (`scripts/check_path_math.py`): sai số 0.0000 đv.

## Quy trình train

Theo đúng vòng lặp: **load data → setup simulation → train WM → train Diffusion (WM đóng băng) → evaluate → tự chơi →
train lại WM**. Mỗi vòng, mô hình vừa train tự chơi 1000 màn mới; dữ liệu đó được thêm vào để train lại WM và Diffusion.

| Vòng | Dữ liệu WM | WM: lệch vị trí khối T sau 16 bước (giảm so với đứng yên) | Đẩy khối T, 16 path | 1 path |
|---|---|---|---|---|
| 0 | demo + sim_v1 | 6.18 đv (78.3%) | 82.8% | 76.8% |
| 1 | + 1000 màn tự chơi (mô hình vòng 0, thắng 80.9%) | 4.31 đv (84.9%) | 90.6% | 80.2% |
| **2** | + 1000 màn tự chơi (mô hình vòng 1, thắng 89.7%) | **3.61 đv (87.3%)** | **92.0%** | **84.8%** |

- Vòng 0 và 1 dùng 16 token (gộp 4×4); vòng 2 dùng 64 token. Ở vòng 0 đã thử cả hai: 64 token cho Diffusion tốt hơn
  (86.4% so với 82.8%, so cặp p = 0.12).
- So cặp từng màn: vòng 2 vs vòng 0 71 / 25 màn (p < 0.0001), vs vòng 0 bản 64 token 64 / 36 (p = 0.007);
  vòng 2 vs vòng 1: 40 / 33 màn (p = 0.48) → mức tăng của vòng tiếp theo
  không còn chứng minh được, nên **dừng vòng lặp ở vòng 2**. Train lại Diffusion vòng 1 với cùng cấu hình từng cho
  87.4% → chênh lệch giữa hai lần train cỡ ±3 điểm.

**World model vòng 2 theo tầm nhìn trước** (sim_v2, lúc khối T đang bị đẩy; sàn = đọc tư thế trên latent thật 2.73 đv):

| Nhìn trước | 1 bước | 4 bước | 8 bước | 16 bước |
|---|---|---|---|---|
| WM dự đoán | 2.86 đv | 2.96 đv | 3.11 đv | **3.61 đv** |
| Đoán "đứng yên" | 3.91 đv | 11.84 đv | 19.18 đv | 28.50 đv |
| Giảm sai số | 26.9% | 75.0% | 83.8% | **87.3%** |
| Cos hướng dịch chuyển | 0.941 | 0.994 | 0.997 | 0.998 |

**Màn thua** (40 / 500): 22 màn khối T đã sát đích (coverage cao nhất ≥ 0.9) nhưng không chốt được 95%; 8 màn 0.5–0.9;
10 màn dưới 0.5 (6 màn khối T không chạm vùng đích).

## Dữ liệu

| Tập | Nguồn | Episode | Frame | Dùng cho |
|---|---|---|---|---|
| demo | `lerobot/pusht` (người điều khiển) | 206 | 25 650 | WM, Diffusion |
| sim_v1 | simulator gym-pusht: policy sơ bộ train trên demo + nhiễu + action ngẫu nhiên | 600 | 158 926 | WM |
| sim_v2 | simulator: policy sơ bộ + nhánh "nếu đi đường khác" | 300 + 9 548 nhánh | 150 116 | **test WM**; Diffusion (màn thắng) |
| self_r1, self_r2 | mô hình vòng 0, vòng 1 tự chơi | 1 000 + 1 000 | 361 000 | WM, Diffusion (màn thắng) |

Seed: 100000.. chọn checkpoint (50 màn) · **600000..600499 test** · 910000.., 920000.. tự chơi — không trùng nhau.

## Chạy thử

```bash
git lfs install                      # trước khi clone (models/diffusion_path.pt 277 MB dùng Git LFS)
git clone <url repo> && cd <thư mục>
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # Linux: .venv/bin/python
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python app_pusht.py    # app: đặt khối T, bấm SPACE để mô hình đẩy
```

Trong app: kéo chuột trái để đặt khối T, chuột phải đặt chấm tròn, Q/E xoay, R vị trí ngẫu nhiên, SPACE chạy;
V bật/tắt vẽ suy nghĩ (16 đường đi + khối T world model tưởng tượng), P dừng sau mỗi lần nghĩ (N đi tiếp),
F chạy nhanh, G quay GIF. Lần chạy đầu tự tải DINOv2 (~90 MB) từ Hugging Face. Không có GPU vẫn chạy được (chậm hơn).

Đánh giá (cần GPU): `python scripts/eval_latent_path.py` (500 màn test) · `python scripts/eval_wm_object.py`
(world model trên sim_v2) · `python scripts/check_path_math.py` (công thức path → action).
Train lại toàn bộ: `bash scripts/run_loop.sh` trên máy GPU Linux (RTX 3090: ~6–7 giờ mỗi vòng); cần `data/sim_v1.npz`,
`data/sim_v2.npz` (không có trong repo vì dung lượng lớn; `data/pusht_raw.npz` tự tải từ `lerobot/pusht`).

## Cấu trúc

```
app_pusht.py            app xem mô hình đẩy khối T và "suy nghĩ" của nó
models/                 wm.pt (world model + PCA), diffusion_path.pt (Diffusion, Git LFS)
pusht_wm/
  encoders.py           DINOv2 + PCA: ảnh -> token
  world_model.py        world model latent, loss (latent + coverage + FDE)
  wm_planner.py         đầu đọc tư thế, chi phí chấm path
  planner.py            world model lúc chạy: mã hóa frame, tưởng tượng hậu quả của các path
  latent_path.py        Diffusion sinh path, path -> action (toán), chạy đánh giá / tự chơi
  latent_controller.py  bộ điều khiển cho app
  sim.py, data.py, latent_data.py, collect.py, ...   simulator, dữ liệu
scripts/                run_loop.sh (toàn bộ quy trình), train_wm, train_latent_path, eval_*, cache_latents, check_path_math
configs/wm_image.yaml   cấu hình world model
results/                kết quả gốc từng vòng (json, log)
```

Dựa trên: Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (2023); Zhou et al.,
*DINO-WM: World Models on Pre-trained Visual Features* (2024); Oquab et al., *DINOv2* (2023); dữ liệu `lerobot/pusht`;
môi trường `gym-pusht`.
