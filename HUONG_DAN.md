# Hướng dẫn chi tiết — cách hệ thống hoạt động, cách train, cách đánh giá, toàn bộ kết quả

Tài liệu cho thành viên nhóm **viết báo cáo**. Mọi con số dưới đây là số đã chạy thật (không ước lượng); file kết quả gốc
nằm trong `results/`. Ký hiệu: **đv** = đơn vị toạ độ của bàn (bàn 512 × 512 đv); **10 bước = 1 giây**.

---

## 1. Đề bài và phạm vi

> Xây dựng **World Model**: từ trạng thái hiện tại và hành động của robot, **dự đoán sự thay đổi vị trí vật thể**;
> dùng cùng **Diffusion Policy** để điều khiển robot đẩy khối chữ T vào vùng đích (bài Push-T).

- Môi trường: `gym-pusht` (mô phỏng vật lý 2D bằng pymunk). Robot là **một chấm tròn** (bán kính 15 đv) chỉ **đẩy** được,
  không kéo, không nhấc. Bàn có 4 bức tường. Action = toạ độ (x, y) đích của chấm tròn (bộ điều khiển PD kéo chấm tròn tới đó).
- Quan sát: ảnh RGB 96 × 96 + vị trí chấm tròn. **Vị trí và góc của khối T KHÔNG được cho** → phải đọc từ ảnh.
- Dữ liệu người làm mẫu: **206 demo** `lerobot/pusht` (khoảng 25.6 nghìn bước).
- Thành công = khối T phủ **> 95%** vùng đích (coverage) trong **≤ 300 bước** (chuẩn của gym-pusht / LeRobot).

---

## 2. Kiến trúc mô hình chính

```
ảnh t-1, t ─► [1] WM encoder ─► s_t ─► [3] Diffusion ─► 16 PATH ─► [2] WM dynamics ─► chọn path ─► [4] Mô hình nghịch ─► 8 action
               (ResNet18)       │      (U-Net 1D)       (mỗi path:    tưởng tượng hậu quả   tốt nhất       (MLP)
                                │                         16 trạng thái  của từng path
vị trí chấm tròn ───────────────┘                         tương lai)
```
s_t = trạng thái 2 khung gần nhất, mỗi khung 6 số: chấm tròn (x, y), khối T (x, y), sin(góc), cos(góc) (toạ độ chuẩn hoá x/256 − 1).

Một vòng ra quyết định (lặp lại sau mỗi 8 action = 0.8 s, kiểu "receding horizon"):
1. **WM encoder** đọc vị trí + góc khối T từ 2 ảnh gần nhất → ghép với vị trí chấm tròn → s_t.
2. **Diffusion** sinh **16 path** khác nhau từ cùng s_t (mỗi path = 16 trạng thái tương lai của chấm tròn + khối T; 2 trạng thái
   đầu thay bằng trạng thái quan sát thật).
3. **Mô hình nghịch** đổi mỗi path thành 14 action: a_k = f(s_{k-1}, s_k, s_{k+1}).
4. **WM dynamics** "tưởng tượng": thực hiện 14 action đó từ trạng thái hiện tại thì khối T sẽ đi đâu (rollout 14 bước).
5. **Chấm điểm** từng path (càng nhỏ càng tốt):
   - nếu khối T tưởng tượng đạt coverage > 95% ở bước k: chi phí = −2 + k/100 (tới đích càng sớm càng tốt);
   - nếu không: (1 − coverage) + |khối T − đích|/256 + |lệch góc|/π + 0.3·|chấm tròn − khối T|/256 (ở bước cuối);
   - cộng **phạt path phi thực tế** 0.5 × độ lệch TB giữa khối T trong path (Diffusion "muốn") và khối T WM tưởng tượng
     (path đòi khối T đi theo cách vật lý không cho phép thì bị phạt).
6. Chọn path chi phí nhỏ nhất → thực thi **8 action đầu** → quay lại bước 1.

Vai trò của world model: (a) **encoder** = nhận thức (ảnh → trạng thái), (b) **dynamics** = dự đoán hậu quả hành động →
dùng để **lập kế hoạch / chọn** trong các đề xuất của Diffusion. Diffusion = học bắt chước (imitation learning) từ demo.
Không dùng học tăng cường (RL).

### 2.1. Bốn thành phần (file trong `models/`)

| # | Thành phần | Kiến trúc | Vào → ra | Dữ liệu train | Train |
|---|---|---|---|---|---|
| 1 | **WM encoder** `pose_net.pt` | ResNet18 (train từ đầu) | ảnh 96×96 → x, y, góc khối T | ảnh `sim_v1` + `sim_v2` có nhãn thật (tách val theo episode) | 30k bước, batch 256, lr 1e-3 |
| 2 | **WM dynamics** `state_wm.pt` | MLP 4 lớp × 512 (~0.8 triệu tham số) | (s_{t-1}, s_t, a_t) → Δs → s_{t+1}; rollout tự hồi quy nhiều bước | `sim_v1` (600 episode, ~159 nghìn bước, có trạng thái thật) | 40k bước, batch 2048, loss MSE rollout 8 bước, trọng số ×3 cho đoạn khối T bị đẩy |
| 3 | **Diffusion** `path_policy.pt` | U-Net 1D theo thời gian (kênh 256-512-1024, kernel 5), điều kiện s_t qua FiLM | s_t → 16 trạng thái tương lai (6 số/bước) | 206 demo (nhãn khối T do encoder gán) + 238 màn thắng của `sim_v2` | 20k bước, batch 1024, lr 2e-4, bf16, EMA |
| 4 | **Mô hình nghịch** `inverse.pt` | MLP 3 lớp × 512 | (s_{k-1}, s_k, s_{k+1}) → a_k | `sim_v1` + `sim_v2` + demo | 30k bước, batch 4096, loss L1 |

Chi tiết Diffusion (giống bài báo Diffusion Policy, Chi et al. 2023):
- Train: **DDPM 100 bước**, lịch nhiễu squared-cosine, mạng **đoán nhiễu ε**, loss **MSE** (không dùng FDE).
- Suy luận: **DDIM 10 bước** (tất định), trọng số **EMA**.
- Nhìn 2 khung, sinh 16 bước, thực thi 8 bước rồi sinh lại.
- Khác bài báo: (a) sinh **path trạng thái** thay vì action (rồi mô hình nghịch đổi ra action); (b) điều kiện là s_t do
  **WM encoder đọc từ ảnh** thay vì ảnh thô; (c) lấy **16 mẫu** và để **WM dynamics chọn** thay vì 1 mẫu.
- Lúc train cộng nhiễu nhỏ vào s_t (1 đv, 0.5°) để bền với sai số của encoder.
- Lúc train Diffusion, **WM encoder, WM dynamics và mô hình nghịch đều ĐÓNG BĂNG** (đã train trước, không cập nhật).
- Chọn checkpoint: cứ 4000 bước chạy thử 50 màn (seed 100000..), giữ bản thành công cao nhất.

### 2.2. Dữ liệu (thư mục `data/`)

| File | Nội dung | Nguồn |
|---|---|---|
| `pusht_raw.npz` | 206 demo: ảnh, vị trí chấm tròn, action | tự tải từ Hugging Face `lerobot/pusht` lần đầu chạy (`pusht_wm/data.py`) |
| `demo_state.npz` | vị trí + góc khối T của từng bước demo (demo gốc không có) | WM encoder gán nhãn (`train_pose_net.py`) |
| `sim_v1.npz` | 600 episode mô phỏng (đủ kiểu: sạch, nhiễu, ngẫu nhiên), ảnh + trạng thái THẬT | sinh bằng mô phỏng (thành công 31%) |
| `sim_v2.npz` | 300 episode policy + world model chọn path và ~9.5 nghìn nhánh "nếu chọn path khác" gần đích | sinh bằng mô phỏng |

`sim_v1`, `sim_v2`, `pusht_raw` lớn (~170 MB) nên **không có trong repo** — xin link Drive thư mục `data/` từ người giữ
project (chỉ cần khi muốn train lại; xem app / đọc kết quả thì không cần).

---

## 3. Cách đánh giá

- **Seed tách riêng** (mỗi seed = một vị trí ban đầu ngẫu nhiên của khối T + chấm tròn):
  `100000..` chọn checkpoint · `700000..` **tinh chỉnh** (chọn cấu hình) · `600000..600499` **TEST** (chỉ chạy bản cuối, một lần) ·
  `800000..` test sát tường · `820000..` tinh chỉnh sát tường · `300000.., 400000.., 810000..` tự chơi.
- Chỉ số: **(1) tỉ lệ thành công** (± = khoảng tin cậy 95%); **(2) thời gian** = trung vị số bước của các màn thành công.
- So hai mô hình bằng **kiểm định McNemar chính xác theo cặp seed**: đếm số seed chỉ A thắng và số seed chỉ B thắng
  (cùng seed = cùng vị trí ban đầu). p < 0.05 mới coi là khác biệt thật. Số bước so bằng kiểm định dấu trên các seed cả hai cùng thắng.
- Chạy từ **ảnh** (encoder đọc khối T), đúng như lúc dùng thật — không dùng trạng thái thật của bộ mô phỏng.

---

## 4. Kết quả

### 4.1. World model dự đoán thay đổi vị trí khối T (kết quả CHÍNH theo đề bài)
Tập test `sim_v2` (WM chưa từng thấy). Ngữ cảnh 2 bước thật, rồi WM tự rollout k bước bằng **action thật**. Chỉ tính các
lúc **khối T bị đẩy** (dịch > 2 đv hoặc xoay > 1° sau k bước). Mốc so sánh = "đoán khối T đứng yên".
Nguồn: `results/wm_dynamics/object_state_wm_sim_v2.json`.

| k bước | số cửa sổ | Mốc đứng yên: vị trí / góc | **WM dynamics: vị trí / góc** | Giảm sai số |
|---|---|---|---|---|
| 1 | 17.497 | 3.91 / 1.92° | 0.46 / 0.09° | 88.3% |
| 4 | 24.402 | 11.84 / 4.57° | 0.89 / 0.12° | 92.5% |
| 8 | 30.084 | 19.18 / 6.25° | 1.30 / 0.16° | 93.2% |
| 16 | 40.044 | 28.50 / 7.48° | 1.99 / 0.21° | 93.0% |

- Hướng dịch chuyển dự đoán trùng thật (cos trung vị 1.000), độ lớn đúng (tỉ lệ 0.999–1.001).
- So với WM đọc thẳng từ ảnh (DINOv2 + transformer, đã thử trước): WM ảnh chỉ giảm 47–61% sai số (k ≥ 4) và không hơn mốc ở
  k = 1, vì sai số ĐỌC tư thế từ ảnh (~2.3 đv) lớn ngang độ dịch 1 bước → tách "nhận thức" (encoder) và "động lực học"
  (dynamics trên trạng thái) tốt hơn hẳn.
- **Thử phản thực tế** (`scripts/wm_counterfactual.py`, 3000 kịch bản tự dựng, nguồn `results/wm_dynamics/counterfactual.json`):
  đoán đúng chiều xoay 93.8%; hướng trượt đúng (cos > 0.9) 89.2%; sai số sau 12 bước 7.7 đv / 3.7°; đẩy hụt → đoán khối
  đứng yên 94.5%; xếp hạng đúng cách đẩy nào hiệu quả hơn 96.6% → WM có nắm quan hệ nhân quả, không chỉ nội suy dữ liệu.
- WM encoder (`results/wm_encoder/report.json`): sai số vị trí TB 0.41 đv (trung vị 0.33, p95 1.06), góc TB 0.23°.
- Mô hình nghịch (`results/inverse/report.json`): sai số action TB 0.35 (sim_v2) / 0.51 (demo) / 0.83 (sim_v1) đv; ra lệnh
  action dự đoán trong mô phỏng thì chấm tròn bám path lệch 0.34 đv.

### 4.2. Đẩy khối T — 500 seed TEST 600000–600499
Nguồn: `results/policy/test_600000_main.json`, `test_600000_1path_no_wm_select.json`.

| Cấu hình | Thành công (≤ 300 bước) | Thời gian |
|---|---|---|
| **Mô hình chính: 16 path + WM dynamics chọn** | **89.8%** (±2.7) | **143 bước** (14.3 s) |
| Đối chứng: 1 path, không để WM chọn | 86.0% | 175.5 bước |

- WM chọn path → **thành công hơn và nhanh hơn ~32 bước**. Từng seed trên test: 61 seed chỉ mô hình chính thắng, 42 seed chỉ
  bản 1 path thắng (p = 0.076); trên tập tinh chỉnh: 71 vs 32, **p = 0.0002**.
- **Không giới hạn 300 bước** (đề bài không bắt buộc): cho chạy tới 1000 bước (100 s), cùng 500 seed test:
  **trong 300 bước 87.6% → trong 1000 bước 97.4%** (trung vị 151 bước). Phần lớn màn "thua" ở mốc 300 là hết giờ, không phải
  kẹt. (87.6% so với 89.8%: dao động do lấy mẫu ngẫu nhiên giữa hai lần chạy, nằm trong ±2.7.)

Tinh chỉnh cách chọn path (500 seed 700000, `results/policy/path_tune.log`):

| | không phạt | phạt 0.5 | phạt 1 | phạt 2 |
|---|---|---|---|---|
| 1 path (không chọn) | 81.0% / 170 | | | |
| 16 path, WM dynamics chọn | 87.8% / 140 | **88.8% / 142** | 88.0% / 139 | 87.6% / 138 |
| 32 path | | 86.2% / 131 | 87.4% / 136 | 86.0% / 134 |

→ Các cấu hình chọn khác nhau không đáng kể (trong ±2.8) → giới hạn nằm ở chất lượng path của Diffusion, không ở bộ chọn.

### 4.3. Vì sao còn thua (phân tích 51 màn thua trên test, `scripts/analyze_failures.py`, `results/failures/`)
- Chạy lại mỗi seed thua 4 lần (4 seed lấy mẫu khác nhau): 19 seed thắng 4/4, 18 thắng 3/4, 5 thắng 2/4, 2 thắng 1/4,
  **chỉ 7 seed (1.4%) luôn thua** → phần lớn thất bại là **xui khi lấy mẫu**, không phải màn khó.
- Kiểu thua: **52% sát đích nhưng không chốt** (khối lệch 2–8 đv, 1–5°, coverage 0.89–0.93 < 0.95 — demo dừng ngay khi
  đạt 95% nên Diffusion ít thấy ví dụ "chỉnh nhẹ lúc gần xong"); 31% khối bị đẩy ra mép bàn hoặc úp ngược trong góc; 17% khác.
- Encoder không phải nguyên nhân: sai số encoder chỉ tăng SAU khi khối đã bị đẩy ra khỏi khung ảnh.

### 4.4. Thí nghiệm mở rộng: khối T bắt đầu SÁT TƯỜNG / GÓC (ngoài phân phối)
Dữ liệu và test chuẩn luôn đặt tâm khối T trong vùng 100–400 (cách tường ≥ ~100 đv). Thử vị trí khối T cách tường chỉ 2–30 đv
(`pusht_wm/wall_starts.py`: 50% sát góc, 50% sát một cạnh). TEST 500 seed 800000.. (sát tường) và 600000.. (chuẩn):

| Tập | Giới hạn | Mô hình chính | Bản B (học thêm màn tự chơi thắng) | Chỉ chính thắng / chỉ B thắng |
|---|---|---|---|---|
| Sát tường | 300 bước | 55.2% | **60.4%** | 50 / 76, **p = 0.026** |
| Sát tường | 1000 bước | 72.4% | 70.4% | 45 / 35, p = 0.31 |
| Chuẩn | 300 bước | 87.6% | 85.4% | 62 / 51, p = 0.35 |
| Chuẩn | 1000 bước | **97.4%** | 95.4% | 22 / 12, p = 0.12 |

- Bản B = cùng kiến trúc, Diffusion train lại trên demo + sim_v2 + 331 màn **tự chơi** sát tường thắng ≤ 150 bước + 498 màn tự
  chơi vị trí chuẩn thắng ≤ 100 bước ("tự bắt chước có lọc"). B **nhanh hơn có ý nghĩa ở mọi tập** (3–27 bước, p < 10⁻⁵) nhưng
  không thắng nhiều hơn ở vị trí chuẩn → **giữ mô hình chính**. Checkpoint B không đưa lên repo.
- Ra ngoài phân phối, thành công tụt từ ~88% xuống ~55%; ~28% màn **kẹt hẳn** dù cho 1000 bước: khối ép vào tường, khe hở
  < 30 đv (đường kính chấm tròn) nên chấm tròn không chui vào đẩy ra được; mô hình cũng không tự nghĩ ra động tác mới ngoài demo.
- Học thêm từ màn tự chơi: nếu **không lọc** hoặc quá nhiều dữ liệu sát tường thì mô hình **quên** cách đẩy ở giữa bàn
  (vị trí chuẩn tụt tới 77%); **cân bằng** dữ liệu (lọc chặt + thêm tự chơi chuẩn) thì giữ được.

### 4.5. Các hướng đã thử nhưng không giữ

| Thử | Kết quả (test 600000 trừ khi ghi khác) | Kết luận |
|---|---|---|
| Diffusion Policy gốc đọc thẳng ảnh (ResNet18, train 100k bước trên GTX 1650) | 64.8% / 190 bước | mốc tự cài lại (train ít, chấm gắt hơn bài báo) |
| WM ảnh (DINOv2 + transformer) chọn path | 86.6% / 156 | WM ảnh sai nhiều ở khâu đọc tư thế |
| Diffusion sinh thẳng ACTION + WM chọn | 92.0% / 137 | ngang mô hình chính (p = 0.24) nhưng không theo sơ đồ sinh path |
| Diffusion path với loss FDE (sai lệch tư thế cuối) | ~67% | FDE làm trung bình hoá các cách đẩy khác nhau của demo → tệ đi |
| WM dynamics thêm loss FDE làm bộ chọn | tinh chỉnh 88.2% (vs 88.8%) | suy luận phản thực tế tốt hơn nhưng không tăng thành công |
| s_t gộp encoder + dynamics (kiểu bộ lọc Kalman, mạng gộp học bằng loss FDE) | 87.6% / 143.5 | s_t chính xác hơn 11% nhưng encoder vốn đã đủ tốt (p = 0.22) |
| Tự bắt chước có lọc (vị trí chuẩn, thắng ≤ 100 bước) | 86.2% / 133 | nhanh hơn có ý nghĩa (p = 0.03) nhưng không tăng thành công |
| Phạt WM tưởng tượng khối về phía tường | sát tường 58.3% vs 57.0% | không giúp (p = 0.67) |

### 4.6. Hạn chế và hướng phát triển (gợi ý cho báo cáo)
- **Pha chốt cuối**: phần lớn thua là sát đích nhưng không đạt 95% → lập kế hoạch bằng WM (tìm cú đẩy nhỏ) khi khối đã gần đích,
  hoặc thêm dữ liệu "chỉnh nhẹ lúc gần xong".
- **Ngoài phân phối** (sát tường): học bắt chước chỉ giỏi trong vùng dữ liệu đã thấy; cần dữ liệu "chuyên gia" cho tình huống
  mới (vd tìm chuỗi đẩy bằng chính bộ mô phỏng rồi cho Diffusion bắt chước).
- **Mô phỏng ≠ thực tế**: trong mô phỏng có tường và chấm tròn không nhấc được; robot thật có thể nhấc đầu đẩy hoặc vòng ra
  ngoài mép bàn → các thế kẹt góc sẽ giải được. (Nhóm giữ phạm vi Push-T 2D chuẩn để so sánh được với tài liệu.)
- **Tầm nhìn ngắn**: WM chỉ chấm 14 bước (1.4 s) → không chọn được kế hoạch dài "lùi ra rồi mới đẩy"; quyết định lại mỗi 0.8 s
  đôi khi làm chấm tròn "đổi ý" chạy qua chạy lại.

Gợi ý hình cho báo cáo: sơ đồ ở mục 2; bảng 4.1 (vẽ đường sai số theo k: WM vs mốc); bảng 4.2; ảnh chụp app (đường path xám,
path chọn xanh lá, khối T Diffusion muốn đạt màu cam vs khối T WM tưởng tượng màu xanh); video `results/failures/*.mp4`.
Ảnh động có sẵn: `assets/demo_grid.gif` (4 màn thành công), `assets/demo_fail_near_goal.gif` (màn thua sát đích) —
tạo lại bằng `python scripts/make_gifs.py` (đổi video `results/failures/*.mp4` sang GIF).

---

## 5. Cài đặt và chạy

```bash
python -m venv .venv
# Windows: .venv\Scripts\python.exe   |   Linux/Mac: .venv/bin/python
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements.txt
```
- Không có GPU NVIDIA: cài torch bản CPU (`pip install torch torchvision`) — app vẫn chạy được, chậm hơn.
- Máy GPU Linux thuê: `bash scripts/setup_remote.sh` (card RTX 50xx: thêm `TORCH_INDEX=https://download.pytorch.org/whl/cu130`).
- Máy không màn hình: `export SDL_VIDEODRIVER=dummy`.
- File `models/path_policy.pt` (252 MB) lưu bằng **Git LFS**: cài `git lfs install` trước khi clone, hoặc chạy `git lfs pull` sau clone.

**App xem mô hình đẩy** (cần màn hình):
```powershell
.venv\Scripts\python.exe app_pusht.py
```
Chuột trái kéo: khối T · chuột phải: chấm tròn · lăn chuột hoặc Q/E: xoay · R: vị trí ngẫu nhiên · SPACE: chạy/dừng ·
V: vẽ "suy nghĩ" · P: tạm dừng sau mỗi lần nghĩ (N: tiếp) · F: chạy nhanh · C: thử "phạt đổi ý" (thử nghiệm, chưa đánh giá) · ESC.
Khung nét đứt = vùng tâm khối T có trong dữ liệu; đặt khối ngoài khung (sát tường) là ngoài phân phối.

**Đánh giá** (nên chạy trên GPU; 500 màn ~10 phút):
```bash
python scripts/eval_state_dp.py                          # TEST 500 seed 600000.. -> outputs/eval/*.json
python scripts/eval_state_dp.py --seed-start 700000      # tập tinh chỉnh
python scripts/eval_state_dp.py --n-samples 1            # đối chứng: 1 path, không để WM chọn
python scripts/eval_state_dp.py --max-steps 1000         # cho chạy tới 1000 bước (in cả kết quả trong 300 bước)
python scripts/eval_state_dp.py --starts wall --seed-start 800000   # khối T bắt đầu sát tường
python scripts/eval_state_dp.py --n-seeds 4 --n-video 4  # chạy thử + lưu video
python scripts/analyze_failures.py                       # phân tích màn thua -> outputs/failures/
python scripts/wm_counterfactual.py                      # thử phản thực tế của WM dynamics
```

**Train lại từ đầu** (cần `data/` đầy đủ; GPU; kết quả ra `outputs/`, không ghi đè `models/`):
```bash
bash scripts/run_main.sh
```
Thứ tự trong script (thời gian tham khảo trên RTX 5060 Ti):
1. `train_pose_net.py` — WM encoder (~1 giờ); đồng thời gán lại nhãn khối T cho demo → `data/demo_state.npz`.
2. `train_state_wm.py --hidden 512 --w-move 3 --steps 40000 --batch 2048` — WM dynamics (~30 phút).
3. `train_inverse.py` — mô hình nghịch (~10 phút).
4. `train_state_dp.py --pose-net ... --state-wm ... --inverse ...` — Diffusion sinh path, 3 phần trên đóng băng (~2 giờ).
5. `eval_state_dp.py` — test 500 seed.

Thí nghiệm sát tường: `scripts/run_wall.sh` (mốc + tự chơi + train) và `scripts/run_wallB.sh` (bản B + test; cần thêm
`data/self_path.npz` — dữ liệu tự chơi vị trí chuẩn, không có trong repo). Chạy song song nhiều tiến trình rồi gộp kết quả
bằng `scripts/merge_results.py`.

---

## 6. Bản đồ code

| File | Nội dung |
|---|---|
| `pusht_wm/pose_net.py` | WM encoder (ResNet18 → x, y, sin, cos khối T) |
| `pusht_wm/state_wm.py` | WM dynamics (MLP, `step`, `rollout`), mã hoá trạng thái 6 số |
| `pusht_wm/state_dp.py` | Diffusion: điều kiện, sinh path (`path_and_actions`), chấm điểm + chọn path (`path_cost`), `act_from_images` |
| `pusht_wm/inverse.py` | mô hình nghịch |
| `pusht_wm/state_planner.py` | tính coverage khối T trên GPU, chi phí chọn path (`dense_cost_state`) |
| `pusht_wm/models.py`, `diffusion.py` | U-Net 1D có điều kiện FiLM, lịch nhiễu DDPM/DDIM, EMA |
| `pusht_wm/evaluate.py` | chạy nhiều môi trường song song, tính tỉ lệ thành công |
| `pusht_wm/controller.py` | bộ điều khiển dùng trong app |
| `pusht_wm/wall_starts.py` | sinh vị trí ban đầu sát tường/góc |
| `pusht_wm/data.py`, `sim.py`, `collect.py`, `object_metrics.py`, `paths.py` | tải demo, tạo môi trường, trạng thái mô phỏng, đo sai số WM, đường dẫn checkpoint |
| `app_pusht.py` | app pygame xem mô hình đẩy và "suy nghĩ" của nó |
