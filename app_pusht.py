"""App xem MÔ HÌNH CHÍNH đẩy khối T: tự đặt khối T + chấm tròn, bấm chạy.
Mô hình: frame -> WM -> s_t latent -> Diffusion sinh 16 PATH robot -> WM tưởng tượng, chọn -> action (toán).

    .venv\\Scripts\\python.exe app_pusht.py

Điều khiển (khi đang ĐẶT):
  Chuột trái (giữ + kéo)  : di chuyển khối T
  Chuột phải               : đặt chấm tròn (đầu đẩy)
  Lăn chuột / Q, E         : xoay khối T (5° mỗi nấc; giữ Shift: 1°)
  R                        : vị trí ngẫu nhiên
  SPACE                    : BẮT ĐẦU chạy (bấm lại để dừng)
  G                        : bắt đầu / dừng QUAY GIF (lưu assets/app_<giờ>.gif; tự lưu khi đẩy thành công)
  V                        : bật/tắt vẽ "suy nghĩ" của model (path ứng viên + khối T tưởng tượng)
  P                        : bật/tắt TẠM DỪNG sau mỗi lần suy nghĩ (bấm N để làm tiếp) -> xem kỹ từng lần
  F                        : bật/tắt chạy nhanh (không giới hạn 10 bước/giây)
  ESC                      : thoát
Tự quay GIF một màn (không cần bấm phím): APP_DEMO="<seed>:<file.gif>[:<số bước tối đa>]" python app_pusht.py
Vùng xanh lá = đích. Không giới hạn số bước: chạy tới khi thành công (coverage > 95%) hoặc bấm SPACE để dừng.
"""
import os
import sys
from pathlib import Path

if os.environ.get("SDL_VIDEODRIVER") == "dummy" and not os.environ.get("APP_HEADLESS"):  # app cần cửa sổ thật
    del os.environ["SDL_VIDEODRIVER"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

import numpy as np
import pygame

from pusht_wm.latent_controller import LATENT_POLICY, LatentController
from pusht_wm.sim import make_env

# Các bản Diffusion có thể xem (cùng sơ đồ, khác dữ liệu train). Phím M đổi bản (khi đang ĐẶT).
# (tên, checkpoint, lớp điều khiển)
MODELS = [("WM latent + Diffusion path", LATENT_POLICY, LatentController)]

# ---------------- bố cục (px) ----------------
VIEW = 600                      # bàn chơi
PAD = 16
HEAD = 64                       # thanh tiêu đề
BX, BY = PAD, HEAD + 8          # góc trên-trái bàn chơi
PANEL_X = BX + VIEW + PAD       # cột bên phải
PANEL_W = 400
BOTTOM_Y = BY + VIEW + PAD      # hàng dưới (biểu đồ + phím tắt)
BOTTOM_H = 176
WIN_W = PANEL_X + PANEL_W + PAD
WIN_H = BOTTOM_Y + BOTTOM_H + PAD
PLOT_MIN_STEPS = 300            # trục thời gian biểu đồ: tối thiểu 300 bước, tự giãn khi chạy lâu hơn

# ---------------- màu ----------------
BG = (241, 243, 248)
CARD = (255, 255, 255)
BORDER = (226, 229, 237)
SHADOW = (228, 231, 239)
TEXT = (30, 36, 50)
MUTED = (110, 117, 133)
FAINT = (160, 166, 180)
ACCENT = (79, 70, 229)          # chàm
GREEN = (22, 163, 74)
ORANGE = (234, 108, 0)
AMBER = (217, 119, 6)
RED = (220, 38, 38)
C_STATE, C_ACTION = (37, 99, 235), ORANGE      # biểu đồ: xanh = vị trí chấm tròn, cam = action
C_PICK, C_CAND = GREEN, (150, 150, 162)        # path được chọn / path ứng viên

T_PARTS = [np.array([(-60, 30), (60, 30), (60, 0), (-60, 0)], float),
           np.array([(-15, 30), (-15, 120), (15, 120), (15, 30)], float)]
T_COG = np.array([0.0, 0.0])   # trạng thái (x, y) = body.position = GỐC thân khối T (không phải trọng tâm)
# Vùng đặt khối T của dữ liệu train/test (gym-pusht: tâm khối T trong 100..400). Ngoài vùng này (vd sát tường)
# mô hình CHƯA từng thấy -> có thể đẩy khối kẹt vào tường/góc (chấm tròn chỉ đẩy, không kéo được).
TRAIN_LO, TRAIN_HI = 100.0, 400.0


def t_polygons(x, y, ang):
    """Đa giác (toạ độ trên lớp bàn chơi) của khối T có body.position (gốc thân) tại (x, y), góc ang (rad)."""
    c, s_ = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s_], [s_, c]])
    k = VIEW / 512
    return [[tuple(((np.array([x, y]) + rot @ (v - T_COG)) * k).tolist()) for v in part] for part in T_PARTS]


def to_sim(px, py):
    """Toạ độ chuột trên cửa sổ -> toạ độ mô phỏng."""
    return (float(np.clip((px - BX) * 512 / VIEW, 20, 492)), float(np.clip((py - BY) * 512 / VIEW, 20, 492)))


def on_board(pos):
    return BX <= pos[0] < BX + VIEW and BY <= pos[1] < BY + VIEW


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Push-T · World Model + Diffusion Policy")

    def F(size, bold=False):
        return pygame.font.SysFont("segoeui,arial,notosans,dejavusans", size, bold=bold)
    f_title, f_sub = F(24, True), F(14)
    f_card, f_body, f_small, f_big = F(13, True), F(15), F(13), F(30, True)
    f_key = F(12, True)
    clock = pygame.time.Clock()

    env = make_env(max_episode_steps=10**9)  # KHÔNG giới hạn số bước: chạy tới khi thành công hoặc bấm SPACE
    rng = np.random.default_rng()
    agent = [150.0, 400.0]
    block = [330.0, 330.0]
    angle_deg = 0.0
    ctrl = None
    model_i = 0                                  # bản đang dùng trong MODELS
    rec = None                                   # phím G: danh sách khung đang quay GIF (None = không quay)
    rec_k = 0
    running = fast = dragging = False
    status, steps, cov, max_cov, queue, obs = "", 0, 0.0, 0.0, [], None
    hist_state, hist_action = [], []   # motor_0 = x, motor_1 = y (giống dataset viewer)
    show_imag, pause_think, waiting = True, False, False
    imag, think_no, last_check = None, 0, None   # last_check: (coverage dự đoán, thật, lệch vị trí khối T, lệch góc °)
    read_err = None                              # sai số encoder đọc s_t (vị trí, góc °) ở lần suy nghĩ gần nhất
    thinking = False                             # đang chờ mô hình nghĩ (để tô nhãn trạng thái)

    # ---------------- quay GIF ----------------
    def capture(every=2):
        """Chụp thanh tiêu đề + bàn chơi + bảng bên phải (không gồm hàng dưới); mỗi `every` lần gọi lấy 1 khung."""
        nonlocal rec_k
        if rec is None:
            return
        rec_k += 1
        if rec_k % every or len(rec) >= 900:
            return
        from PIL import Image
        a = pygame.surfarray.array3d(screen).transpose(1, 0, 2)[:BY + VIEW + PAD // 2]
        im = Image.fromarray(a)
        rec.append(im.resize((int(im.width * 0.62), int(im.height * 0.62)), Image.LANCZOS))

    def save_gif(path=None):
        """Lưu GIF (xem nhanh x2: chụp 1 khung / 2 bước, phát 10 khung/giây)."""
        import time
        from PIL import Image
        Path("assets").mkdir(exist_ok=True)
        path = Path(path) if path else Path("assets") / time.strftime("app_%Y%m%d_%H%M%S.gif")
        fr = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in rec]
        fr[0].save(path, save_all=True, append_images=fr[1:], duration=100, loop=0, optimize=True)
        return path

    # ---------------- môi trường ----------------
    def block_now():
        u = env.unwrapped
        return float(u.block.position[0]), float(u.block.position[1]), float(u.block.angle)

    def place():
        nonlocal obs, cov, imag, think_no, last_check, read_err
        imag, think_no, last_check, read_err = None, 0, None, None
        obs, info = env.reset(options={"reset_to_state": [*agent, *block, np.deg2rad(angle_deg)]})
        hist_state.clear()
        hist_action.clear()
        hist_state.append(np.asarray(obs["agent_pos"], float))
        cov = float(env.unwrapped._get_coverage()) if hasattr(env.unwrapped, "_get_coverage") else 0.0

    def in_train_zone():
        return TRAIN_LO <= block[0] <= TRAIN_HI and TRAIN_LO <= block[1] <= TRAIN_HI

    # ---------------- vẽ cơ bản ----------------
    def rrect(surf, color, rect, r=12, width=0):
        pygame.draw.rect(surf, color, rect, width, border_radius=r)

    def card(x, y, w, h, title=None):
        """Thẻ trắng bo góc có bóng nhẹ; trả về y bắt đầu nội dung."""
        rrect(screen, SHADOW, (x, y + 2, w, h), 14)
        rrect(screen, CARD, (x, y, w, h), 14)
        rrect(screen, BORDER, (x, y, w, h), 14, 1)
        if title:
            screen.blit(f_card.render(title.upper(), True, MUTED), (x + 16, y + 12))
            return y + 36
        return y + 14

    def text(s, x, y, f=None, color=TEXT, right=False):
        img = (f or f_body).render(s, True, color)
        screen.blit(img, (x - img.get_width(), y) if right else (x, y))
        return img.get_width()

    def wrap(s, f, max_w):
        out, cur = [], ""
        for w in s.split(" "):
            t = (cur + " " + w) if cur else w
            if f.size(t)[0] > max_w and cur:
                out.append(cur)
                cur = w
            else:
                cur = t
        return out + [cur] if cur else out

    def pill(x, y, label, color, f=None):
        """Nhãn bo tròn (nền nhạt + chữ đậm cùng màu); trả về bề rộng."""
        f = f or f_card
        w = f.size(label)[0] + 26
        light = tuple(int(c + (255 - c) * 0.85) for c in color)
        rrect(screen, light, (x, y, w, 26), 13)
        pygame.draw.circle(screen, color, (x + 11, y + 13), 4)
        text(label, x + 19, y + 4, f, color)
        return w

    def dashed(surf, color, pts, dash=6, gap=4, width=2):
        """Đường nét đứt qua dãy điểm (pygame không có sẵn)."""
        acc, on = 0.0, True
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            seg = float(np.hypot(x1 - x0, y1 - y0))
            if on:
                pygame.draw.line(surf, color, (x0, y0), (x1, y1), width)
            acc += seg
            if acc >= (dash if on else gap):
                acc, on = 0.0, not on

    def dense(corners, n=120):
        """Chia các cạnh thành nhiều điểm nhỏ để nét đứt đều."""
        return [(float(x0 + (x1 - x0) * t), float(y0 + (y1 - y0) * t)) for (x0, y0), (x1, y1) in
                zip(corners[:-1], corners[1:]) for t in np.linspace(0, 1, n, endpoint=False)] + [corners[-1]]

    def keycap(x, y, key, label, state=None):
        """Phím tắt: ô phím + nhãn; state True/False tô màu bật/tắt (None: phím thường)."""
        w = max(28, f_key.size(key)[0] + 14)
        fill = (238, 240, 246) if state is None else ((220, 252, 231) if state else (243, 244, 247))
        edge = BORDER if state is None else (GREEN if state else BORDER)
        rrect(screen, (210, 214, 224), (x, y + 2, w, 22), 6)
        rrect(screen, fill, (x, y, w, 22), 6)
        rrect(screen, edge, (x, y, w, 22), 6, 1)
        kw = f_key.size(key)[0]
        text(key, x + (w - kw) // 2, y + 3, f_key, TEXT)
        text(label, x + w + 8, y + 3, f_small, GREEN if state else (MUTED if state is not None else TEXT))

    # ---------------- các khối giao diện ----------------
    def draw_header():
        pygame.draw.rect(screen, CARD, (0, 0, WIN_W, HEAD))
        pygame.draw.line(screen, BORDER, (0, HEAD - 1), (WIN_W, HEAD - 1))
        # logo: chữ T nhỏ
        rrect(screen, ACCENT, (PAD, 14, 36, 36), 10)
        pygame.draw.rect(screen, CARD, (PAD + 8, 22, 20, 6))
        pygame.draw.rect(screen, CARD, (PAD + 15, 22, 6, 20))
        text("Push-T", PAD + 48, 8, f_title, TEXT)
        text("World Model + Diffusion Policy", PAD + 48, 38, f_sub, MUTED)
        # nhãn trạng thái
        if running and waiting:
            lab, col = "Tạm dừng — bấm N", AMBER
        elif running and thinking:
            lab, col = "Đang nghĩ…", AMBER
        elif running:
            lab, col = "Đang chạy", ACCENT
        elif status.startswith("THÀNH"):
            lab, col = "Thành công", GREEN
        elif steps:
            lab, col = "Đã dừng", MUTED
        else:
            lab, col = "Đang đặt vị trí", MUTED
        x = 330
        x += pill(x, 19, lab, col) + 10
        if rec is not None:
            x += pill(x, 19, "● Đang quay GIF", RED) + 10
        msg = status if not status.startswith("THÀNH") else status.split("!")[0] + "!"
        if msg and x < WIN_W - 60:
            lines = wrap(msg, f_small, WIN_W - x - PAD)
            text(lines[0] + (" …" if len(lines) > 1 else ""), x, 23, f_small, MUTED)

    def draw_board():
        rrect(screen, SHADOW, (BX - 4, BY - 2, VIEW + 8, VIEW + 8), 16)
        rrect(screen, CARD, (BX - 4, BY - 4, VIEW + 8, VIEW + 8), 16)
        u = env.unwrapped                                       # vẽ KHÔNG kèm dấu + của action
        frame = u._get_img(u._draw(), width=VIEW, height=VIEW, render_action=False)
        board = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        k = VIEW / 512                                          # khung nét đứt: vùng tâm khối T có trong dữ liệu
        a0, a1 = TRAIN_LO * k, TRAIN_HI * k
        dashed(board, (214, 217, 226), dense([(a0, a0), (a1, a0), (a1, a1), (a0, a1), (a0, a0)]), 12, 8, 2)
        draw_imagination(board)
        if not running and steps == 0 and not in_train_zone():  # cảnh báo khi ĐANG ĐẶT ngoài vùng dữ liệu
            band = pygame.Surface((VIEW, 30), pygame.SRCALPHA)
            band.fill((*AMBER, 215))
            board.blit(band, (0, 0))
            t = f_small.render("Tâm khối T ngoài vùng dữ liệu (khung nét đứt) — sát tường dễ bị kẹt", True, (255, 255, 255))
            board.blit(t, ((VIEW - t.get_width()) // 2, 6))
        mask = pygame.Surface((VIEW, VIEW), pygame.SRCALPHA)    # bo góc bàn chơi
        rrect(mask, (255, 255, 255, 255), (0, 0, VIEW, VIEW), 12)
        board = board.convert_alpha()
        board.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        screen.blit(board, (BX, BY))
        rrect(screen, BORDER, (BX, BY, VIEW, VIEW), 12, 1)

    def draw_imagination(surf):
        """Path ứng viên (xám) + khối T WM TƯỞNG TƯỢNG ở cuối mỗi path; path được chọn xanh lá (đoạn sẽ thực thi
        vẽ dày) + vết tâm khối T dự đoán; khối T mà path của Diffusion MUỐN đạt: viền cam."""
        if imag is None or not show_imag:
            return
        k = VIEW / 512
        layer = pygame.Surface((VIEW, VIEW), pygame.SRCALPHA)
        paths, pick, pose = imag["paths"], imag["pick"], imag["pose"]
        order = [i for i in range(len(paths)) if i != pick] + [pick]   # path được chọn vẽ sau cùng (nằm trên)
        for i in order:
            chosen = i == pick
            col = C_PICK if chosen else C_CAND
            pts = [(float(x * k), float(y * k)) for x, y in paths[i]]
            if len(pts) > 1:
                pygame.draw.lines(layer, (*col, 230 if chosen else 80), False, pts, 3 if chosen else 1)
            if chosen and len(pts) > imag["act"]:
                pygame.draw.lines(layer, (*col, 255), False, pts[:imag["act"] + 1], 5)
            if pose is not None:
                ax, ay, bx, by, sn, cs = pose[i, -1]
                ang = float(np.arctan2(sn, cs))
                for poly in t_polygons(bx * 512, by * 512, ang):
                    if chosen:
                        pygame.draw.polygon(layer, (*C_PICK, 60), poly)
                    pygame.draw.polygon(layer, (*col, 230 if chosen else 90), poly, 2 if chosen else 1)
                pygame.draw.circle(layer, (*col, 200 if chosen else 80), (float(ax * VIEW), float(ay * VIEW)),
                                   6 if chosen else 3)
        plan = imag.get("plan")
        if plan is not None:
            _, _, bx, by, sn, cs = plan[pick, -1]
            for poly in t_polygons(bx * 512, by * 512, float(np.arctan2(sn, cs))):
                pygame.draw.polygon(layer, (*C_ACTION, 220), poly, 2)
            trail = [(float(b[2] * VIEW), float(b[3] * VIEW)) for b in plan[pick]]
            if len(trail) > 1:
                pygame.draw.lines(layer, (*C_ACTION, 200), False, trail, 2)
        if pose is not None:
            trail = [(float(b[2] * VIEW), float(b[3] * VIEW)) for b in pose[pick]]
            if len(trail) > 1:
                pygame.draw.lines(layer, (*C_PICK, 200), False, trail, 2)
        surf.blit(layer, (0, 0))

    def draw_state_card(x, y, w):
        h = 192
        cy = card(x, y, w, h, "Trạng thái")
        text(MODELS[model_i][0], x + 16, cy, f_body, ACCENT)
        if len(MODELS) > 1:
            text("M: đổi bản", x + w - 16, cy, f_small, FAINT, right=True)
        cy += 26
        text(f"{steps / 10:.1f} s", x + 16, cy - 2, f_big, TEXT)
        text(f"{steps} bước · 10 bước = 1 giây", x + 16 + f_big.size(f"{steps / 10:.1f} s")[0] + 12, cy + 12, f_small, MUTED)
        cy += 46
        # thanh coverage: hiện tại (đặc), cao nhất (vạch), ngưỡng thắng 95% (vạch xanh)
        bw = w - 32
        text("Coverage (khối T phủ vùng đích)", x + 16, cy, f_small, MUTED)
        text(f"{cov * 100:.1f}%", x + w - 16, cy - 1, f_body, GREEN if cov > 0.95 else TEXT, right=True)
        cy += 22
        rrect(screen, (236, 238, 244), (x + 16, cy, bw, 10), 5)
        if cov > 0:
            rrect(screen, GREEN if cov > 0.95 else ACCENT, (x + 16, cy, max(10, int(bw * min(cov, 1))), 10), 5)
        mx = x + 16 + int(bw * min(max_cov, 1))
        pygame.draw.line(screen, TEXT, (mx, cy - 3), (mx, cy + 12), 2)
        tx = x + 16 + int(bw * 0.95)
        pygame.draw.line(screen, GREEN, (tx, cy - 5), (tx, cy + 14), 2)
        cy += 18
        text(f"cao nhất {max_cov * 100:.1f}%", x + 16, cy, f_small, MUTED)
        text("ngưỡng thắng 95%", x + w - 16, cy, f_small, GREEN, right=True)
        cy += 22
        text(f"Góc khối T: {angle_deg:.0f}°  (đích 45°)", x + 16, cy, f_small, MUTED)
        return y + h

    def draw_think_card(x, y, w):
        h = 224
        cy = card(x, y, w, h, "Mô hình suy nghĩ")
        if imag is None:
            for i, ln in enumerate(wrap("Mỗi 0.8 s: WM đọc ảnh ra s_t → Diffusion đề xuất 16 đường đi cho robot → "
                                        "WM tưởng tượng hậu quả của từng đường → chọn đường tốt nhất.",
                                        f_body, w - 32)):
                text(ln, x + 16, cy + i * 22, f_body, MUTED)
            text("Bấm SPACE để bắt đầu.", x + 16, cy + 4 * 22 + 8, f_body, ACCENT)
            return y + h
        i, K = imag["pick"], imag["pose"].shape[1]
        _, _, bx, by, sn, cs = imag["pose"][i, -1]
        d = float(np.hypot(bx * 512 - 256, by * 512 - 256))
        da = abs((np.degrees(np.arctan2(sn, cs)) - 45 + 180) % 360 - 180)
        sc = imag["score"]
        rows = [("Lần nghĩ", f"#{think_no}"),
                ("Path được chọn", f"#{i + 1} / {len(imag['paths'])}"),
                (f"Coverage tưởng tượng (+{K} bước)", f"{imag['cov'][i, -1] * 100:.0f}%"),
                ("Khối T tưởng tượng → đích", f"{d:.0f} đv · {da:.0f}°"),
                ("Điểm: chọn / tệ nhất", f"{sc[i]:.2f} / {sc.min():.2f}")]
        if last_check:
            rows.append(("Lần trước: đoán → thật", f"{last_check[0] * 100:.0f}% → {last_check[1] * 100:.0f}%"))
            rows.append(("WM tưởng tượng lệch thật", f"{last_check[2]:.1f} đv · {last_check[3]:.1f}°"))
        for j, (k_, v) in enumerate(rows):
            ry = cy + j * 23
            if j:
                pygame.draw.line(screen, (242, 243, 247), (x + 16, ry - 4), (x + w - 16, ry - 4))
            text(k_, x + 16, ry, f_small, MUTED)
            text(v, x + w - 16, ry - 1, f_body, TEXT, right=True)
        return y + h

    def draw_legend_card(x, y, w):
        h = BY + VIEW + 4 - y
        cy = card(x, y, w, h, "Chú thích")
        items = [("line", C_CAND, "16 đường đi robot Diffusion đề xuất"),
                 ("thick", C_PICK, "đường được chọn (đậm: 8 bước sẽ làm)"),
                 ("tee_g", C_PICK, "khối T WM TƯỞNG TƯỢNG (cuối đường)"),
                 ("dash", (190, 194, 206), "vùng dữ liệu train/test")]
        for j, (kind, col, lab) in enumerate(items):
            ix, iy = x + 16, cy + j * 23
            if kind == "line":
                pygame.draw.line(screen, col, (ix, iy + 9), (ix + 22, iy + 9), 2)
            elif kind == "thick":
                pygame.draw.line(screen, col, (ix, iy + 9), (ix + 22, iy + 9), 5)
            elif kind == "dash":
                dashed(screen, col, dense([(ix, iy + 9), (ix + 22, iy + 9)], 20), 5, 4, 2)
            else:
                pygame.draw.rect(screen, col, (ix + 2, iy + 3, 18, 5), 0 if kind == "tee_g" else 2)
                pygame.draw.rect(screen, col, (ix + 8, iy + 3, 6, 13), 0 if kind == "tee_g" else 2)
            text(lab, ix + 34, iy + 1, f_small, TEXT)

    def draw_plots(x, y, w, h):
        """Biểu đồ vị trí chấm tròn (x, y) theo thời gian: liền xanh = thực tế, nét đứt cam = action."""
        cy = card(x, y, w, h, "Chuyển động chấm tròn")
        lx = x + w - 16
        pygame.draw.line(screen, C_STATE, (lx - 196, y + 19), (lx - 178, y + 19), 2)
        text("vị trí", lx - 172, y + 11, f_small, MUTED)
        dashed(screen, C_ACTION, dense([(lx - 116, y + 19), (lx - 96, y + 19)], 20), 4, 3, 2)
        text("action (lệnh)", lx - 90, y + 11, f_small, MUTED)
        S = np.array(hist_state) if hist_state else np.zeros((0, 2))
        A = np.array(hist_action) if hist_action else np.zeros((0, 2))
        ph = (y + h - 12 - cy) // 2
        x0, x1 = x + 92, x + w - 64
        span = max(PLOT_MIN_STEPS, len(S) - 1)
        for j, name in enumerate(("x", "y")):
            top = cy + j * ph
            vals = np.concatenate([S[:, j], A[:, j]]) if len(S) + len(A) else np.array([0.0, 512.0])
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 20:
                lo, hi = lo - 10, hi + 10
            text(name, x + 18, top + ph // 2 - 20, F(16, True), TEXT)
            text(f"{lo:.0f}–{hi:.0f}", x + 18, top + ph // 2 + 2, f_small, FAINT)
            rrect(screen, (248, 249, 252), (x0, top, x1 - x0, ph - 6), 6)
            ys = lambda v: top + 4 + (1 - (v - lo) / (hi - lo)) * (ph - 14)  # noqa: E731
            xs = lambda t: x0 + (x1 - x0) * t / span  # noqa: E731
            if len(S) > 1:
                pygame.draw.lines(screen, C_STATE, False, [(xs(t), ys(v)) for t, v in enumerate(S[:, j])], 2)
            if len(A) > 1:
                dashed(screen, C_ACTION, [(xs(t), ys(v)) for t, v in enumerate(A[:, j])])
            if len(S):
                text(f"{S[-1, j]:.0f}", x + w - 16, top + ph // 2 - 10, f_body, C_STATE, right=True)

    def draw_keys(x, y, w, h):
        cy = card(x, y, w, h, "Phím tắt")
        keys = [("SPACE", "chạy / dừng", None), ("R", "vị trí ngẫu nhiên", None),
                ("V", "vẽ suy nghĩ", show_imag), ("P", "dừng sau mỗi lần nghĩ", pause_think),
                ("F", "chạy nhanh", fast), ("G", "quay GIF", rec is not None),
                ("Q/E", "xoay khối T", None), ("N", "đi tiếp (khi P bật)", None)]
        cw = (w - 32) // 2
        for j, (k_, lab, st) in enumerate(keys):
            keycap(x + 16 + (j % 2) * cw, cy + (j // 2) * 28, k_, lab, st)
        text("Chuột trái kéo: khối T · chuột phải: chấm tròn · ESC: thoát", x + 16, y + h - 24, f_small, FAINT)

    def redraw():
        screen.fill(BG)
        draw_header()
        draw_board()
        y = draw_state_card(PANEL_X, BY - 4, PANEL_W) + 12
        y = draw_think_card(PANEL_X, y, PANEL_W) + 12
        draw_legend_card(PANEL_X, y, PANEL_W)
        draw_plots(BX - 4, BOTTOM_Y, VIEW + 8, BOTTOM_H)
        draw_keys(PANEL_X, BOTTOM_Y, PANEL_W, BOTTOM_H)
        pygame.display.flip()

    if os.environ.get("APP_HEADLESS"):           # chỉ để chụp ảnh giao diện (không nạp mô hình)
        place()
        status = "Đặt vị trí rồi bấm SPACE"
        redraw()
        pygame.image.save(screen, os.environ["APP_HEADLESS"])
        return

    place()
    status = "Đặt vị trí rồi bấm SPACE"
    # Chế độ tự quay GIF (không cần bấm phím): APP_DEMO="<seed>:<file.gif>[:<số bước tối đa>]"
    # -> vị trí ban đầu chuẩn của gym-pusht theo seed, tự chạy, quay, lưu GIF rồi thoát.
    demo = os.environ.get("APP_DEMO")
    demo_out, demo_max = None, 10**9
    if demo:
        parts = demo.split(":")
        demo_out = parts[1]
        demo_max = int(parts[2]) if len(parts) > 2 else 300
        o, _ = env.reset(seed=int(parts[0]))
        u = env.unwrapped
        agent = [float(v) for v in o["agent_pos"]]
        tgt, ang = np.array(u.block.position, float), float(u.block.angle)
        # reset_to_state hiểu (x, y) khối T theo gốc khác body.position -> đo độ lệch ở đúng góc này rồi bù lại
        env.reset(options={"reset_to_state": [*agent, *tgt, ang]})
        off = np.array(u.block.position, float) - tgt
        block = list(tgt - off)
        angle_deg = float(np.degrees(ang)) % 360
        ctrl = MODELS[model_i][2](MODELS[model_i][1])
        place()
        ctrl.reset(obs)
        running, steps, max_cov, queue, waiting, fast = True, 0, cov, [], False, True
        rec, rec_k = [], 0
        status = "Đang chạy..."
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN:
                shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                if ev.key == pygame.K_SPACE:
                    if running:
                        running = False
                        status = f"Đã dừng ở bước {steps}, coverage cao nhất {max_cov*100:.1f}%. Bấm SPACE để chạy lại"
                    else:
                        if ctrl is None:
                            status = "Đang nạp mô hình (lần đầu mất vài giây)..."
                            redraw()
                            ctrl = MODELS[model_i][2](MODELS[model_i][1])
                        place()
                        ctrl.reset(obs)   # place() đã xóa biểu đồ cũ
                        running, steps, max_cov, queue, waiting = True, 0, cov, [], False
                        status = "Đang chạy..."
                elif not running:
                    if ev.key == pygame.K_m and len(MODELS) > 1:
                        model_i = (model_i + 1) % len(MODELS)
                        ctrl = None                      # nạp lại khi bấm SPACE
                        status = f"Đã chọn: {MODELS[model_i][0]} — bấm SPACE để chạy"
                    elif ev.key in (pygame.K_q, pygame.K_e):
                        angle_deg = (angle_deg + (1 if shift else 5) * (1 if ev.key == pygame.K_e else -1)) % 360
                        place()
                    elif ev.key == pygame.K_r:
                        agent = list(rng.uniform(50, 450, 2))
                        block = list(rng.uniform(100, 400, 2))
                        angle_deg = float(rng.uniform(0, 360))
                        steps, max_cov = 0, 0.0
                        place()
                if ev.key == pygame.K_g:
                    if rec is None:
                        rec, rec_k = [], 0
                        status = "Đang quay GIF — bấm G để dừng và lưu"
                    else:
                        status = f"Đã lưu GIF: {save_gif()}" if rec else "Chưa có khung nào"
                        rec = None
                elif ev.key == pygame.K_f:
                    fast = not fast
                elif ev.key == pygame.K_v:
                    show_imag = not show_imag
                elif ev.key == pygame.K_p:
                    pause_think = not pause_think
                    if not pause_think:
                        waiting = False
                elif ev.key == pygame.K_n and waiting:
                    waiting = False
            if not running:
                if ev.type == pygame.MOUSEBUTTONDOWN and on_board(ev.pos):
                    if ev.button == 1:
                        dragging = True
                        block = list(to_sim(*ev.pos))
                    elif ev.button == 3:
                        agent = list(to_sim(*ev.pos))
                    steps, max_cov = 0, 0.0
                    place()
                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    dragging = False
                elif ev.type == pygame.MOUSEMOTION and dragging and on_board(ev.pos):
                    block = list(to_sim(*ev.pos))
                    place()
                elif ev.type == pygame.MOUSEWHEEL:
                    shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                    angle_deg = (angle_deg + ev.y * (1 if shift else 5)) % 360
                    place()

        if running and waiting:          # tạm dừng sau lần suy nghĩ: chờ phím N
            redraw()
            clock.tick(30)
            continue
        if running:
            if not queue:
                if imag is not None and imag["cov"] is not None:   # coverage đã đoán (sau các action đã làm) vs thật
                    k = min(imag["act"], imag["pose"].shape[1]) - 1
                    _, _, pbx, pby, psn, pcs = imag["pose"][imag["pick"], k]
                    rbx, rby, rang = block_now()
                    dang = abs((np.degrees(np.arctan2(psn, pcs) - rang) + 180) % 360 - 180)
                    last_check = (float(imag["cov"][imag["pick"], k]), cov,
                                  float(np.hypot(pbx * 512 - rbx, pby * 512 - rby)), float(dang))
                thinking = True
                redraw()
                queue = list(ctrl.next_actions())
                thinking = False
                imag, think_no = ctrl.imagination, think_no + 1
                status = "Đang chạy..."
                if pause_think:
                    waiting = True
                    status = "Tạm dừng: xem suy nghĩ rồi bấm N"
                    continue
            a = queue.pop(0)
            obs, _, term, _, info = env.step(a)
            ctrl.observe(obs, a)
            hist_action.append(np.asarray(a, float))
            hist_state.append(np.asarray(obs["agent_pos"], float))
            steps += 1
            cov = float(info["coverage"])
            max_cov = max(max_cov, cov)
            if term:
                running, queue = False, []
                status = f"THÀNH CÔNG sau {steps} bước ({steps/10:.1f} giây)!"
                redraw()
                if rec is not None and rec:                 # đang quay: giữ khung cuối 1.5 s rồi tự lưu
                    for _ in range(30):
                        capture(every=2)
                    status += f" Đã lưu GIF: {save_gif(demo_out)}"
                    rec = None
                    redraw()
                if demo:
                    print(f"THẮNG {steps} bước -> {demo_out}")
                    pygame.quit()
                    return
                continue
            redraw()
            capture()
            if demo and steps >= demo_max:                  # chế độ tự quay: hết số bước -> lưu và thoát
                status = f"Hết {steps} bước, coverage cao nhất {max_cov*100:.1f}%"
                redraw()
                for _ in range(30):
                    capture(every=2)
                save_gif(demo_out)
                print(f"THUA (coverage cao nhất {max_cov:.3f}) -> {demo_out}")
                pygame.quit()
                return
            if not fast:
                clock.tick(10)  # đúng tốc độ mô phỏng: 10 bước/giây
        else:
            redraw()
            clock.tick(30)


if __name__ == "__main__":
    main()
