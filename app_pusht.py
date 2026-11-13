"""App xem MÔ HÌNH CHÍNH đẩy khối T: tự đặt khối T + chấm tròn, bấm chạy.
Mô hình: frame -> WM encoder đọc s_t -> Diffusion sinh 16 PATH -> WM dynamics chọn -> mô hình nghịch -> action.
(App cũ có mọi chế độ thí nghiệm: archive/app_pusht_old.py)

    .venv\\Scripts\\python.exe app_pusht.py

Điều khiển (khi đang ĐẶT):
  Chuột trái (giữ + kéo)  : di chuyển khối T
  Chuột phải               : đặt chấm tròn (đầu đẩy)
  Lăn chuột / Q, E         : xoay khối T (5° mỗi nấc; giữ Shift: 1°)
  R                        : vị trí ngẫu nhiên
  M                        : đổi bản mô hình (mô hình chính <-> bản B)
  C                        : bật/tắt phạt đổi ý (kiên trì một kế hoạch; thử nghiệm)
  SPACE                    : BẮT ĐẦU chạy (bấm lại để dừng)
  V                        : bật/tắt vẽ "suy nghĩ" của model (path ứng viên + khối T tưởng tượng)
  P                        : bật/tắt TẠM DỪNG sau mỗi lần suy nghĩ (bấm N để làm tiếp) -> xem kỹ từng lần
  F                        : bật/tắt chạy nhanh (không giới hạn 10 bước/giây)
  ESC                      : thoát
Vùng xanh lá = đích. Không giới hạn số bước: chạy tới khi thành công (coverage > 95%) hoặc bấm SPACE để dừng.
"""
import os
import sys
from pathlib import Path

if os.environ.get("SDL_VIDEODRIVER") == "dummy":  # app cần cửa sổ thật
    del os.environ["SDL_VIDEODRIVER"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

import numpy as np
import pygame

from pusht_wm.controller import DESC, Controller
from pusht_wm.paths import PATH_POLICY

# Các bản Diffusion có thể xem (cùng sơ đồ, khác dữ liệu train). Phím M đổi bản (khi đang ĐẶT).
MODELS = [("Mô hình chính", PATH_POLICY),
          ("Bản B: + tự chơi sát tường/chuẩn (nhanh hơn)", "outputs/wall/path_wallB/ckpt_best.pt")]
MODELS = [m for m in MODELS if Path(m[1]).exists()]
from pusht_wm.sim import make_env

VIEW = 600          # kích thước bàn chơi hiển thị (px)
PANEL = 360         # bảng thông tin bên phải
PLOT_H = 115        # chiều cao mỗi biểu đồ motor
PLOTS = 2 * PLOT_H + 16
PLOT_MIN_STEPS = 300  # trục thời gian của biểu đồ: tối thiểu 300 bước, tự giãn khi chạy lâu hơn
C_STATE, C_ACTION = (37, 99, 235), (234, 108, 0)   # xanh = observation.state, cam = action
T_PARTS = [np.array([(-60, 30), (60, 30), (60, 0), (-60, 0)], float),
           np.array([(-15, 30), (-15, 120), (15, 120), (15, 30)], float)]
T_COG = np.array([0.0, 0.0])   # trạng thái (x, y) = body.position = GỐC thân khối T (không phải trọng tâm)
C_PICK = (22, 163, 74)          # xanh lá: path được chọn
C_CAND = (150, 150, 160)        # xám: các path ứng viên khác
# Vùng đặt khối T của dữ liệu train/test (gym-pusht: tâm khối T trong 100..400). Ngoài vùng này (vd sát tường)
# mô hình CHƯA từng thấy -> có thể đẩy khối kẹt vào tường/góc (chấm tròn chỉ đẩy, không kéo được).
TRAIN_LO, TRAIN_HI = 100.0, 400.0


def t_polygons(x, y, ang):
    """Đa giác (toạ độ màn hình) của khối T có body.position (gốc thân) tại (x, y), góc ang (rad)."""
    c, s_ = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s_], [s_, c]])
    k = VIEW / 512
    return [[tuple(((np.array([x, y]) + rot @ (v - T_COG)) * k).tolist()) for v in part] for part in T_PARTS]


def to_sim(px, py):
    return float(np.clip(px * 512 / VIEW, 20, 492)), float(np.clip(py * 512 / VIEW, 20, 492))


def main():
    pygame.init()
    screen = pygame.display.set_mode((VIEW + PANEL, VIEW + PLOTS))
    pygame.display.set_caption("Push-T: Diffusion Policy + World Model")
    font = pygame.font.SysFont("segoeui,arial", 18)
    small = pygame.font.SysFont("segoeui,arial", 15)
    big = pygame.font.SysFont("segoeui,arial", 26, bold=True)
    clock = pygame.time.Clock()

    env = make_env(max_episode_steps=10**9)  # KHÔNG giới hạn số bước: chạy tới khi thành công hoặc bấm SPACE
    rng = np.random.default_rng()
    agent = [150.0, 400.0]
    block = [330.0, 330.0]
    angle_deg = 0.0
    ctrl = None
    model_i = 0                                  # bản đang dùng trong MODELS
    consistency = False                          # phím C: phạt đổi ý (thử nghiệm, chưa đánh giá)
    running = fast = dragging = False
    status, steps, cov, max_cov, queue, obs = "", 0, 0.0, 0.0, [], None
    hist_state, hist_action = [], []   # motor_0 = x, motor_1 = y (giống dataset viewer)
    show_imag, pause_think, waiting = True, False, False
    imag, think_no, last_check = None, 0, None   # last_check: (coverage dự đoán, thật, lệch vị trí khối T, lệch góc °)
    read_err = None                              # sai số encoder đọc s_t (vị trí, góc °) ở lần suy nghĩ gần nhất

    def block_now():
        u = env.unwrapped
        return float(u.block.position[0]), float(u.block.position[1]), float(u.block.angle)

    def place():
        nonlocal obs, cov, imag, think_no, last_check
        imag, think_no, last_check = None, 0, None
        obs, info = env.reset(options={"reset_to_state": [*agent, *block, np.deg2rad(angle_deg)]})
        hist_state.clear()
        hist_action.clear()
        hist_state.append(np.asarray(obs["agent_pos"], float))
        cov = float(env.unwrapped._get_coverage()) if hasattr(env.unwrapped, "_get_coverage") else 0.0

    def draw_text(lines, x, y, f=font, color=(30, 30, 30), gap=24, max_w=PANEL - 28):
        """Vẽ từng dòng, TỰ XUỐNG DÒNG nếu dài hơn bảng (tránh chữ tràn khỏi cửa sổ)."""
        for ln in lines:
            indent = len(ln) - len(ln.lstrip(" "))
            words, cur = ln.split(" "), ""
            for w in words:
                trial = (cur + " " + w) if cur else w
                if f.size(trial)[0] > max_w and cur.strip():
                    screen.blit(f.render(cur, True, color), (x, y))
                    y += gap
                    cur = " " * (indent + 2) + w
                else:
                    cur = trial
            screen.blit(f.render(cur, True, color), (x, y))
            y += gap
        return y

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

    def draw_plots():
        """Biểu đồ motor_0 (x) / motor_1 (y): state liền màu xanh, action nét đứt màu cam."""
        top = VIEW
        W = VIEW + PANEL
        pygame.draw.rect(screen, (255, 255, 255), (0, top, W, PLOTS))  # nền trắng, đồng bộ với bàn chơi
        pygame.draw.line(screen, (210, 212, 220), (0, top), (W, top), 2)
        screen.blit(small.render("Joints     —— observation.state      - - - action", True, (90, 90, 100)), (12, top + 2))
        S = np.array(hist_state) if hist_state else np.zeros((0, 2))
        A = np.array(hist_action) if hist_action else np.zeros((0, 2))
        x0, x1 = 230, W - 110
        for j in range(2):
            y_top = top + 20 + j * PLOT_H
            pygame.draw.line(screen, (225, 227, 233), (0, y_top), (W, y_top), 1)
            vals = np.concatenate([S[:, j], A[:, j]]) if len(S) + len(A) else np.array([0.0, 512.0])
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 20:
                lo, hi = lo - 10, hi + 10
            screen.blit(font.render(f"motor_{j}", True, (30, 30, 30)), (12, y_top + 14))
            screen.blit(small.render(f"{lo:.1f} - {hi:.1f}", True, (120, 120, 130)), (12, y_top + 40))
            ys = lambda v: y_top + 12 + (1 - (v - lo) / (hi - lo)) * (PLOT_H - 30)  # noqa: E731
            span = max(PLOT_MIN_STEPS, len(S) - 1)
            xs = lambda t: x0 + (x1 - x0) * t / span  # noqa: E731
            if len(S) > 1:
                pygame.draw.lines(screen, C_STATE, False, [(xs(t), ys(v)) for t, v in enumerate(S[:, j])], 2)
            if len(A) > 1:
                dashed(screen, C_ACTION, [(xs(t), ys(v)) for t, v in enumerate(A[:, j])])
            if steps:
                pygame.draw.line(screen, (170, 170, 180), (xs(steps), y_top + 6), (xs(steps), y_top + PLOT_H - 12), 1)
            if len(S):
                screen.blit(font.render(f"{S[-1, j]:.1f}", True, C_STATE), (W - 95, y_top + 14))
            if len(A):
                screen.blit(font.render(f"{A[-1, j]:.1f}", True, C_ACTION), (W - 95, y_top + 40))

    def draw_imagination():
        """Vẽ các path ứng viên (xám mờ) + khối T world model TƯỞNG TƯỢNG ở cuối mỗi path;
        path được chọn: xanh lá đậm (đoạn sẽ thực thi thật vẽ dày hơn) + khối T tưởng tượng tô xanh
        + vết đường đi dự đoán của tâm khối T."""
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
                pygame.draw.lines(layer, (*col, 230 if chosen else 90), False, pts, 3 if chosen else 1)
            if chosen and len(pts) > imag["act"]:
                pygame.draw.lines(layer, (*col, 255), False, pts[:imag["act"] + 1], 5)
            if pose is not None:
                ax, ay, bx, by, sn, cs = pose[i, -1]
                ang = float(np.arctan2(sn, cs))
                for poly in t_polygons(bx * 512, by * 512, ang):
                    if chosen:
                        pygame.draw.polygon(layer, (*C_PICK, 70), poly)
                    pygame.draw.polygon(layer, (*col, 230 if chosen else 110), poly, 2 if chosen else 1)
                pygame.draw.circle(layer, (*col, 200 if chosen else 90),
                                   (float(ax * VIEW), float(ay * VIEW)), 6 if chosen else 4)
        plan = imag.get("plan")
        if plan is not None:      # khối T mà PATH của Diffusion MUỐN đạt (cam) để so với WM tưởng tượng (xanh)
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
        screen.blit(layer, (0, 0))

    def imagination_lines():
        if imag is None:
            return []
        n = len(imag["paths"])
        head = []
        if read_err is not None:
            head.append(f"WM encoder đọc s_t: lệch {read_err[0]:.1f} đv, {read_err[1]:.1f}°")
        i, K = imag["pick"], imag["pose"].shape[1]
        ax, ay, bx, by, sn, cs = imag["pose"][i, -1]
        d = float(np.hypot(bx * 512 - 256, by * 512 - 256))
        da = abs((np.degrees(np.arctan2(sn, cs)) - 45 + 180) % 360 - 180)
        sc = imag["score"]
        out = head + [f"Lần {think_no}: xét {n} path, chọn path #{i + 1}",
               f"  Tưởng tượng sau {K} bước: coverage {imag['cov'][i, -1] * 100:.0f}%",
               f"  khối T cách đích {d:.0f} đv, lệch góc {da:.0f}°",
               f"  Điểm: chọn {sc[i]:.2f} | tệ nhất {sc.min():.2f}"]
        if last_check:
            out.append(f"  Lần trước: đoán {last_check[0] * 100:.0f}% → thật {last_check[1] * 100:.0f}%")
            out.append(f"  khối T tưởng tượng lệch thật {last_check[2]:.1f} đv, {last_check[3]:.1f}°")
        return out

    def redraw(msg_color=(30, 30, 30)):
        u = env.unwrapped                                       # vẽ KHÔNG kèm dấu + của action
        frame = u._get_img(u._draw(), width=VIEW, height=VIEW, render_action=False)
        screen.blit(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), (0, 0))
        k = VIEW / 512                                          # khung nét đứt: vùng tâm khối T có trong dữ liệu
        a0, a1 = TRAIN_LO * k, TRAIN_HI * k
        corners = [(a0, a0), (a1, a0), (a1, a1), (a0, a1), (a0, a0)]          # chia cạnh thành điểm nhỏ -> nét đứt đều
        pts = [(float(x0 + (x1 - x0) * t), float(y0 + (y1 - y0) * t)) for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:])
               for t in np.linspace(0, 1, 120, endpoint=False)] + [corners[-1]]
        dashed(screen, (222, 222, 230), pts, dash=12, gap=8, width=3)
        draw_imagination()
        pygame.draw.rect(screen, (245, 246, 250), (VIEW, 0, PANEL, VIEW))
        x, y = VIEW + 16, 16
        y = draw_text(["Push-T"], x, y, big, (40, 40, 120), 32)
        y = draw_text(["Diffusion Policy + World Model"], x, y, small, (90, 90, 140), 26)
        y = draw_text([f"Bản: {MODELS[model_i][0]}", DESC, f"Bước: {steps}  ({steps/10:.1f} giây)",
                       f"Coverage: {cov*100:.1f}%  (cao nhất {max_cov*100:.1f}%)"], x, y)
        y += 6
        y = draw_text([status], x, y, font, msg_color)
        if not running and not (TRAIN_LO <= block[0] <= TRAIN_HI and TRAIN_LO <= block[1] <= TRAIN_HI):
            y = draw_text(["⚠ Tâm khối T ngoài khung nét đứt (vùng dữ liệu train/test): mô hình chưa học vị trí sát tường, "
                           "dễ đẩy kẹt."], x, y, small, (200, 90, 0), 20)
        y += 4
        il = imagination_lines()
        if il:
            y = draw_text(["MODEL SUY NGHĨ:"], x, y, font, C_PICK, 22)
            y = draw_text(il, x, y, small, (40, 40, 40), 20) + 6
        draw_text(["Chuột trái: khối T · phải: chấm tròn", "Lăn chuột / Q,E: xoay · R: ngẫu nhiên",
                   "M: đổi bản mô hình · SPACE: chạy/dừng",
                   f"C: phạt đổi ý ({'BẬT' if consistency else 'tắt'}, thử nghiệm)",
                   f"V: vẽ suy nghĩ ({'BẬT' if show_imag else 'tắt'}) · P: tạm dừng ({'BẬT' if pause_think else 'tắt'})",
                   f"F: chạy nhanh ({'BẬT' if fast else 'tắt'}) · ESC: thoát",
                   f"Góc khối T: {angle_deg:.0f}° (đích 45°)"], x, y, small, (90, 90, 90), 20)
        draw_plots()
        pygame.display.flip()

    place()
    status = "Đặt vị trí rồi bấm SPACE"
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
                            status = "Đang nạp model (lần đầu mất vài giây)..."
                            redraw()
                            ctrl = Controller(MODELS[model_i][1])
                        ctrl.consistency = consistency
                        place()
                        ctrl.reset(obs)   # place() đã xóa biểu đồ cũ
                        running, steps, max_cov, queue, waiting = True, 0, cov, [], False
                        status = "Đang chạy..."
                elif not running:
                    if ev.key == pygame.K_c:
                        consistency = not consistency
                        if ctrl is not None:
                            ctrl.consistency = consistency
                        status = f"Phạt đổi ý (kiên trì 1 kế hoạch): {'BẬT' if consistency else 'tắt'}"
                    elif ev.key == pygame.K_m and len(MODELS) > 1:
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
                if ev.key == pygame.K_f:
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
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.pos[0] < VIEW:
                    if ev.button == 1:
                        dragging = True
                        block = list(to_sim(*ev.pos))
                    elif ev.button == 3:
                        agent = list(to_sim(*ev.pos))
                    steps, max_cov = 0, 0.0
                    place()
                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    dragging = False
                elif ev.type == pygame.MOUSEMOTION and dragging and ev.pos[0] < VIEW:
                    block = list(to_sim(*ev.pos))
                    place()
                elif ev.type == pygame.MOUSEWHEEL:
                    shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                    angle_deg = (angle_deg + ev.y * (1 if shift else 5)) % 360
                    place()

        if running and waiting:          # tạm dừng sau lần suy nghĩ: chờ phím N
            redraw((120, 80, 0))
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
                status = "Model đang suy nghĩ..."
                redraw((120, 80, 0))
                queue = list(ctrl.next_actions())
                imag, think_no = ctrl.imagination, think_no + 1
                if ctrl.read is not None:                                   # WM encoder đọc s_t lệch bao nhiêu
                    rbx, rby, rang = block_now()
                    rx, ry, ra = ctrl.read
                    read_err = (float(np.hypot(rx - rbx, ry - rby)),
                                float(abs((np.degrees(ra - rang) + 180) % 360 - 180)))
                else:
                    read_err = None
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
                redraw((0, 140, 0))
                continue
            redraw()
            if not fast:
                clock.tick(10)  # đúng tốc độ mô phỏng: 10 bước/giây
        else:
            redraw((0, 140, 0) if status.startswith("THÀNH") else (200, 0, 0) if status.startswith("THẤT") else (30, 30, 30))
            clock.tick(30)


if __name__ == "__main__":
    main()
