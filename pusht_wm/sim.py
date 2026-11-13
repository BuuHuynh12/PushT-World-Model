"""Bước 2: môi trường mô phỏng gym-pusht và công cụ kiểm tra khớp với dataset.

Dataset lerobot/pusht chỉ lưu vị trí đầu đẩy (state 2D), không lưu pose của khối T.
Để replay một demo trong sim, ta ước lượng pose ban đầu (x, y, góc) của khối T
bằng cách render sim ở nhiều pose và so với frame đầu tiên của video
("render-and-compare"), rồi chạy lại đúng chuỗi action.
"""
from __future__ import annotations

import numpy as np

# Màu khối T khi render 96x96: thân (143,163,184) và viền LightSlateGray (119,136,153)
BLOCK_RGBS = np.array([[143, 163, 184], [119, 136, 153]], np.float32)


def make_env(obs_type: str = "pixels_agent_pos", max_episode_steps: int | None = None):
    """max_episode_steps: None = giữ mặc định của gym-pusht (300). App xem trực tiếp truyền số rất lớn
    để chạy không giới hạn (TimeLimit của gymnasium chỉ bị ghi đè khi truyền tham số này)."""
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (đăng ký env)

    kw = {} if max_episode_steps is None else {"max_episode_steps": max_episode_steps}
    return gym.make("gym_pusht/PushT-v0", obs_type=obs_type, render_mode="rgb_array",
                    observation_width=96, observation_height=96, **kw)


def _block_mask(img: np.ndarray, tol: float = 30.0) -> np.ndarray:
    x = img.astype(np.float32)[..., None, :]
    d = np.linalg.norm(x - BLOCK_RGBS, axis=-1).min(-1)
    return d < tol


def render_pose(env, agent_xy, block_pose) -> np.ndarray:
    """Đặt sim về trạng thái (agent_xy, block x, y, góc) và trả về ảnh 96x96."""
    state = [*agent_xy, *block_pose]
    obs, _ = env.reset(options={"reset_to_state": state})
    return obs["pixels"]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def estimate_block_pose(env, image: np.ndarray, agent_xy, n_angles: int = 72,
                        verbose: bool = False) -> tuple[np.ndarray, float]:
    """Ước lượng (x, y, góc) của khối T từ một ảnh 96x96 bằng render-and-compare.

    Mục tiêu: tối đa IoU giữa mask khối T của ảnh thật và ảnh render.
    1) Lưới thô: vị trí quanh tâm mask (bước 16) x n_angles góc.
    2) Tinh chỉnh cục bộ: giảm dần bước (x, y, góc) đến khi không cải thiện.
    Trả về (pose, iou).
    """
    target = _block_mask(image)
    if target.sum() < 10:
        raise ValueError("Không tìm thấy khối T trong ảnh (mask quá nhỏ)")

    # Tâm mask (pixel) -> tọa độ sim (0..512); xác định chiều trục bằng một lần render thử
    ys, xs = np.nonzero(target)
    cx, cy = xs.mean() * 512 / 96, ys.mean() * 512 / 96
    probe = _block_mask(render_pose(env, agent_xy, [256.0, 128.0, 0.0]))
    flip_y = np.nonzero(probe)[0].mean() * 512 / 96 > 256  # y sim tăng lên trên hay xuống dưới
    if flip_y:
        cy = 512 - cy

    def score(p):
        return _iou(target, _block_mask(render_pose(env, agent_xy, p)))

    best, best_s = None, -1.0
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    # tâm mask lệch tâm khối (CoM của chữ T) tới ~40 đơn vị -> lưới ±48
    offsets = (-48, -32, -16, 0, 16, 32, 48)
    for dx in offsets:
        for dy in offsets:
            for a in angles:
                p = np.array([cx + dx, cy + dy, a])
                s = score(p)
                if s > best_s:
                    best, best_s = p, s
    if verbose:
        print(f"  lưới thô: iou={best_s:.3f} pose={best}")

    steps = np.array([8.0, 8.0, np.deg2rad(5)])
    while steps[0] > 0.25:
        improved = False
        for k in range(3):
            for sign in (-1, 1):
                p = best.copy()
                p[k] += sign * steps[k]
                s = score(p)
                if s > best_s:
                    best, best_s, improved = p, s, True
        if not improved:
            steps /= 2
    if verbose:
        print(f"  tinh chỉnh: iou={best_s:.3f} pose={best}")
    return best, best_s


def replay_episode(env, images: np.ndarray, state: np.ndarray, action: np.ndarray,
                   init_block_pose) -> dict:
    """Replay chuỗi action từ pose ban đầu, so sánh với dataset từng bước.

    Quy ước LeRobot: action[t] được thực thi khi đang ở obs[t] -> sinh ra obs[t+1].
    """
    obs, _ = env.reset(options={"reset_to_state": [*state[0], *init_block_pose]})
    sim_imgs, agent_err, block_iou, coverage = [obs["pixels"]], [], [], []
    for t in range(len(action) - 1):
        obs, _, _, _, info = env.step(action[t])
        sim_imgs.append(obs["pixels"])
        agent_err.append(float(np.linalg.norm(obs["agent_pos"] - state[t + 1])))
        block_iou.append(_iou(_block_mask(obs["pixels"]), _block_mask(images[t + 1])))
        coverage.append(float(info["coverage"]))
    return {
        "sim_images": np.stack(sim_imgs),
        "agent_err": np.array(agent_err),   # sai số vị trí đầu đẩy (đơn vị sim, 0..512)
        "block_iou": np.array(block_iou),   # IoU mask khối T, sim vs video
        "coverage": np.array(coverage),
    }
