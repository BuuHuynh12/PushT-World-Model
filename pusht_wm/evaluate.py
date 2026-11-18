"""Đánh giá policy trong gym-pusht trên tập seed cố định.

Chạy song song N env (mỗi seed một env), gom obs của các env chưa kết thúc thành một
batch cho policy -> nhanh hơn chạy tuần tự nhiều lần.
Chỉ số: success rate (coverage > 0.95 trong <= 300 bước) và mean max-coverage.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from .sim import make_env

EVAL_SEEDS = list(range(100000, 100050))  # 50 seed cố định (giống repo gốc: test_start_seed=100000)


def _act_fn_from_policy(policy, num_inference_steps=None, seed: int = 0):
    """Bọc DiffusionPolicy thành hàm (images uint8 [B,n_obs,H,W,3], states [B,n_obs,2]) -> actions."""
    dev = next(policy.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)  # nhiễu khởi tạo tất định

    def act(images, states):
        img = torch.from_numpy(images).permute(0, 1, 4, 2, 3).contiguous()
        return policy.predict_action(img, states, num_inference_steps, generator=gen)
    return act


@torch.no_grad()
def evaluate(policy=None, act_fn=None, seeds=EVAL_SEEDS, max_steps: int = 300, n_obs: int = 2,
             num_inference_steps=None, n_video: int = 0, sample_seed: int = 0, init_states=None) -> dict:
    """policy: có .act(...) (ở chế độ eval), hoặc act_fn tùy ý.

    act_fn(images [B,n_obs,96,96,3] uint8, states [B,n_obs,2]) -> actions [B, k, 2] (đơn vị sim).
    init_states: None (vị trí ban đầu chuẩn của gym-pusht theo seed) hoặc list [agent x, y, khối x, y, góc] cho từng seed
    (vị trí đặt sẵn).
    """
    if act_fn is None:
        was_training = policy.training
        policy.eval()
        act_fn = _act_fn_from_policy(policy, num_inference_steps, sample_seed)
    n = len(seeds)
    # gym-pusht mặc định cắt ở 300 bước; muốn chạy lâu hơn phải nới TimeLimit của gymnasium
    envs = [make_env(max_episode_steps=max_steps if max_steps > 300 else None) for _ in range(n)]
    hist_img, hist_state = [], []
    prev = np.zeros((n, 2), np.float32)  # action vừa thực hiện (đầu episode: vị trí hiện tại = đứng yên)
    for k_, (env, s) in enumerate(zip(envs, seeds)):
        opt = None if init_states is None else {"reset_to_state": np.asarray(init_states[k_], np.float64)}
        obs, _ = env.reset(seed=int(s), options=opt)
        prev[k_] = obs["agent_pos"]
        hist_img.append([obs["pixels"]] * n_obs)        # đầu episode: lặp frame đầu
        hist_state.append([obs["agent_pos"]] * n_obs)
    done = np.zeros(n, bool)
    success = np.zeros(n, bool)
    max_cov = np.zeros(n)
    steps = np.zeros(n, int)
    frames = [[hist_img[i][-1]] for i in range(min(n_video, n))]

    while not done.all():
        active = np.flatnonzero(~done)
        imgs = np.stack([np.stack(hist_img[i][-n_obs:]) for i in active])
        sts = np.stack([np.stack(hist_state[i][-n_obs:]) for i in active]).astype(np.float32)
        # act_fn có thuộc tính wants_prev=True -> nhận thêm action vừa làm (policy WM -> Diffusion cần)
        # wants_ids=True -> nhận thêm id môi trường + chính môi trường (bước GỘP s_t cần nhớ theo từng màn;
        # env CHỈ dùng để GHI LOG trạng thái thật, policy không được đọc)
        kw = {"ids": active, "envs": [envs[i] for i in active]} if getattr(act_fn, "wants_ids", False) else {}
        actions = act_fn(imgs, sts, prev[active], **kw) if getattr(act_fn, "wants_prev", False) else act_fn(imgs, sts)
        for j, i in enumerate(active):
            for a in actions[j]:
                obs, _, term, _, info = envs[i].step(a.astype(np.float32))
                prev[i] = a
                steps[i] += 1
                hist_img[i].append(obs["pixels"])
                hist_state[i].append(obs["agent_pos"])
                max_cov[i] = max(max_cov[i], info["coverage"])
                if i < len(frames):
                    frames[i].append(obs["pixels"])
                if term or steps[i] >= max_steps:
                    success[i] = bool(info["is_success"])
                    done[i] = True
                    break
            hist_img[i] = hist_img[i][-n_obs:]
            hist_state[i] = hist_state[i][-n_obs:]
    for env in envs:
        env.close()
    if policy is not None and was_training:
        policy.train()
    return {
        "success_rate": float(success.mean()),
        "mean_max_coverage": float(max_cov.mean()),
        "per_seed": [{"seed": int(s), "success": bool(su), "max_coverage": float(c), "steps": int(st)}
                     for s, su, c, st in zip(seeds, success, max_cov, steps)],
        "videos": [np.stack(f) for f in frames],
    }


def save_video(frames: np.ndarray, path, fps: int = 10, scale: int = 3):
    import av

    frames = frames.repeat(scale, 1).repeat(scale, 2)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
