"""Bước 4: thu thập thêm dữ liệu trong sim cho world model.

Mỗi episode được gán 1 chế độ (theo seed, tái lập được):
  clean   : policy baseline nguyên bản (nhiễu sample khác nhau giữa các env)
  noise   : policy + nhiễu Gaussian lên action (độ lệch theo chunk + rung từng bước), sigma ngẫu nhiên
  mixed   : mỗi chunk có xác suất p bị thay bằng chunk ngẫu nhiên
  random  : toàn bộ chunk ngẫu nhiên (không dùng policy)
Lưu cả episode thất bại; mỗi bước lưu ảnh, agent_pos, state 5D đầy đủ, action thực thi,
reward (= coverage, cùng quy ước với dataset gốc), success và nguồn action.

Quy ước thời gian giống LeRobot: action[t] thực thi khi đang ở obs[t]; reward[t], success[t]
là kết quả SAU khi thực thi action[t].
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from dataclasses import dataclass

import numpy as np
import torch

from .data import RawPushT
from .sim import make_env

MODES = ("clean", "noise", "mixed", "random")
SRC = {"policy": 1, "noise": 2, "random": 3}


@dataclass
class CollectConfig:
    # nghiêng về gần phân phối policy (bước 8 cần xếp hạng chunk của chính policy)
    mode_probs: tuple = (0.3, 0.4, 0.2, 0.1)   # xác suất các chế độ clean/noise/mixed/random
    noise_sigma: tuple = (3.0, 20.0)          # sigma (đơn vị sim) lấy đều trong khoảng
    mixed_p: tuple = (0.2, 0.5)               # xác suất thay chunk bằng chunk ngẫu nhiên
    random_local_std: float = 80.0            # chunk ngẫu nhiên: đích quanh agent ...
    random_global_p: float = 0.3              # ... hoặc đích đều trên toàn bàn với xác suất này
    max_steps: int = 300
    n_action_steps: int = 8


def get_full_state(env) -> np.ndarray:
    """(agent x, y, block x, y, góc) — đúng thứ tự của reset_to_state."""
    u = env.unwrapped
    return np.array([*u.agent.position, *u.block.position, u.block.angle], np.float64)


def episode_plan(seed: int, cfg: CollectConfig) -> dict:
    """Tham số của 1 episode, chỉ phụ thuộc seed."""
    r = np.random.default_rng(seed)
    mode = MODES[r.choice(4, p=cfg.mode_probs)]
    return {"mode": mode, "sigma": float(r.uniform(*cfg.noise_sigma)),
            "p_rand": float(r.uniform(*cfg.mixed_p)), "rng": r}


def random_chunk(agent_xy, k, rng, cfg: CollectConfig) -> np.ndarray:
    """Chunk ngẫu nhiên mượt: đi thẳng từ vị trí hiện tại tới một đích ngẫu nhiên trong k bước."""
    if rng.random() < cfg.random_global_p:
        goal = rng.uniform(20, 492, 2)
    else:
        goal = agent_xy + rng.normal(0, cfg.random_local_std, 2)
    w = np.arange(1, k + 1)[:, None] / k
    path = agent_xy[None] * (1 - w) + goal[None] * w + rng.normal(0, 3, (k, 2))
    return np.clip(path, 0, 512)


@torch.no_grad()
def collect(policy, seeds, cfg: CollectConfig = CollectConfig(), batch_envs: int = 50,
            sample_seed: int = 0, log=print) -> tuple[RawPushT, list[dict]]:
    """Chạy các episode theo lô batch_envs env song song. Trả về (RawPushT, thông tin từng episode)."""
    dev = next(policy.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(sample_seed)
    n_obs, k = policy.cfg.n_obs, cfg.n_action_steps
    episodes, infos = [], []
    for b0 in range(0, len(seeds), batch_envs):
        bseeds = seeds[b0:b0 + batch_envs]
        envs = [make_env() for _ in bseeds]
        plans = [episode_plan(int(s), cfg) for s in bseeds]
        rec = [dict(img=[], pos=[], full=[], act=[], rew=[], suc=[], src=[]) for _ in bseeds]
        hist_img, hist_pos, cur = [], [], []
        for env, s in zip(envs, bseeds):
            obs, _ = env.reset(seed=int(s))
            hist_img.append([obs["pixels"]] * n_obs)
            hist_pos.append([obs["agent_pos"]] * n_obs)
            cur.append(obs)
        done = np.zeros(len(bseeds), bool)
        while not done.all():
            active = np.flatnonzero(~done)
            need_pol = [i for i in active if plans[i]["mode"] != "random"]
            pol_act = {}
            if need_pol:
                imgs = np.stack([np.stack(hist_img[i][-n_obs:]) for i in need_pol])
                sts = np.stack([np.stack(hist_pos[i][-n_obs:]) for i in need_pol]).astype(np.float32)
                img_t = torch.from_numpy(imgs).permute(0, 1, 4, 2, 3).contiguous()
                a = policy.predict_action(img_t, sts, generator=gen)
                pol_act = {i: a[j] for j, i in enumerate(need_pol)}
            for i in active:
                p, r = plans[i], plans[i]["rng"]
                pos = np.asarray(cur[i]["agent_pos"], np.float64)
                if p["mode"] == "random" or (p["mode"] == "mixed" and r.random() < p["p_rand"]):
                    chunk, src = random_chunk(pos, k, r, cfg), SRC["random"]
                elif p["mode"] == "noise":
                    off = r.normal(0, p["sigma"], 2)                    # lệch chung cả chunk
                    chunk = pol_act[i] + off + r.normal(0, p["sigma"] / 3, (k, 2))
                    src = SRC["noise"]
                else:
                    chunk, src = pol_act[i], SRC["policy"]
                chunk = np.clip(chunk, 0, 512).astype(np.float32)
                for a in chunk:
                    R = rec[i]
                    R["img"].append(cur[i]["pixels"])
                    R["pos"].append(cur[i]["agent_pos"])
                    R["full"].append(get_full_state(envs[i]))
                    R["act"].append(a)
                    R["src"].append(src)
                    obs, _, term, _, info = envs[i].step(a)
                    R["rew"].append(info["coverage"])
                    R["suc"].append(bool(info["is_success"]))
                    cur[i] = obs
                    hist_img[i].append(obs["pixels"])
                    hist_pos[i].append(obs["agent_pos"])
                    if term or len(R["act"]) >= cfg.max_steps:
                        done[i] = True
                        break
                hist_img[i] = hist_img[i][-n_obs:]
                hist_pos[i] = hist_pos[i][-n_obs:]
        for env in envs:
            env.close()
        for s, p, R in zip(bseeds, plans, rec):
            episodes.append(R)
            infos.append({"seed": int(s), "mode": p["mode"], "sigma": p["sigma"], "p_rand": p["p_rand"],
                          "len": len(R["act"]), "success": bool(any(R["suc"])),
                          "max_coverage": float(max(R["rew"]))})
        done_n = len(infos)
        sr = np.mean([x["success"] for x in infos])
        log(f"  {done_n}/{len(seeds)} episode | success tích lũy {sr:.2f} | "
            f"frame {sum(x['len'] for x in infos)}")
    return _to_raw(episodes, infos), infos


def _to_raw(episodes, infos) -> RawPushT:
    lens = np.array([len(R["act"]) for R in episodes])
    cat = lambda k, dt: np.concatenate([np.asarray(R[k], dt) for R in episodes])  # noqa: E731
    return RawPushT(
        images=np.stack([im for R in episodes for im in R["img"]]).astype(np.uint8),
        state=cat("pos", np.float32), action=cat("act", np.float32),
        reward=cat("rew", np.float32), success=cat("suc", bool),
        episode_index=np.repeat(np.arange(len(episodes)), lens).astype(np.int64),
        episode_ends=np.cumsum(lens).astype(np.int64), fps=10,
        state_full=cat("full", np.float64), source=cat("src", np.int8),
        episode_seed=np.array([x["seed"] for x in infos], np.int64))
