"""Bộ điều khiển dùng cho app (WM latent: không dùng tọa độ làm trạng thái).

    frame (2 ảnh) -> WM (DINOv2 + transformer) -> s_t latent -> Diffusion sinh 16 PATH robot
    -> path -> action bằng toán (giải ngược bộ PD) -> WM tưởng tượng hậu quả từng path -> chọn -> 8 action

Giao diện: reset(obs) / observe(obs, action) / next_actions() -> 8 action; imagination = "suy nghĩ" để app vẽ.
"""
from __future__ import annotations

import numpy as np
import torch

LATENT_POLICY = "models/diffusion_path.pt"   # 92.0% (500 seed 600000..); world model đi kèm: models/wm.pt
N_PATHS = 16
DESC = "frame → WM → s_t latent → Diffusion 16 PATH → WM chọn → action (toán)"


class LatentController:
    def __init__(self, ckpt: str = LATENT_POLICY, device=None, seed: int = 0):
        from .latent_path import LatentPathPolicy

        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.p = LatentPathPolicy.from_checkpoint(ckpt, device=self.dev)
        self.read = None              # mô hình này không đọc tọa độ khối T
        self.imagination = None
        self.consistency = False      # (không dùng; giữ cho app)

    def reset(self, obs):
        self.hist_img = [obs["pixels"]] * 2
        self.hist_pos = [np.asarray(obs["agent_pos"], np.float32)] * 2
        self.prev_act = np.asarray(obs["agent_pos"], np.float32)

    def observe(self, obs, action):
        self.hist_img = [self.hist_img[-1], obs["pixels"]]
        self.hist_pos = [self.hist_pos[-1], np.asarray(obs["agent_pos"], np.float32)]
        self.prev_act = np.asarray(action, np.float32)

    @torch.no_grad()
    def next_actions(self) -> np.ndarray:
        a = self.p.act_from_images(np.stack(self.hist_img)[None], np.stack(self.hist_pos)[None],
                                   self.prev_act[None], n_samples=N_PATHS, generator=self.gen)[0]
        im = self.p.imagination
        self.imagination = {"paths": im["robot_path"][0],          # vẽ PATH robot (quỹ đạo Diffusion sinh)
                            "pick": int(im["pick"][0]), "act": im["act"],
                            "pose": im["pose"][0], "cov": im["cov"][0], "score": im["score"][0], "plan": None}
        return a
