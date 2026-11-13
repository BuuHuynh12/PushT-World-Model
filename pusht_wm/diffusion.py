"""Scheduler DDPM (train) và DDIM (suy luận nhanh), dự đoán nhiễu epsilon.

Cài đặt theo đúng cấu hình Diffusion Policy gốc:
  beta_schedule = squaredcos_cap_v2, 100 bước, clip_sample=True, prediction_type=epsilon,
  DDIM set_alpha_to_one=True, eta=0 (tất định).
"""
from __future__ import annotations

import math

import torch


def cosine_betas(T: int, max_beta: float = 0.999) -> torch.Tensor:
    """squaredcos_cap_v2 (Nichol & Dhariwal 2021)."""
    f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
    return torch.tensor([min(1 - f((i + 1) / T) / f(i / T), max_beta) for i in range(T)],
                        dtype=torch.float64)


class NoiseScheduler:
    def __init__(self, num_train_steps: int = 100, clip_sample: bool = True):
        self.T = num_train_steps
        self.clip_sample = clip_sample
        betas = cosine_betas(num_train_steps)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)

    # ------------------------------------------------------------------ train
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """q(x_t | x_0) = sqrt(a_t) x0 + sqrt(1-a_t) eps."""
        a = self.alphas_cumprod.to(x0.device)[t].float().view(-1, *([1] * (x0.dim() - 1)))
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    # ------------------------------------------------------------ inference
    def timesteps(self, num_inference_steps: int) -> list[int]:
        """Giống diffusers: [T-ratio, ..., ratio, 0] (DDIM) / [T-1..0] (DDPM)."""
        ratio = self.T // num_inference_steps
        return [i * ratio for i in reversed(range(num_inference_steps))]

    def _pred_x0(self, x, eps, a_t):
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        return x0.clamp(-1, 1) if self.clip_sample else x0

    def ddim_step(self, eps, t: int, x, num_inference_steps: int):
        ac = self.alphas_cumprod
        prev = t - self.T // num_inference_steps
        a_t = torch.as_tensor(float(ac[t]), device=x.device)
        a_prev = torch.as_tensor(float(ac[prev]) if prev >= 0 else 1.0, device=x.device)
        x0 = self._pred_x0(x, eps, a_t)
        return a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps

    def ddpm_step(self, eps, t: int, x, generator=None):
        """Bước DDPM đầy đủ (dùng khi num_inference_steps == T)."""
        ac = self.alphas_cumprod
        a_t = float(ac[t])
        a_prev = float(ac[t - 1]) if t > 0 else 1.0
        beta_t = 1 - a_t / a_prev
        x0 = self._pred_x0(x, eps, torch.as_tensor(a_t, device=x.device))
        # trung bình hậu nghiệm q(x_{t-1} | x_t, x0)
        c0 = math.sqrt(a_prev) * beta_t / (1 - a_t)
        ct = math.sqrt(1 - beta_t) * (1 - a_prev) / (1 - a_t)
        mean = c0 * x0 + ct * x
        if t > 0:
            var = (1 - a_prev) / (1 - a_t) * beta_t
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            mean = mean + math.sqrt(max(var, 1e-20)) * noise
        return mean

    @torch.no_grad()
    def sample(self, eps_fn, shape, device, num_inference_steps: int = 10,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Sinh mẫu từ nhiễu. eps_fn(x, t_tensor) -> eps. DDIM nếu số bước < T, ngược lại DDPM."""
        x = torch.randn(shape, generator=generator, device=device)
        use_ddpm = num_inference_steps >= self.T
        for t in self.timesteps(self.T if use_ddpm else num_inference_steps):
            tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = eps_fn(x, tt).float()
            x = self.ddpm_step(eps, t, x, generator) if use_ddpm else \
                self.ddim_step(eps, t, x, num_inference_steps)
        return x
