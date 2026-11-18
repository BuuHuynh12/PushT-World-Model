"""Mô hình: world model latent + Diffusion sinh PATH (không dùng tọa độ vật thể làm trạng thái):

    frame -> WM ảnh (đóng băng) -> s_t latent -> Diffusion -> PATH (quỹ đạo robot cần đi) -> toán -> action

- s_t: token latent của 2 frame gần nhất (DINOv2 + PCA) và/hoặc biểu diễn bên trong transformer của world model.
  KHÔNG có tọa độ khối T ở bất kỳ đâu trong đường chạy.
- PATH: 16 vị trí tương lai của chấm tròn (robot) p_{t+1..t+16}, chuẩn hóa p/256 - 1. Có sẵn trong mọi dataset
  (observation.state của lerobot/pusht) -> KHÔNG cần gán nhãn.
- Path -> action: THUẦN TOÁN. Vận tốc cần có v_k = (p_{k+1} - p_k) / 0.1 s; lệnh gửi simulator là điểm đích của bộ
  điều khiển PD, tính bằng cách giải ngược phương trình của bộ điều khiển (xem path_to_actions). Không có mô hình học.
- Chọn path: Diffusion đề xuất N path; world model tưởng tượng hậu quả của từng path trong latent rồi chọn path tốt nhất.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import NoiseScheduler
from .latent_data import norm_action
from .models import ConditionalUnet1D


def norm_pos(p):
    """Vị trí đơn vị sim [0, 512] -> [-1, 1]."""
    return p / 256.0 - 1.0


def unnorm_pos(p):
    return (p + 1.0) * 256.0


# ============================================================================
# Path -> action: THUẦN TOÁN (không có mô hình học)
# ============================================================================
# Robot (chấm tròn) của gym-pusht là vật thể động học, được lái bằng bộ điều khiển PD: mỗi bước mô phỏng dt = 0.01 s
#     v <- v + [k_p (a - p) - k_v v] dt ;   p <- p + v dt          (k_p = 100, k_v = 20; a = điểm đích = action)
# và mỗi action được giữ 10 bước mô phỏng (0.1 s). Đây là hệ TUYẾN TÍNH theo (p, v, a), nên sau một action:
#     p' = M00 p + M01 v + G0 a ;   v' = M10 p + M11 v + G1 a
# Muốn robot tới điểm kế tiếp của path (p' = p_next) thì giải ngược:  a = (p_next - M00 p - M01 v) / G0.
# Vận tốc v không quan sát được trực tiếp nhưng suy ra chính xác từ hai vị trí gần nhất và action vừa làm.
def pd_coeffs(k_p: float = 100.0, k_v: float = 20.0, dt: float = 0.01, n: int = 10):
    """-> (M [2, 2], G [2]) của một action (n bước mô phỏng)."""
    A = np.array([[1 - k_p * dt * dt, dt * (1 - k_v * dt)], [-k_p * dt, 1 - k_v * dt]])
    B = np.array([k_p * dt * dt, k_p * dt])
    M, G = np.eye(2), np.zeros(2)
    for _ in range(n):
        M, G = A @ M, A @ G + B
    return M, G


PD_M, PD_G = pd_coeffs()
CONTROL_DT = 0.1      # giây cho mỗi action (10 Hz)


def path_velocity(pos, path):
    """Vận tốc robot cần có trên từng đoạn của path (đơn vị sim / giây): v_k = (p_{k+1} - p_k) / 0.1 s.
    pos [M, 2] vị trí hiện tại, path [M, H, 2] -> [M, H, 2] (v_x, v_y)."""
    seq = torch.cat([pos[:, None], path], 1)
    return (seq[:, 1:] - seq[:, :-1]) / CONTROL_DT


def path_to_actions(pos, prev_act, path, lo: float = 0.0, hi: float = 512.0):
    """Đổi path thành action bằng toán (đơn vị sim).
    pos [M, 2, 2] vị trí robot quan sát ở (t-1, t); prev_act [M, 2] action đã làm ở t-1; path [M, H, 2] = p_{t+1..t+H}
    -> action [M, H, 2]: a_k là điểm đích gửi cho bộ điều khiển PD để robot đi từ p_k tới p_{k+1}."""
    (m00, m01), (m10, m11) = PD_M.tolist()
    g0, g1 = PD_G.tolist()
    p0, p = pos[:, 0], pos[:, 1]
    v0 = (p - m00 * p0 - g0 * prev_act) / m01             # vận tốc ở t-1 (suy từ p_{t-1}, p_t, a_{t-1})
    v = m10 * p0 + m11 * v0 + g1 * prev_act                # vận tốc ở t
    out = []
    for k in range(path.shape[1]):
        a = ((path[:, k] - m00 * p - m01 * v) / g0).clamp(lo, hi)   # action bị giới hạn trong bàn [0, 512]
        p, v = m00 * p + m01 * v + g0 * a, m10 * p + m11 * v + g1 * a   # robot thật sự tới đâu với action đã kẹp
        out.append(a)
    return torch.stack(out, 1)


# ============================================================================
# Diffusion đọc s_t latent -> path
# ============================================================================
@dataclass
class LatentPathConfig:
    feature: str = "both"            # latent | wm_hidden | both (ghép token latent với lớp ẩn của world model)
    P: int = 16                      # số token mỗi frame
    D_in: int = 384                  # chiều mỗi token của s_t
    n_obs: int = 2
    horizon: int = 16                # số điểm của path
    n_action_steps: int = 8          # số action thực thi trước khi nghĩ lại
    use_agent_pos: bool = True       # thêm vị trí chấm tròn (cảm biến của chính robot) vào điều kiện
    d_tok: int = 256                 # bộ đọc token (transformer nhỏ + token tóm tắt)
    n_layers: int = 3
    n_heads: int = 4
    feat_noise: float = 0.05         # nhiễu Gauss lên s_t khi train (thay cho random crop của ảnh)
    dropout: float = 0.1
    down_dims: list = field(default_factory=lambda: [256, 512, 1024])
    kernel_size: int = 5
    step_embed_dim: int = 128
    num_train_steps: int = 100
    num_inference_steps: int = 10


class TokenReader(nn.Module):
    """s_t [B, n_obs, P, D_in] -> vector điều kiện [B, d_tok]: transformer trên mọi token của n_obs frame,
    đọc ra bằng một token tóm tắt (attention), không ép phẳng qua MLP."""

    def __init__(self, cfg: LatentPathConfig):
        super().__init__()
        d = cfg.d_tok
        self.inp = nn.Sequential(nn.LayerNorm(cfg.D_in), nn.Linear(cfg.D_in, d))
        self.pos_s = nn.Parameter(torch.randn(cfg.P, d) * 0.02)
        self.pos_f = nn.Parameter(torch.randn(cfg.n_obs, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.noise = cfg.feat_noise

    def forward(self, f):
        if self.training and self.noise > 0:
            f = f + self.noise * torch.randn_like(f)
        B = f.shape[0]
        x = self.inp(f) + self.pos_s[None, None] + self.pos_f[None, :, None]
        x = torch.cat([self.cls.expand(B, -1, -1), x.flatten(1, 2)], 1)
        return self.out(self.enc(x)[:, 0])


class LatentPathPolicy(nn.Module):
    def __init__(self, cfg: LatentPathConfig):
        super().__init__()
        self.cfg = cfg
        self.reader = TokenReader(cfg)
        cond = cfg.d_tok + (cfg.n_obs * 2 if cfg.use_agent_pos else 0)
        self.unet = ConditionalUnet1D(2, cond, cfg.step_embed_dim, cfg.down_dims, cfg.kernel_size)
        self.scheduler = NoiseScheduler(cfg.num_train_steps)
        self.scorer = None           # WMScorer (encoder + world model), gắn khi chạy online
        self.imagination = None      # "suy nghĩ" của lần ra quyết định gần nhất (app vẽ)

    # ------------------------------------------------------------- s_t từ world model
    @staticmethod
    def features(wm, feature: str, z, prev_act_raw):
        """z [B, n_obs, P, D] latent đã chuẩn hóa; prev_act_raw [B, 2] action đã làm ở t-1 (đơn vị sim)
        -> s_t [B, n_obs, P, D_in]. Không dùng action tương lai (không rò rỉ đáp án)."""
        if feature == "latent":
            return z
        a = norm_action(prev_act_raw.float())[:, None].expand(-1, z.shape[1], -1)   # frame t dùng tạm action t-1
        with torch.no_grad():
            h = wm.hidden(z, a).float()
        return h if feature == "wm_hidden" else torch.cat([z, h], -1)

    @staticmethod
    def feature_dim(feature: str, D: int, d_model: int) -> int:
        return {"latent": D, "wm_hidden": d_model, "both": D + d_model}[feature]

    def _cond(self, feat, pos_n):
        c = self.reader(feat)
        return torch.cat([c, pos_n.flatten(1)], -1) if self.cfg.use_agent_pos else c

    # ----------------------------------------------------------------- train
    def compute_loss(self, feat, pos_n, path_n):
        """feat [B, n_obs, P, D_in]; pos_n [B, n_obs, 2] vị trí chấm tròn (chuẩn hóa); path_n [B, H, 2]. Loss DDPM chuẩn."""
        cond = self._cond(feat, pos_n)
        noise = torch.randn_like(path_n)
        t = torch.randint(0, self.scheduler.T, (path_n.shape[0],), device=path_n.device)
        eps = self.unet(self.scheduler.add_noise(path_n, noise, t), t, cond)
        return F.mse_loss(eps.float(), noise)

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def sample_paths(self, feat, pos_n, n_samples: int = 1, generator=None, num_inference_steps=None):
        """-> path chuẩn hóa [B*n_samples, H, 2]."""
        cond = self._cond(feat, pos_n)
        if n_samples > 1:
            cond = cond.repeat_interleave(n_samples, 0)
        k = num_inference_steps or self.cfg.num_inference_steps
        return self.scheduler.sample(lambda x, t: self.unet(x, t, cond), (cond.shape[0], self.cfg.horizon, 2),
                                     cond.device, k, generator)

    def attach_scorer(self, scorer):
        object.__setattr__(self, "scorer", scorer)
        return self

    @torch.no_grad()
    def act_from_images(self, frames, pos_raw, prev_act_raw, n_samples: int = 1, generator=None):
        """frames uint8 [B, 2, H, W, 3]; pos_raw [B, 2, 2] vị trí chấm tròn; prev_act_raw [B, 2] (đơn vị sim)
        -> action [B, n_action_steps, 2] (đơn vị sim). n_samples > 1: world model tưởng tượng và chọn path."""
        sc, cfg = self.scorer, self.cfg
        dev = sc.dev
        B, N = len(frames), n_samples
        z = sc.encode(frames)                                                        # s_t (latent) [B, 2, P, D]
        prev = torch.from_numpy(np.asarray(prev_act_raw, np.float32)).to(dev)
        feat = self.features(sc.model, cfg.feature, z, prev)
        pos_n = norm_pos(torch.from_numpy(np.asarray(pos_raw, np.float32)).to(dev))
        path = unnorm_pos(self.sample_paths(feat, pos_n, n_samples=N, generator=generator))   # [B*N, H, 2] đơn vị sim
        pos = unnorm_pos(pos_n).repeat_interleave(N, 0)
        act = path_to_actions(pos, prev.repeat_interleave(N, 0), path)               # path -> action (thuần toán)
        a_sim = act.view(B, N, cfg.horizon, 2).cpu().numpy()
        pick = np.zeros(B, int)
        im = {"robot_path": path.view(B, N, cfg.horizon, 2).cpu().numpy(), "paths": a_sim,
              "velocity": path_velocity(pos[:, -1], path).view(B, N, cfg.horizon, 2).cpu().numpy(),
              "act": cfg.n_action_steps, "pose": None, "cov": None, "score": None}
        if N > 1:
            pose, cov, score = sc.imagine(None, np.asarray(prev_act_raw, np.float32), a_sim, z=z)
            pick = score.argmax(1)
            im.update(pose=pose, cov=cov, score=score)
        im["pick"] = pick
        self.imagination = im
        return a_sim[np.arange(B), pick, :cfg.n_action_steps]

    # ------------------------------------------------------------- lưu / tải
    def checkpoint(self, extra=None):
        return {"latent_path_policy": True, "config": asdict(self.cfg), "state_dict": self.state_dict(), **(extra or {})}

    @classmethod
    def from_checkpoint(cls, ckpt, wm_ckpt=None, device=None):
        """wm_ckpt: None -> lấy đường dẫn world model ghi trong checkpoint. device: None -> không gắn world model."""
        if not isinstance(ckpt, dict):
            ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
        p = cls(LatentPathConfig(**ckpt["config"]))
        p.load_state_dict(ckpt["state_dict"])
        if device is not None:
            from .planner import WMScorer
            p = p.to(device).eval()
            p.attach_scorer(WMScorer(wm_ckpt or ckpt["wm_ckpt"], device=str(device)))
        return p


# ============================================================================
# Chạy trong simulator: đánh giá + thu dữ liệu tự chơi (cho vòng lặp train lại world model)
# ============================================================================
@torch.no_grad()
def rollout(policy: LatentPathPolicy, seeds, n_samples: int = 16, max_steps: int = 300, batch_envs: int = 100,
            sample_seed: int = 0, record: bool = False, log=print):
    """Chạy mỗi seed một màn từ ảnh. -> (per_seed [dict], RawPushT hoặc None).
    record=True: ghi lại ảnh, vị trí chấm tròn, action, coverage, và trạng thái thật của simulator (chỉ để tính
    loss FDE / đánh giá world model; policy KHÔNG được đọc)."""
    from .collect import get_full_state
    from .sim import make_env

    gen = torch.Generator(device=policy.scorer.dev).manual_seed(sample_seed)
    per_seed, recs = [], []
    for b0 in range(0, len(seeds), batch_envs):
        bs = seeds[b0:b0 + batch_envs]
        n = len(bs)
        envs = [make_env(max_episode_steps=max_steps if max_steps > 300 else None) for _ in bs]
        img, pos, prev = [], [], np.zeros((n, 2), np.float32)
        rec = [dict(img=[], pos=[], full=[], act=[], rew=[], suc=[]) for _ in bs]
        for i, (env, s) in enumerate(zip(envs, bs)):
            obs, _ = env.reset(seed=int(s))
            img.append([obs["pixels"]] * 2), pos.append([obs["agent_pos"]] * 2)
            prev[i] = obs["agent_pos"]
        done, success = np.zeros(n, bool), np.zeros(n, bool)
        max_cov, steps = np.zeros(n), np.zeros(n, int)
        while not done.all():
            act = np.flatnonzero(~done)
            acts = policy.act_from_images(np.stack([np.stack(img[i]) for i in act]),
                                          np.stack([np.stack(pos[i]) for i in act]).astype(np.float32), prev[act],
                                          n_samples=n_samples, generator=gen)
            for j, i in enumerate(act):
                R = rec[i]
                for a in acts[j]:
                    if record:
                        R["img"].append(img[i][-1]), R["pos"].append(pos[i][-1]), R["full"].append(get_full_state(envs[i]))
                    obs, _, term, _, info = envs[i].step(a.astype(np.float32))
                    if record:
                        R["act"].append(a), R["rew"].append(float(info["coverage"])), R["suc"].append(bool(info["is_success"]))
                    prev[i] = a
                    steps[i] += 1
                    img[i].append(obs["pixels"]), pos[i].append(obs["agent_pos"])
                    max_cov[i] = max(max_cov[i], info["coverage"])
                    if term or steps[i] >= max_steps:
                        success[i], done[i] = bool(info["is_success"]), True
                        break
                img[i], pos[i] = img[i][-2:], pos[i][-2:]
        for e in envs:
            e.close()
        per_seed += [{"seed": int(s), "success": bool(su), "max_coverage": float(c), "steps": int(st)}
                     for s, su, c, st in zip(bs, success, max_cov, steps)]
        recs += rec
        log(f"  {len(per_seed)}/{len(seeds)} màn | success tới giờ {np.mean([p['success'] for p in per_seed]):.3f}")
    raw = None
    if record:
        from .data import RawPushT
        lens = np.array([len(R["act"]) for R in recs])
        cat = lambda k, dt: np.concatenate([np.asarray(R[k], dt) for R in recs])  # noqa: E731
        raw = RawPushT(images=cat("img", np.uint8), state=cat("pos", np.float32), action=cat("act", np.float32),
                       reward=cat("rew", np.float32), success=cat("suc", bool),
                       episode_index=np.repeat(np.arange(len(recs)), lens).astype(np.int64),
                       episode_ends=np.cumsum(lens).astype(np.int64), fps=10, state_full=cat("full", np.float64),
                       source=np.full(lens.sum(), 4, np.int8), episode_seed=np.asarray(seeds, np.int64))
    return per_seed, raw


def summarize(per_seed) -> dict:
    succ = np.array([x["success"] for x in per_seed])
    st = [x["steps"] for x in per_seed if x["success"]]
    return {"success_rate": float(succ.mean()), "ci95": float(1.96 * np.sqrt(succ.mean() * (1 - succ.mean()) / len(succ))),
            "steps_to_success_median": float(np.median(st)) if st else None,
            "mean_max_coverage": float(np.mean([x["max_coverage"] for x in per_seed]))}
