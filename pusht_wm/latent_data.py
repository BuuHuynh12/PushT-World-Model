"""dữ liệu chuỗi latent cho world model.

Ghép latent đã cache (data/latents/<encoder>/<dataset>.npy) với action/reward của từng frame.
Chỉ đọc các mảng nhỏ trong file .npz (không nạp ảnh) -> nhẹ RAM.
Latent được đưa toàn bộ lên GPU nếu đủ chỗ (DINOv2 ~2.8 GB fp16) để lấy batch không tốn CPU.

Quy ước: z[t] = latent của obs t; a[t] = action thực thi tại t; r[t] = coverage SAU action t
(= coverage của obs t+1). World model học: (z[<=t], a[<=t]) -> z[t+1], r[t].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

def data_file(name: str) -> str:
    """Tên tập -> file .npz (demo = dataset lerobot/pusht đã giải mã; còn lại là dữ liệu thu trong simulator)."""
    return "data/pusht_raw.npz" if name == "demo" else f"data/{name}.npz"


def pose_vec6(state_full: np.ndarray) -> np.ndarray:
    """state_full [N, 5] (chấm tròn x, y, khối T x, y, góc rad) -> [N, 6] = (x, y, x, y) / 512, sin, cos."""
    s = state_full
    return np.concatenate([s[:, :4] / 512.0, np.sin(s[:, 4:5]), np.cos(s[:, 4:5])], 1).astype(np.float32)


def norm_action(a):
    """Action đơn vị sim [0, 512] -> [-1, 1] (cố định, không phụ thuộc tập dữ liệu)."""
    return a / 256.0 - 1.0


@dataclass
class LatentStats:
    mean: torch.Tensor  # [D] trung bình theo chiều, gộp mọi token
    std: torch.Tensor   # [D]

    def state_dict(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}


def pool_tokens(z, pool: int):
    """Avg-pool lưới token g x g -> pool x pool (z [N, g*g, D], numpy hoặc torch).
    Đã đo (probe MLP): DINOv2 8x8 -> 4x4 gần như không mất thông tin pose/coverage,
    còn giảm 4 lần số token -> world model nhanh ~4 lần."""
    g = int(round(z.shape[1] ** 0.5))
    if not pool or pool >= g:
        return z
    is_np = isinstance(z, np.ndarray)
    t = torch.from_numpy(z) if is_np else z
    out = []
    for i in range(0, len(t), 8192):  # theo khối để tiết kiệm RAM
        x = t[i:i + 8192].float().view(-1, g, g, t.shape[-1]).permute(0, 3, 1, 2)
        out.append(torch.nn.functional.adaptive_avg_pool2d(x, pool).flatten(2).transpose(1, 2).to(t.dtype))
    out = torch.cat(out)
    return out.numpy() if is_np else out


class LatentSeqData:
    def __init__(self, encoder: str, datasets=("demo", "sim_v1"), latent_root="data/latents",
                 device="cuda", on_gpu: bool | None = None, pool: int = 0):
        z_list, a_list, r_list, ends, src = [], [], [], [], []
        pose_list, has_list, pos_list = [], [], []
        off = 0
        for name in datasets:
            z = np.load(Path(latent_root) / encoder / f"{name}.npy", mmap_mode="r")
            d = np.load(data_file(name))
            ep_end = d["episode_ends"]
            assert len(z) == ep_end[-1] == len(d["action"]), f"{name}: số frame latent khác dữ liệu"
            z_list.append(pool_tokens(np.asarray(z), pool))
            a_list.append(norm_action(d["action"].astype(np.float32)))
            r_list.append(d["reward"].astype(np.float32))
            pos_list.append(d["state"].astype(np.float32))            # vị trí chấm tròn (có trong mọi tập)
            # tư thế khối T: simulator ghi sẵn (state_full); demo lerobot/pusht KHÔNG có -> has_pose = False
            has = "state_full" in d.files
            pose_list.append(pose_vec6(d["state_full"]) if has else np.zeros((len(z), 6), np.float32))
            has_list.append(np.full(len(z), has))
            ends.append(ep_end + off)
            src.append(np.full(len(ep_end), name))
            off += len(z)
        z = np.concatenate(z_list)
        self.P, self.D = z.shape[1:]
        self.episode_ends = np.concatenate(ends)
        self.episode_starts = np.r_[0, self.episode_ends[:-1]]
        self.episode_source = np.concatenate(src)
        self.n_episodes = len(self.episode_ends)
        self.device = torch.device(device)
        need = z.nbytes / 2**30
        free = torch.cuda.mem_get_info()[0] / 2**30 if self.device.type == "cuda" else 0
        # chỉ đưa latent lên GPU nếu sau đó còn >= 5 GB cho train (đã gặp OOM khi để 2.8 GB trên card 8 GB)
        self.on_gpu = (free - need >= 5.0) if on_gpu is None else on_gpu
        zt = torch.from_numpy(z)
        self.z = zt.to(self.device) if self.on_gpu else zt.pin_memory()
        self.a = torch.from_numpy(np.concatenate(a_list)).to(self.device)
        self.r = torch.from_numpy(np.concatenate(r_list)).to(self.device)
        self.pose = torch.from_numpy(np.concatenate(pose_list)).to(self.device)        # [N, 6]
        self.has_pose = torch.from_numpy(np.concatenate(has_list)).to(self.device)     # [N] bool
        self.agent_pos = torch.from_numpy(np.concatenate(pos_list)).to(self.device)    # [N, 2] đơn vị sim
        self.n_frames = len(z)

    def fit_stats(self, episodes) -> LatentStats:
        """mean/std mỗi chiều latent, tính trên frame của các episode train (lấy mẫu tối đa 20k frame)."""
        idx = np.concatenate([np.arange(self.episode_starts[e], self.episode_ends[e]) for e in episodes])
        idx = np.random.default_rng(0).choice(idx, min(20000, len(idx)), replace=False)
        x = self.z[torch.from_numpy(np.sort(idx)).to(self.z.device)].float().reshape(-1, self.D)
        return LatentStats(x.mean(0).to(self.device), x.std(0).clamp_min(1e-4).to(self.device))

    def window_starts(self, episodes, length: int) -> np.ndarray:
        """Mọi chỉ số bắt đầu s sao cho [s, s+length) nằm trọn trong 1 episode."""
        out = [np.arange(self.episode_starts[e], self.episode_ends[e] - length + 1) for e in episodes]
        return np.concatenate([o for o in out if len(o)]).astype(np.int64)

    def get(self, starts: np.ndarray, length: int, stats: LatentStats):
        """-> z [B, L, P, D] (đã chuẩn hóa, float32), a [B, L, 2], r [B, L]."""
        idx = torch.from_numpy(starts[:, None] + np.arange(length)[None]).to(self.z.device)
        z = self.z[idx].to(self.device, non_blocking=True).float()
        z = (z - stats.mean) / stats.std
        idx = idx.to(self.device)
        return z, self.a[idx], self.r[idx]

    def get_pose(self, starts: np.ndarray, length: int):
        """-> pose [B, L, 6] tư thế thật, has_pose [B, L] (False: tập không có tư thế khối T)."""
        idx = torch.from_numpy(starts[:, None] + np.arange(length)[None]).to(self.device)
        return self.pose[idx], self.has_pose[idx]
