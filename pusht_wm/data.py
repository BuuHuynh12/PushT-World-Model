"""Bước 1: load dataset lerobot/pusht (parquet + video) thành mảng numpy và Dataset PyTorch.

Không phụ thuộc thư viện `lerobot` (tránh lệch phiên bản); chỉ cần
huggingface_hub, pandas, pyarrow, av (PyAV, có decoder AV1), numpy, torch.

Hỗ trợ cả layout v2.x (mỗi episode 1 file) và v3.x (nhiều episode gộp 1 file):
đọc toàn bộ parquet, sắp theo `index`, decode toàn bộ video theo thứ tự file
và kiểm tra tổng số frame khớp số dòng.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_KEY = "observation.image"


# ----------------------------------------------------------------------------
# Tải và đọc dữ liệu thô
# ----------------------------------------------------------------------------
def download(repo_id: str = "lerobot/pusht", local_dir: str | None = None,
             revision: str | None = None) -> Path:
    """Tải snapshot dataset từ Hugging Face Hub, trả về thư mục gốc."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo_id, repo_type="dataset",
                             local_dir=local_dir, revision=revision)
    return Path(path)


def _decode_video(path: Path) -> np.ndarray:
    import av

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)


@dataclass
class RawPushT:
    images: np.ndarray        # [N, 96, 96, 3] uint8
    state: np.ndarray         # [N, 2] float32  (vị trí đầu đẩy)
    action: np.ndarray        # [N, 2] float32  (vị trí đích của đầu đẩy)
    reward: np.ndarray        # [N]   float32
    success: np.ndarray       # [N]   bool
    episode_index: np.ndarray # [N]   int64
    episode_ends: np.ndarray  # [E]   chỉ số kết thúc (exclusive) của từng episode
    fps: int
    # --- Trường tùy chọn: chỉ có ở dữ liệu thu trong sim (bước 4); dataset gốc = None
    state_full: np.ndarray | None = None  # [N, 5] float64 (agent x, y, block x, y, góc) TRƯỚC khi thực thi action[t]
    source: np.ndarray | None = None      # [N] int8, xem SOURCE_NAMES: action[t] đến từ đâu
    episode_seed: np.ndarray | None = None  # [E] seed reset của env (tái lập trạng thái đầu)

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ends)

    def episode_slice(self, ep: int) -> slice:
        start = 0 if ep == 0 else int(self.episode_ends[ep - 1])
        return slice(start, int(self.episode_ends[ep]))

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__
                                     if getattr(self, k) is not None})

    @classmethod
    def load(cls, path: str | Path) -> "RawPushT":
        d = np.load(path)
        return cls(**{k: (int(d[k]) if k == "fps" else d[k]) for k in cls.__dataclass_fields__ if k in d})


# Nguồn của action trong dữ liệu sim (trường `source`)
SOURCE_NAMES = {0: "demo", 1: "policy", 2: "policy+noise", 3: "random", 4: "wm_plan", 5: "branch"}


def concat_raw(parts: list[RawPushT]) -> RawPushT:
    """Ghép nhiều tập (vd demo + sim). Trường tùy chọn thiếu ở một tập sẽ bị bỏ;
    riêng `source` của tập không có được điền 0 (demo)."""
    ends, off, eps, ep_off = [], 0, [], 0
    for p in parts:
        ends.append(p.episode_ends + off)
        eps.append(p.episode_index - p.episode_index.min() + ep_off)
        off += len(p.images)
        ep_off += p.n_episodes
    cat = lambda k: np.concatenate([getattr(p, k) for p in parts])  # noqa: E731
    opt = lambda k: cat(k) if all(getattr(p, k) is not None for p in parts) else None  # noqa: E731
    return RawPushT(
        images=cat("images"), state=cat("state"), action=cat("action"), reward=cat("reward"),
        success=cat("success"), episode_index=np.concatenate(eps), episode_ends=np.concatenate(ends),
        fps=parts[0].fps, state_full=opt("state_full"),
        source=np.concatenate([p.source if p.source is not None else np.zeros(len(p.images), np.int8)
                               for p in parts]),
        episode_seed=opt("episode_seed"))


def load_raw(root: str | Path) -> RawPushT:
    """Đọc thư mục dataset kiểu LeRobot (v2.x hoặc v3.x)."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    print(f"[data] codebase_version={info.get('codebase_version')} "
          f"episodes={info.get('total_episodes')} frames={info.get('total_frames')} fps={info.get('fps')}")

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Không thấy parquet trong {root/'data'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("index").reset_index(drop=True)

    vids = sorted((root / "videos").rglob(f"*{IMAGE_KEY}*/**/*.mp4")) or \
        sorted(p for p in (root / "videos").rglob("*.mp4") if IMAGE_KEY in str(p))
    if not vids:
        raise FileNotFoundError(f"Không thấy video {IMAGE_KEY} trong {root/'videos'}")
    images = np.concatenate([_decode_video(v) for v in vids])
    if len(images) != len(df):
        raise ValueError(f"Số frame video ({len(images)}) khác số dòng parquet ({len(df)}). "
                         "Kiểm tra lại phiên bản dataset / thứ tự file video.")

    ep = df["episode_index"].to_numpy().astype(np.int64)
    if np.any(np.diff(ep) < 0):
        raise ValueError("episode_index không tăng dần sau khi sắp theo index")
    episode_ends = np.flatnonzero(np.diff(ep)) + 1
    episode_ends = np.append(episode_ends, len(ep)).astype(np.int64)

    return RawPushT(
        images=images.astype(np.uint8),
        state=np.stack(df["observation.state"].to_numpy()).astype(np.float32),
        action=np.stack(df["action"].to_numpy()).astype(np.float32),
        reward=df["next.reward"].to_numpy().astype(np.float32),
        success=df["next.success"].to_numpy().astype(bool),
        episode_index=ep,
        episode_ends=episode_ends,
        fps=int(info.get("fps", 10)),
    )


def get_pusht(root: str | Path | None = None, cache: str | Path = "data/pusht_raw.npz",
              repo_id: str = "lerobot/pusht") -> RawPushT:
    """Load có cache: lần đầu tải + decode, các lần sau đọc file .npz."""
    cache = Path(cache)
    if cache.exists():
        return RawPushT.load(cache)
    if root is None:
        root = download(repo_id, local_dir=str(cache.parent / repo_id.replace("/", "__")))
    raw = load_raw(root)
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw.save(cache)
    return raw


# ----------------------------------------------------------------------------
# Chuẩn hóa
# ----------------------------------------------------------------------------
@dataclass
class MinMaxNormalizer:
    """Đưa về [-1, 1] như Diffusion Policy gốc."""
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "MinMaxNormalizer":
        return cls(x.min(0).astype(np.float32), x.max(0).astype(np.float32))

    def normalize(self, x):
        return (x - self.lo) / (self.hi - self.lo + 1e-8) * 2 - 1

    def unnormalize(self, x):
        return (x + 1) / 2 * (self.hi - self.lo + 1e-8) + self.lo

    def state_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_state_dict(cls, d):
        return cls(np.array(d["lo"], np.float32), np.array(d["hi"], np.float32))


# ----------------------------------------------------------------------------
# Cắt cửa sổ huấn luyện
# ----------------------------------------------------------------------------
def make_windows(episode_ends: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    """Trả về mảng [M, 2] gồm (chỉ số frame hiện tại t, episode), mỗi frame một mẫu.

    Trong __getitem__: obs tại t-n_obs+1..t, action tại t-n_obs+1..t-n_obs+horizon
    (giống Diffusion Policy: action chunk bắt đầu cùng frame obs cũ nhất).
    Ở hai đầu episode, chỉ số được kẹp (padding lặp frame biên).
    """
    out = []
    for ep in episodes:
        start = 0 if ep == 0 else int(episode_ends[ep - 1])
        end = int(episode_ends[ep])
        for t in range(start, end):
            out.append((t, ep))
    return np.asarray(out, dtype=np.int64)


class PushTWindowDataset:
    """Dataset PyTorch cho policy và world model.

    Mỗi phần tử là dict:
      image   [n_obs, 3, 96, 96] float trong [0, 1]
      state   [n_obs, 2]         chuẩn hóa [-1, 1]
      action  [horizon, 2]       chuẩn hóa [-1, 1]
      reward  [horizon]          reward thật của các bước trong chunk
      is_pad  [horizon]          True nếu bước action bị padding (ngoài episode)
    """

    def __init__(self, raw: RawPushT, episodes, n_obs: int = 2, horizon: int = 16,
                 state_norm: MinMaxNormalizer | None = None,
                 action_norm: MinMaxNormalizer | None = None):
        self.raw = raw
        self.n_obs, self.horizon = n_obs, horizon
        self.episodes = np.asarray(episodes)
        self.index = make_windows(raw.episode_ends, self.episodes)
        # Normalizer fit trên tập train; tập val phải dùng lại normalizer của train
        mask = np.isin(raw.episode_index, self.episodes)
        self.state_norm = state_norm or MinMaxNormalizer.fit(raw.state[mask])
        self.action_norm = action_norm or MinMaxNormalizer.fit(raw.action[mask])

        # Tính trước chỉ số obs/action cho mọi mẫu (dùng chung cho __getitem__ và get_batch)
        t = self.index[:, 0]
        ep = self.index[:, 1]
        s = np.where(ep == 0, 0, raw.episode_ends[np.maximum(ep - 1, 0)])
        e = raw.episode_ends[ep]
        t0 = t - n_obs + 1
        obs_raw = t0[:, None] + np.arange(n_obs)[None]
        act_raw = t0[:, None] + np.arange(horizon)[None]
        self.obs_idx = np.clip(obs_raw, s[:, None], e[:, None] - 1)
        self.act_idx = np.clip(act_raw, s[:, None], e[:, None] - 1)
        self.is_pad = (act_raw < s[:, None]) | (act_raw >= e[:, None])
        # state/action chuẩn hóa sẵn một lần
        self._state_n = self.state_norm.normalize(raw.state).astype(np.float32)
        self._action_n = self.action_norm.normalize(raw.action).astype(np.float32)

    def __len__(self):
        return len(self.index)

    def get_batch(self, ids, device="cpu") -> dict:
        """Lấy batch vector hóa (nhanh hơn DataLoader trên Windows).

        Ảnh trả về dạng uint8 [B, n_obs, 3, 96, 96]; chuyển sang float [0,1] trên GPU
        để giảm băng thông CPU->GPU.
        """
        import torch

        ids = np.asarray(ids)
        oi, ai = self.obs_idx[ids], self.act_idx[ids]
        img = torch.from_numpy(self.raw.images[oi]).to(device, non_blocking=True)
        return {
            "image": img.permute(0, 1, 4, 2, 3).contiguous(),
            "state": torch.from_numpy(self._state_n[oi]).to(device),
            "action": torch.from_numpy(self._action_n[ai]).to(device),
            "reward": torch.from_numpy(self.raw.reward[ai]).to(device),
            "is_pad": torch.from_numpy(self.is_pad[ids]).to(device),
        }

    def __getitem__(self, i):
        import torch

        oi, ai = self.obs_idx[i], self.act_idx[i]
        img = self.raw.images[oi].astype(np.float32) / 255.0
        return {
            "image": torch.from_numpy(img).permute(0, 3, 1, 2).contiguous(),
            "state": torch.from_numpy(self._state_n[oi]),
            "action": torch.from_numpy(self._action_n[ai]),
            "reward": torch.from_numpy(self.raw.reward[ai]),
            "is_pad": torch.from_numpy(self.is_pad[i]),
        }


def train_val_split(n_episodes: int, val_ratio: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_val = max(1, int(round(n_episodes * val_ratio)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])
