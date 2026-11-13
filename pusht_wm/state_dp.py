"""WM -> Diffusion (bản 2): Diffusion Policy nhìn thế giới QUA world model có cấu trúc.

World model = encoder (ảnh -> tư thế khối T, pusht_wm/pose_net.py) + dynamics (world model trạng thái, bước 11).
Policy nhận điều kiện từ world model, chế độ `feature`:
  state   : chỉ trạng thái (chấm tròn, khối T, sin/cos góc) ở t-1, t do encoder đọc     (mốc so sánh)
  wm      : + biểu diễn ẩn của dynamics h(s_{t-1}, s_t, a_{t-1})  (hiểu biết động lực học)
  wm_pred : + "tưởng tượng" của dynamics: trạng thái K bước tới nếu chấm tròn tiếp tục đi về đích action t-1
            (hậu quả của chuyển động hiện tại)
Dynamics đóng băng (đã học ở bước 11). Không dùng action tương lai làm đầu vào (không rò rỉ đáp án).
U-Net 1D khuếch tán, chunk 16 / thực thi 8, DDIM 10 bước: như baseline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MinMaxNormalizer
from .diffusion import NoiseScheduler
from .models import ConditionalUnet1D
from .state_wm import StateWorldModel, encode_state


@dataclass
class StateDPConfig:
    feature: str = "wm"              # state | wm | wm_pred
    state_wm_ckpt: str = "models/state_wm.pt"
    pose_net_ckpt: str = "models/pose_net.pt"
    wm_hidden: int = 512
    pred_k: int = 8
    n_obs: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    action_dim: int = 2
    cond_hidden: int = 256           # chiều sau khi nén biểu diễn ẩn của WM
    state_noise_pos: float = 1.0     # nhiễu khi train (đơn vị sim) ~ sai số encoder -> bền với nhãn/ước lượng lệch
    state_noise_deg: float = 0.5
    dropout: float = 0.0
    down_dims: list = field(default_factory=lambda: [256, 512, 1024])
    kernel_size: int = 5
    step_embed_dim: int = 128
    num_train_steps: int = 100
    num_inference_steps: int = 10
    # output = "action": Diffusion sinh thẳng chuỗi action (như Diffusion Policy gốc)
    # output = "path"  : Diffusion sinh ĐƯỜNG ĐI trạng thái (chấm tròn + khối T, 6 chiều/bước) -> mô hình nghịch -> action
    output: str = "action"
    inverse_ckpt: str = "models/inverse.pt"
    feas_w: float = 0.0              # khi chọn path bằng WM: phạt độ lệch giữa path và "tưởng tượng" của WM
    edge_w: float = 0.0              # phạt khối T (WM tưởng tượng) đi ra ngoài vùng giữa bàn (sát tường/góc dễ kẹt)
    edge_band: float = 150.0         # vùng giữa: |tâm khối T - 256| <= edge_band (đơn vị sim) không bị phạt
    # pred = "eps": U-Net đoán nhiễu (Diffusion Policy gốc); "x0": U-Net đoán THẲNG path -> loss trên path:
    #   L = MSE(path) + fde_w * FDE,  FDE = sqrt(dx^2 + dy^2) + R*|dθ| của khối T ở bước CUỐI (R = 60: nửa chiều dài
    #   khối T -> sai lệch góc quy ra độ lệch của đầu khối, cùng đơn vị với vị trí)
    pred: str = "eps"
    fde_w: float = 0.0
    fde_r: float = 60.0


def add_state_noise(sf: torch.Tensor, pos: float, deg: float) -> torch.Tensor:
    """sf [..., 5] (agent x,y, khối x,y, góc) -> nhiễu lên khối T (encoder đọc khối T; agent là số thật)."""
    n = torch.randn_like(sf[..., 2:5])
    return torch.cat([sf[..., :2], sf[..., 2:4] + pos * n[..., :2], sf[..., 4:5] + np.radians(deg) * n[..., 2:3]], -1)


def enc_state_t(sf: torch.Tensor) -> torch.Tensor:
    """torch [..., 5] -> [..., 6] (giống encode_state của numpy)."""
    return torch.cat([sf[..., :4] / 256.0 - 1.0, torch.sin(sf[..., 4:5]), torch.cos(sf[..., 4:5])], -1)


def path_fde(pred: torch.Tensor, true: torch.Tensor, r: float = 60.0) -> torch.Tensor:
    """FDE khối T ở bước CUỐI của path (đơn vị sim): sqrt(dx^2 + dy^2) + r*|dθ|.
    pred, true [B, H, 6] trạng thái mã hóa (x/256-1, ..., sin, cos)."""
    dp = (pred[:, -1, 2:4] - true[:, -1, 2:4]).pow(2).sum(-1).add(1e-10).sqrt() * 256.0
    sp, cp, st, ct = pred[:, -1, 4], pred[:, -1, 5], true[:, -1, 4], true[:, -1, 5]
    dth = torch.atan2(sp * ct - cp * st, cp * ct + sp * st)
    return dp + r * dth.abs()


class WMCond(nn.Module):
    """(trạng thái t-1, t, action t-1) --world model--> vector điều kiện cho U-Net."""

    def __init__(self, cfg: StateDPConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.n_obs * 6
        if cfg.feature in ("wm", "wm_pred"):
            self.h = nn.Sequential(nn.LayerNorm(cfg.wm_hidden), nn.Linear(cfg.wm_hidden, cfg.cond_hidden), nn.Mish())
            d += cfg.cond_hidden
        if cfg.feature == "wm_pred":
            d += cfg.pred_k * 6
        self.out_dim = d
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, wm: StateWorldModel | None, sf_obs: torch.Tensor, prev_act: torch.Tensor):
        """sf_obs [B, n_obs, 5] trạng thái (đơn vị sim); prev_act [B, 2] action đã làm ở t-1 (đơn vị sim)."""
        s = enc_state_t(sf_obs)                                   # [B, n_obs, 6]
        parts = [s.flatten(1)]
        if self.cfg.feature in ("wm", "wm_pred"):
            a = prev_act / 256.0 - 1.0
            with torch.no_grad():
                h = wm.hidden(s[:, -2], s[:, -1], a)
            parts.append(self.h(h))
            if self.cfg.feature == "wm_pred":
                with torch.no_grad():
                    fut = wm.rollout(s[:, -2], s[:, -1], a[:, None].expand(-1, self.cfg.pred_k, -1))
                parts.append(fut.flatten(1))
        return self.drop(torch.cat(parts, -1))


class StateDiffusionPolicy(nn.Module):
    def __init__(self, cfg: StateDPConfig, action_norm: MinMaxNormalizer):
        super().__init__()
        self.cfg = cfg
        self.cond = WMCond(cfg)
        self.out_dim = 6 if cfg.output == "path" else cfg.action_dim
        self.unet = ConditionalUnet1D(self.out_dim, self.cond.out_dim, cfg.step_embed_dim, cfg.down_dims,
                                      cfg.kernel_size)
        self.scheduler = NoiseScheduler(cfg.num_train_steps)
        self.action_norm = action_norm
        self.wm: StateWorldModel | None = None      # đóng băng, KHÔNG lưu trong state_dict của policy
        self.pose_net = None                        # encoder ảnh -> khối T (chỉ cần khi chạy từ ảnh)
        self.inverse = None                         # path -> action (chỉ khi output = "path"), đóng băng

    def attach(self, wm: StateWorldModel | None, pose_net=None, inverse=None):
        if inverse is not None:
            object.__setattr__(self, "inverse", inverse.eval().requires_grad_(False))
        object.__setattr__(self, "wm", wm.eval().requires_grad_(False) if wm is not None else None)
        object.__setattr__(self, "pose_net", pose_net.eval().requires_grad_(False) if pose_net is not None else None)
        return self

    def compute_loss(self, sf_obs, prev_act, action_n):
        cond = self.cond(self.wm, sf_obs, prev_act)
        noise = torch.randn_like(action_n)
        t = torch.randint(0, self.scheduler.T, (action_n.shape[0],), device=action_n.device)
        out = self.unet(self.scheduler.add_noise(action_n, noise, t), t, cond).float()
        if self.cfg.pred == "x0":
            loss = F.mse_loss(out, action_n)
            if self.cfg.fde_w > 0:
                loss = loss + self.cfg.fde_w * path_fde(out, action_n, self.cfg.fde_r).mean() / 256.0
            return loss
        return F.mse_loss(out, noise)

    def _eps_fn(self, cond):
        """hàm eps(x_t, t) cho bộ lấy mẫu; dạng x0 -> quy về eps = (x_t - sqrt(a) x0) / sqrt(1 - a)."""
        if self.cfg.pred != "x0":
            return lambda x, t: self.unet(x, t, cond)
        ac = self.scheduler.alphas_cumprod

        def f(x, t):
            a = ac.to(x.device)[t].float().view(-1, 1, 1)
            x0 = self.unet(x, t, cond).float().clamp(-1, 1)
            return (x - a.sqrt() * x0) / (1 - a).sqrt().clamp_min(1e-4)
        return f

    @torch.no_grad()
    def sample_chunk(self, sf_obs, prev_act, n_samples: int = 1, generator=None, num_inference_steps=None):
        """-> chunk ĐÃ CHUẨN HÓA [B*n_samples, horizon, 2]."""
        cond = self.cond(self.wm, sf_obs, prev_act)
        if n_samples > 1:
            cond = cond.repeat_interleave(n_samples, 0)
        k = num_inference_steps or self.cfg.num_inference_steps
        shape = (cond.shape[0], self.cfg.horizon, self.out_dim)
        return self.scheduler.sample(self._eps_fn(cond), shape, cond.device, k, generator)

    @torch.no_grad()
    def act_from_images(self, images: np.ndarray, agent_xy: np.ndarray, prev_act: np.ndarray, generator=None,
                        n_samples: int = 1):
        """Chạy online: ảnh uint8 [B, n_obs, 96, 96, 3], agent_xy [B, n_obs, 2], prev_act [B, 2] (đơn vị sim)
        -> action [B, n_action_steps, 2] (đơn vị sim)."""
        return self.act_from_state(self.read_state(images, agent_xy), prev_act, generator, n_samples)

    @torch.no_grad()
    def read_state(self, images: np.ndarray, agent_xy: np.ndarray) -> np.ndarray:
        """WM encoder đọc s_t: ảnh [B, n_obs, 96, 96, 3] + agent_xy [B, n_obs, 2] -> state_full [B, n_obs, 5]."""
        dev = next(self.unet.parameters()).device
        B, n = images.shape[:2]
        blk = self.pose_net.block_pose(images.reshape(-1, *images.shape[2:]), dev).reshape(B, n, 3)
        return np.concatenate([agent_xy.astype(np.float64), blk], -1)

    @torch.no_grad()
    def act_from_state(self, sf_obs: np.ndarray, prev_act: np.ndarray, generator=None, n_samples: int = 1):
        """n_samples > 1: Diffusion đề xuất N chunk, WORLD MODEL tưởng tượng hậu quả của từng chunk
        (rollout từ trạng thái hiện tại), chấm bằng chi phí dày theo coverage -> chọn chunk tốt nhất."""
        mb = max(1, 2048 // n_samples)          # lô nhỏ: B*N chunk + lưới coverage rất tốn VRAM khi B = 500 env
        if len(sf_obs) > mb:
            return np.concatenate([self.act_from_state(sf_obs[i:i + mb], np.asarray(prev_act)[i:i + mb], generator,
                                                       n_samples) for i in range(0, len(sf_obs), mb)])
        dev = next(self.unet.parameters()).device
        sf = torch.from_numpy(sf_obs).float().to(dev)
        if self.cfg.output == "path":
            return self._act_path(sf, torch.from_numpy(np.asarray(prev_act, np.float32)).to(dev), n_samples, generator)
        ch = self.sample_chunk(sf, torch.from_numpy(np.asarray(prev_act, np.float32)).to(dev),
                               n_samples=n_samples, generator=generator)
        s0 = self.cfg.n_obs - 1
        if n_samples > 1:
            from .state_planner import TeeCoverage, dense_cost_state
            if getattr(self, "_cover", None) is None:
                object.__setattr__(self, "_cover", TeeCoverage(dev))
            B = sf.shape[0]
            a = torch.from_numpy(self.action_norm.unnormalize(ch.cpu().numpy())).float().to(dev)   # [B*N, H, 2]
            s = enc_state_t(sf).repeat_interleave(n_samples, 0)                                     # [B*N, n_obs, 6]
            sel = self.wm if getattr(self, "select_wm", None) is None else self.select_wm     # WM dùng để CHỌN (có thể khác WM làm đặc trưng)
            traj = sel.rollout(s[:, -2], s[:, -1], a[:, s0:] / 256.0 - 1.0)                        # [B*N, H-s0, 6]
            cost, _ = dense_cost_state(traj, self._cover)
            best = cost.view(B, n_samples).argmin(1)
            ch = ch.view(B, n_samples, *ch.shape[1:])[torch.arange(B, device=dev), best]
        ch = self.action_norm.unnormalize(ch.cpu().numpy())
        return ch[:, s0:s0 + self.cfg.n_action_steps]

    def _cover_fn(self, dev):
        from .state_planner import TeeCoverage
        if getattr(self, "_cover", None) is None:
            object.__setattr__(self, "_cover", TeeCoverage(dev))
        return self._cover

    @torch.no_grad()
    def path_and_actions(self, sf, prev, n_samples=1, generator=None):
        """Diffusion sinh N path [B*N, H, 6]; 2 bước đầu thay bằng trạng thái QUAN SÁT (t-1, t);
        mô hình nghịch -> action chuẩn hóa [B*N, H-2, 2] (a_t ... a_{t+H-3})."""
        x = self.sample_chunk(sf, prev, n_samples=n_samples, generator=generator)
        s = enc_state_t(sf).repeat_interleave(n_samples, 0)                       # [B*N, n_obs, 6]
        p = torch.cat([s, x[:, self.cfg.n_obs:]], 1)
        sc = p[..., 4:6] / p[..., 4:6].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        p = torch.cat([p[..., :4], sc], -1)
        acts = self.inverse(p[:, :-2], p[:, 1:-1], p[:, 2:])
        return p, acts, s

    def path_cost(self, traj, p):
        """Chi phí chọn path (càng nhỏ càng tốt) -> (cost [N], coverage tưởng tượng [N, K]).
        traj: trạng thái WM tưởng tượng [N, K, 6]; p: path của Diffusion [N, K+2, 6] (chuẩn hóa x/256-1)."""
        from .state_planner import dense_cost_state
        cost, cov = dense_cost_state(traj, self._cover_fn(traj.device))
        if self.cfg.feas_w > 0:     # path phi thực tế: khối T trong path lệch khối T WM tưởng tượng (chuẩn hóa /256)
            cost = cost + self.cfg.feas_w * (traj[..., 2:4] - p[:, 2:, 2:4]).norm(dim=-1).mean(-1)
        if self.cfg.edge_w > 0:     # khối T bị đẩy ra ngoài vùng giữa bàn (về phía tường) -> dễ kẹt, chấm tròn không kéo được
            out = (traj[..., 2:4].abs() - self.cfg.edge_band / 256.0).clamp_min(0).sum(-1)
            cost = cost + self.cfg.edge_w * out.mean(-1)
        return cost, cov

    @torch.no_grad()
    def _act_path(self, sf, prev, n_samples, generator):
        p, acts, s = self.path_and_actions(sf, prev, n_samples, generator)
        if n_samples > 1:
            # WM SUY LUẬN: thực hiện các action suy ra từ path thì khối T đi đâu? chấm theo coverage (+ độ khả thi)
            from .state_planner import dense_cost_state
            sel = self.wm if getattr(self, "select_wm", None) is None else self.select_wm
            traj = sel.rollout(s[:, -2], s[:, -1], acts)                             # [B*N, H-2, 6]
            cost, _ = self.path_cost(traj, p)
            B = sf.shape[0]
            best = cost.view(B, n_samples).argmin(1)
            acts = acts.view(B, n_samples, *acts.shape[1:])[torch.arange(B, device=sf.device), best]
        return ((acts[:, :self.cfg.n_action_steps] + 1) * 256).cpu().numpy()

    def checkpoint(self, extra=None):
        sd = {k: v for k, v in self.state_dict().items()}
        return {"state_dp": True, "config": asdict(self.cfg), "state_dict": sd,
                "action_norm": self.action_norm.state_dict(), **(extra or {})}

    @classmethod
    def from_checkpoint(cls, ckpt, map_location="cpu", attach: bool = True):
        """Tải policy; attach=True: tự tải world model trạng thái + encoder ảnh theo đường dẫn trong config."""
        from .paths import resolve
        if not isinstance(ckpt, dict):
            ckpt = torch.load(resolve(ckpt), map_location=map_location, weights_only=False)
        p = cls(StateDPConfig(**ckpt["config"]), MinMaxNormalizer.from_state_dict(ckpt["action_norm"]))
        p.load_state_dict(ckpt["state_dict"])
        if attach:
            from .pose_net import PoseNet
            wm = StateWorldModel.from_checkpoint(p.cfg.state_wm_ckpt)
            pn = PoseNet()
            pn.load_state_dict(torch.load(resolve(p.cfg.pose_net_ckpt), map_location="cpu", weights_only=False)["state_dict"])
            inv = None
            if p.cfg.output == "path":
                from .inverse import InverseModel
                inv = InverseModel.from_checkpoint(p.cfg.inverse_ckpt)
            p.attach(wm, pn, inv)
        return p

    def to(self, *a, **k):
        super().to(*a, **k)
        if self.wm is not None:
            object.__setattr__(self, "wm", self.wm.to(*a, **k))
        if self.pose_net is not None:
            object.__setattr__(self, "pose_net", self.pose_net.to(*a, **k))
        if self.inverse is not None:
            object.__setattr__(self, "inverse", self.inverse.to(*a, **k))
        return self


__all__ = ["StateDPConfig", "StateDiffusionPolicy", "add_state_noise", "enc_state_t", "encode_state"]
