"""Bộ điều khiển của MÔ HÌNH CHÍNH (dùng cho app): nhận ảnh, trả về các action cần thực thi.

    frame (2 ảnh) -> WM encoder đọc s_t -> Diffusion sinh 16 PATH trạng thái -> WM dynamics tưởng tượng hậu quả
    của từng path + phạt path phi thực tế (0.5) -> chọn path tốt nhất -> mô hình nghịch -> 8 action

Test 500 seed 600000..600499: 89.8% thành công, trung vị 143 bước.
(Bộ điều khiển cũ có các chế độ thí nghiệm khác: archive/pusht_wm_old/controller_old.py)
"""
from __future__ import annotations

import numpy as np
import torch

from .paths import PATH_POLICY

N_PATHS = 16      # số path Diffusion đề xuất mỗi lần nghĩ
N_ACT = 8         # số action thực thi trước khi nghĩ lại
FEAS_W = 0.5      # phạt: khối T trong path lệch khối T WM tưởng tượng (path phi thực tế)
EDGE_W = 0.0      # phạt: WM tưởng tượng khối T bị đẩy về phía tường (0 = tắt, như bản đã test 89.8%)
CONS_W = 1.0      # phạt ĐỔI Ý (khi bật, phím C trong app): chấm tròn của path mới lệch phần còn lại của path đã chọn lần
                  # trước -> kiên trì một kế hoạch, bớt chạy qua chạy lại. CHƯA đánh giá trên 500 seed: mặc định TẮT.
DESC = "frame → WM đọc s_t → Diffusion 16 PATH → WM chọn → action"


class Controller:
    def __init__(self, ckpt: str = PATH_POLICY, device=None, seed: int = 0):
        from .state_dp import StateDiffusionPolicy

        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.backends.cudnn.benchmark = True
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.sp = StateDiffusionPolicy.from_checkpoint(ckpt).to(self.dev).eval()
        self.sp.cfg.feas_w, self.sp.cfg.edge_w = FEAS_W, EDGE_W
        self.read = None                   # s_t encoder đọc được ở lần nghĩ gần nhất (khối T: x, y, góc)
        self.imagination = None
        self.consistency = False           # bật/tắt phạt đổi ý
        self.prev_plan = None              # path (trạng thái chuẩn hóa) đã chọn ở lần nghĩ trước

    def reset(self, obs):
        self.hist_img = [obs["pixels"]] * 2
        self.hist_pos = [np.asarray(obs["agent_pos"], np.float32)] * 2
        self.prev_act = np.asarray(obs["agent_pos"], np.float32)
        self.prev_plan = None

    def observe(self, obs, action):
        """Gọi sau mỗi env.step(action)."""
        self.hist_img = [self.hist_img[-1], obs["pixels"]]
        self.hist_pos = [self.hist_pos[-1], np.asarray(obs["agent_pos"], np.float32)]
        self.prev_act = np.asarray(action, np.float32)

    @torch.no_grad()
    def next_actions(self) -> np.ndarray:
        """-> action [N_ACT, 2] (đơn vị sim) cần thực thi tiếp; lưu "suy nghĩ" vào self.imagination để app vẽ."""
        sp, dev = self.sp, self.dev
        imgs = np.stack(self.hist_img)                                         # [2, 96, 96, 3]
        blk = sp.pose_net.block_pose(imgs, dev)                                # [2, 3] WM encoder: khối T đọc từ ảnh
        sf = np.concatenate([np.stack(self.hist_pos).astype(np.float64), blk], -1)[None]   # [1, 2, 5] = s_t
        self.read = blk[-1]
        sft = torch.from_numpy(sf).float().to(dev)
        prev = torch.from_numpy(self.prev_act[None].astype(np.float32)).to(dev)
        # Diffusion sinh N PATH trạng thái [N, 16, 6] -> mô hình nghịch -> action [N, 14, 2] (chuẩn hóa)
        pth, an, s = sp.path_and_actions(sft, prev, n_samples=N_PATHS, generator=self.gen)
        a = ((an + 1) * 256).clamp(0, 512)
        traj = sp.wm.rollout(s[:, -2], s[:, -1], an)                           # WM dynamics tưởng tượng hậu quả
        cost, cov = sp.path_cost(traj, pth)                                    # coverage + phạt phi thực tế (+ sát tường)
        if self.consistency and self.prev_plan is not None:
            # phần CHƯA làm của kế hoạch trước (bước N_ACT.. của 14 bước tương lai) <-> đầu path mới (cùng thời điểm)
            rem = self.prev_plan[N_ACT:]                                       # [14 - N_ACT, 6]
            cons = (pth[:, 2:2 + len(rem), :2] - rem[None, :, :2]).norm(dim=-1).mean(-1)   # lệch chấm tròn (/256)
            cost = cost + CONS_W * cons
        pick = int(cost.argmin())
        self.prev_plan = pth[pick, 2:].clone()
        plan = pth[:, 2:].float().cpu().numpy().copy()                         # khối T mà path MUỐN đạt
        plan[..., :4] = (plan[..., :4] + 1) / 2
        pose = traj.float().cpu().numpy().copy()                               # khối T WM TƯỞNG TƯỢNG sẽ xảy ra
        pose[..., :4] = (pose[..., :4] + 1) / 2                                # -> toạ độ 0..1 như app vẽ
        self.imagination = {"paths": a.cpu().numpy(), "pick": pick, "act": N_ACT, "pose": pose,
                            "cov": cov.float().cpu().numpy(), "score": -cost.float().cpu().numpy(), "plan": plan}
        return a[pick, :N_ACT].cpu().numpy()
