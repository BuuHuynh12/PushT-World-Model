"""pusht_wm — World Model latent + Diffusion sinh path cho Push-T.

frame -> WM (DINOv2 + transformer) -> s_t latent -> Diffusion sinh 16 PATH robot -> WM tưởng tượng hậu quả, chọn path
-> path -> action bằng toán (giải ngược bộ điều khiển PD của simulator).
"""
