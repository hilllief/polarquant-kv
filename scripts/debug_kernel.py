"""调试融合 score kernel。"""
import cupy as cp
import torch
import math

from polarquant_kv_cuda.rotation import generate_rotation_matrix
from polarquant_kv_cuda.compress_kernel import compress_gpu, _bit_unpack_quantized
from polarquant_kv_cuda.decompress_kernel import decompress_gpu
from polarquant_kv_cuda.fused_cuda_kernels import _FUSED_SCORE_4BIT

D = 128
R = generate_rotation_matrix(D, seed=42, device="cuda")
torch.manual_seed(42)

K = torch.randn(1, 1, 1, D, dtype=torch.float16, device="cuda")
ck = compress_gpu(K, R, n_bits=4, group_size=32)
K_hat = decompress_gpu(ck, R).squeeze()

q = torch.randn(D, dtype=torch.float32, device="cuda")
q_rot = (q @ R.T).contiguous()

score_std = (q @ K_hat.float() / math.sqrt(D)).item()

# 手动解压
quant = _bit_unpack_quantized(ck.quantized_direction.squeeze(), 4, D).float()
gm = ck.group_mins.squeeze().float()
gs = ck.group_scales.squeeze().float()
rad_val = ck.radius.squeeze().float().item()
quant_g = quant.reshape(4, 32)
direction = (quant_g * gs.unsqueeze(-1) + gm.unsqueeze(-1)).reshape(D)
score_manual = (torch.dot(q_rot, direction) * rad_val / math.sqrt(D)).item()

print(f"Score std:    {score_std:.6f}")
print(f"Score manual: {score_manual:.6f}")

# 用纯 CuPy 数组（不用 DLPack）
q_cp = cp.asarray(q_rot.cpu().numpy())
pk_np = ck.quantized_direction.squeeze().cpu().numpy()
pk_cp = cp.asarray(pk_np)
gm_cp = cp.asarray(gm.cpu().numpy())
gs_cp = cp.asarray(gs.cpu().numpy())
rad_cp = cp.asarray(ck.radius.squeeze().float().cpu().numpy())

print(f"\nCuPy arrays:")
print(f"  q_rot[:4]: {q_cp[:4]}")
print(f"  pk[:4]:    {pk_cp[:4]}")
print(f"  gm:        {gm_cp}")
print(f"  gs:        {gs_cp}")
print(f"  rad:       {rad_cp}")

scores_cp = cp.zeros(1, dtype=cp.float32)
_FUSED_SCORE_4BIT(
    (1,), (256,),
    (q_cp, pk_cp, gm_cp, gs_cp, rad_cp,
     scores_cp, 1, D, 32, 4, 1.0 / math.sqrt(D)),
)
print(f"Score kernel (cupy arrays): {float(scores_cp[0]):.6f}")
