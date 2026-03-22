import numpy as np
import torch
from kaolin.ops.gaussian import sample_points_in_volume as kaolin_sample_points_in_volume

def _rotmat_to_quat_wxyz_torch(R: torch.Tensor) -> torch.Tensor:
    """R: (...,3,3) -> (...,4) in wxyz, numerically stable"""
    # Shoemake
    R = R.to(dtype=torch.float32)
    m00, m01, m02 = R[...,0,0], R[...,0,1], R[...,0,2]
    m10, m11, m12 = R[...,1,0], R[...,1,1], R[...,1,2]
    m20, m21, m22 = R[...,2,0], R[...,2,1], R[...,2,2]

    t = m00 + m11 + m22
    w = torch.empty_like(t)
    x = torch.empty_like(t)
    y = torch.empty_like(t)
    z = torch.empty_like(t)

    # 3 conditions
    cond0 = t > 0.0
    cond1 = (~cond0) & (m00 > m11) & (m00 > m22)
    cond2 = (~cond0) & (~cond1) & (m11 > m22)
    cond3 = (~cond0) & (~cond1) & (~cond2)

    # t > 0
    r = torch.sqrt(torch.clamp(t + 1.0, min=1e-12))
    w = torch.where(cond0, 0.5 * r, w)
    inv4 = torch.where(cond0, 0.5 / r, torch.zeros_like(r))
    x = torch.where(cond0, (m21 - m12) * inv4, x)
    y = torch.where(cond0, (m02 - m20) * inv4, y)
    z = torch.where(cond0, (m10 - m01) * inv4, z)

    # m00 
    r1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12))
    x = torch.where(cond1, 0.5 * r1, x)
    inv4 = torch.where(cond1, 0.5 / r1, inv4)
    w = torch.where(cond1, (m21 - m12) * inv4, w)
    y = torch.where(cond1, (m01 + m10) * inv4, y)
    z = torch.where(cond1, (m02 + m20) * inv4, z)

    # m11 
    r2 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-12))
    y = torch.where(cond2, 0.5 * r2, y)
    inv4 = torch.where(cond2, 0.5 / r2, inv4)
    w = torch.where(cond2, (m02 - m20) * inv4, w)
    x = torch.where(cond2, (m01 + m10) * inv4, x)
    z = torch.where(cond2, (m12 + m21) * inv4, z)

    # m22 
    r3 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-12))
    z = torch.where(cond3, 0.5 * r3, z)
    inv4 = torch.where(cond3, 0.5 / r3, inv4)
    w = torch.where(cond3, (m10 - m01) * inv4, w)
    x = torch.where(cond3, (m02 + m20) * inv4, x)
    y = torch.where(cond3, (m12 + m21) * inv4, y)

    q = torch.stack([w, x, y, z], dim=-1)
    q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-12)
    return q

def _ensure_quat_wxyz(rotations: np.ndarray, device: str) -> torch.Tensor:

    if rotations.ndim == 2 and rotations.shape[-1] == 4:
        q = torch.from_numpy(rotations.astype(np.float32))
        q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-12)
        return q.to(device)
    elif rotations.ndim == 2 and rotations.shape[-1] == 9:
        R = torch.from_numpy(rotations.astype(np.float32).reshape(-1,3,3)).to(device)
        return _rotmat_to_quat_wxyz_torch(R)
    elif rotations.ndim == 3 and rotations.shape[-2:] == (3,3):
        R = torch.from_numpy(rotations.astype(np.float32)).to(device)
        return _rotmat_to_quat_wxyz_torch(R)
    else:
        raise ValueError(f"rotations shape not supported: {rotations.shape}")



def _gs_to_voxel_points(
    means, scales, rotations, *,
    weights=None,
    voxel_size: float = 0.01,   
    bbox_margin: float = 3.0,
    thresh: float = 0.4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    octree_level: int = 6
) -> np.ndarray:
    xyz_t   = torch.from_numpy(means.astype(np.float32)).to(device)
    scale_t = torch.from_numpy(scales.astype(np.float32)).to(device)
    quat_t  = _ensure_quat_wxyz(rotations, device=device)
    opa_t   = torch.ones((xyz_t.shape[0],1), device=device) if weights is None \
              else torch.from_numpy(weights.astype(np.float32)).to(device)

    mins = xyz_t.min(0).values; maxs = xyz_t.max(0).values
    extent = (maxs - mins); extent = extent + bbox_margin * extent.max().clamp(min=1e-6)
    max_len = float(extent.max().item())

    target_N = max(4.0, np.ceil(max_len / max(1e-6, float(voxel_size))))
    level_est = int(np.ceil(np.log(max(target_N, 1.0)) / np.log(3.0))) + 1
    print("Estimated octree level:", level_est)
    octree_level = 6 # int(min(level_est, octree_level))

    pts = kaolin_sample_points_in_volume(
        xyz=xyz_t, scale=scale_t, rotation=quat_t, opacity=opa_t,
        num_samples=None, octree_level=octree_level,
        # opacity_threshold=float(min(thresh, 0.1)),  
        opacity_threshold=thresh, 
        post_scale_factor=1.0, jitter=False, clip_samples_to_input_bbox=False,
        viewpoints=None, scaling_activation=None, scaling_inverse_activation=None
    )
    pts_np = pts.detach().cpu().numpy().astype(np.float32)
    return pts_np


# def _gs_to_voxel_points(
#     means: np.ndarray,          # (N,3), float32
#     scales: np.ndarray,         # (N,3), float32, activated scale
#     rotations: np.ndarray,      # (N,4) wxyz or (N,3,3) or (N,9)
#     *,
#     weights: np.ndarray = None, # (N,) or (N,1), float32
#     voxel_size: float = 0.01,
#     bbox_margin: float = 3.0,
#     thresh: float = 0.4,
#     domain_min: np.ndarray = None,
#     domain_max: np.ndarray = None,
#     device: str = "cuda" if torch.cuda.is_available() else "cpu",
# ) -> np.ndarray:

#     # --- input tensors ---
#     xyz_t   = torch.from_numpy(means.astype(np.float32)).to(device)
#     scale_t = torch.from_numpy(scales.astype(np.float32)).to(device)
#     quat_t  = _ensure_quat_wxyz(rotations, device=device)        # (N,4) wxyz

#     if weights is None:
#         opa_t = torch.ones((xyz_t.shape[0], 1), device=device, dtype=torch.float32)
#     else:
#         opa_t = torch.from_numpy(weights.astype(np.float32).reshape(-1,1)).to(device)

#     # --- map voxel_size -> octree_level ---
#     mins = xyz_t.min(dim=0).values
#     maxs = xyz_t.max(dim=0).values
#     extent = (maxs - mins)
#     extent = extent + bbox_margin * extent.max().clamp(min=1e-6)
#     max_len = float(extent.max().item())
#     target_N = max(4.0, max_len / max(1e-6, float(voxel_size)))
#     level_est = int(round(np.log(max(target_N, 1.0)) / np.log(3.0)))
#     octree_level = int(np.clip(level_est, 6, 10))

#     # --- Kaolin voxelization + volume fill ---
#     pts = kaolin_sample_points_in_volume(
#         xyz=xyz_t,                 # (N,3)
#         scale=scale_t,             # (N,3)
#         rotation=quat_t,           # (N,4) wxyz
#         opacity=opa_t,             # (N,1)
#         mask=None,
#         num_samples=None,
#         octree_level=octree_level,
#         opacity_threshold=float(thresh),
#         post_scale_factor=1.0,
#         jitter=False,
#         clip_samples_to_input_bbox=True,
#         viewpoints=None,
#         scaling_activation=None,
#         scaling_inverse_activation=None,
#     )  # (K,3)

#     pts_np = pts.detach().cpu().numpy().astype(np.float32)

#     # --- optional domain clip ---
#     if (domain_min is not None) and (domain_max is not None):
#         dm = np.asarray(domain_min, dtype=np.float32)
#         dM = np.asarray(domain_max, dtype=np.float32)
#         mask = (pts_np >= dm[None, :]).all(axis=1) & (pts_np <= dM[None, :]).all(axis=1)
#         pts_np = pts_np[mask]

#     return pts_np

# def _gs_to_voxel_points(
#     means, scales, rotations, weights=None,
#     voxel_size=0.01, bbox_margin=3.0, thresh=0.4,
#     domain_min=None, domain_max=None,
# ):
#     N = means.shape[0]
#     if weights is None:
#         weights = np.ones((N,), dtype=np.float32)

#     ext = np.empty_like(means, dtype=np.float32)
#     for i in range(N):
#         R = rotations[i]; s = scales[i]
#         ext[i] = np.sum(np.abs(R) * s, axis=1) * bbox_margin

#     gmin = np.min(means - ext, axis=0)
#     gmax = np.max(means + ext, axis=0)
#     if domain_min is not None: gmin = np.maximum(gmin, np.array(domain_min, np.float32))
#     if domain_max is not None: gmax = np.minimum(gmax, np.array(domain_max, np.float32))

#     xs = np.arange(gmin[0], gmax[0], voxel_size, dtype=np.float32)
#     ys = np.arange(gmin[1], gmax[1], voxel_size, dtype=np.float32)
#     zs = np.arange(gmin[2], gmax[2], voxel_size, dtype=np.float32)
#     if len(xs)==0 or len(ys)==0 or len(zs)==0:
#         return np.zeros((0,3), dtype=np.float32)

#     X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
#     grid = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
#     rho = np.zeros((grid.shape[0],), dtype=np.float32)

#     for i in tqdm(range(N)):
#         mu = means[i]; R = rotations[i]; sig = scales[i]
#         Sigma_inv = R @ np.diag(1.0/(sig**2)) @ R.T
#         diff = grid - mu
#         r = np.linalg.norm(diff, axis=1)
#         idx = np.where(r <= bbox_margin * float(np.max(sig)))[0]
#         if idx.size == 0: continue
#         d = diff[idx]
#         quad = np.einsum('bi,ij,bj->b', d, Sigma_inv, d)
#         rho[idx] += (weights[i] if weights is not None else 1.0) * np.exp(-0.5 * quad).astype(np.float32)

#     return grid[rho >= thresh].astype(np.float32)

