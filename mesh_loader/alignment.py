import torch

def umeyama_similarity(P: torch.Tensor, Q: torch.Tensor):
    """
    P, Q: (B,3) or (N,3) point‐sets.
    Returns R (3×3), s (scalar), t (3,) all on P.device/dtype.
    """
    assert P.shape == Q.shape
    N = P.shape[0]
    device, dtype = P.device, P.dtype

    muP = P.mean(0)
    muQ = Q.mean(0)
    P0  = P - muP
    Q0  = Q - muQ

    C = (Q0.T @ P0) / N
    U, Svals, Vt = torch.linalg.svd(C)
    S_mat = torch.eye(3, device=device, dtype=dtype)
    if (torch.linalg.det(U) * torch.linalg.det(Vt) < 0).item():
        S_mat[2,2] = -1
    R = U @ S_mat @ Vt

    scale = (Svals * torch.diag(S_mat)).sum() / P0.pow(2).sum()
    t     = muQ - scale * (R @ muP)

    return R, scale, t


def align_model_to_gt(predictions, gt):
    """
    Align predictions["world_points"] and predictions["extrinsic"]
    so that the camera translations match gt["extrinsic"].
    Keeps extrinsic in shape (B,3,4).
    """
    # ————————————— Extract
    c_pred = predictions["extrinsic"]  # (B,3,4)
    c_gt   = gt["extrinsic"]           # (B,3,4)
    B, _, _ = c_pred.shape
    device, dtype = c_pred.device, c_pred.dtype

    # camera centers: (B,3)
    t_pred = c_pred[..., :3, 3]
    t_gt   = c_gt  [..., :3, 3]

    # ————————————— Compute similarity
    R, s, t = umeyama_similarity(t_pred, t_gt)  # on GPU

    # ————————————— Align world-points
    pts = predictions["world_points"].reshape(-1, 3)
    pts = (R @ pts.T).T
    predictions["world_points"] = pts * s + t

    # ————————————— Build 4×4 S
    S = torch.eye(4, device=device, dtype=dtype)
    S[:3, :3] = s * R
    S[:3,  3] = t
    S_batch = S.unsqueeze(0).expand(B, 4, 4)  # (B,4,4)

    # ————————————— Pad extrinsics to homogeneous
    ones_row = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)
    ones_row = ones_row.view(1,1,4).expand(B,1,4)  # (B,1,4)
    extr_homo = torch.cat([c_pred, ones_row], dim=1)  # (B,4,4)

    # ————————————— Apply similarity & slice back
    extr_aligned_homo = S_batch @ extr_homo           # (B,4,4)

    predictions["extrinsic"] = extr_aligned_homo
    return predictions
