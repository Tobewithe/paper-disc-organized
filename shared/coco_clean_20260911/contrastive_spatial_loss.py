"""Contrastive & Orthogonal Spatial Regularizers.

Implements:
1. Innovation 2: Pairwise Soft Contrastive Margin Loss (NC-DMS) for co-occurring instances in the same image.
2. Innovation 4: Prototype Grassmannian Orthogonality Regularization & Centerness Soft Geometric Prior.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def prototype_orthogonality_loss(p_tensor):
    """Encourages 32 prototype channels to be mutually orthogonal (decorrelated).
    
    Args:
        p_tensor: [B, N_pts, 32] or [32, H, W]
    Returns:
        scalar loss
    """
    if p_tensor.dim() == 3 and p_tensor.shape[-1] == 32:
        # Flatten B x N_pts -> M x 32
        p_flat = p_tensor.reshape(-1, 32)
    elif p_tensor.dim() == 3 and p_tensor.shape[0] == 32:
        p_flat = p_tensor.permute(1, 2, 0).reshape(-1, 32)
    else:
        p_flat = p_tensor.reshape(-1, 32)
        
    M = p_flat.shape[0]
    p_centered = p_flat - p_flat.mean(dim=0, keepdim=True)
    cov = (p_centered.T @ p_centered) / max(M - 1, 1)  # [32, 32]
    
    # Normalize to correlation matrix
    std = torch.sqrt(torch.diag(cov).clamp_min(1e-6))
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
    
    # Target identity matrix
    eye = torch.eye(32, device=p_tensor.device, dtype=p_tensor.dtype)
    off_diag = (corr - eye).square().mean()
    return off_diag


def pairwise_contrastive_loss(z_all, y_all, image_ids, margin=1.5):
    """Pairwise contrastive margin loss for targets occurring in the same image.
    
    Args:
        z_all: [B, 512] predicted logits
        y_all: [B, 512] binary GT labels
        image_ids: list/tensor of length B indicating image origin
        margin: separation margin (gamma)
    """
    loss = torch.tensor(0.0, device=z_all.device)
    pairs_count = 0
    B = len(image_ids)
    
    # Find targets belonging to the same image
    for i in range(B):
        for j in range(i + 1, min(i + 16, B)):
            if image_ids[i] == image_ids[j]:
                # Both targets are in the same image
                # In target i's sampled points, target i should dominate where y_i == 1
                pos_i = (y_all[i] == 1.0)
                if pos_i.any():
                    # Difference z_i - z_j should be at least margin
                    diff = z_all[i][pos_i] - z_all[j][pos_i]
                    pair_loss_i = F.relu(margin - diff).mean()
                    loss = loss + pair_loss_i
                    pairs_count += 1
                    
                pos_j = (y_all[j] == 1.0)
                if pos_j.any():
                    # In target j's points, target j should dominate where y_j == 1
                    diff = z_all[j][pos_j] - z_all[i][pos_j]
                    pair_loss_j = F.relu(margin - diff).mean()
                    loss = loss + pair_loss_j
                    pairs_count += 1
                    
    if pairs_count > 0:
        loss = loss / pairs_count
    return loss


def apply_centerness_prior(z, boxes, shape, margin_scale=0.5, scale=None):
    """Box-geometry-aligned soft perimeter attenuation prior.
    
    Inside the object core: zero penalty (preserves interior corners and diagonal tips).
    Near perimeter and outside: smooth soft negative penalty to contain outward spillover.
    
    Args:
        z: [B, H, W] logits
        boxes: [B, 4] in xyxy format
        shape: (H, W)
        margin_scale: softness scale
        scale: alias for margin_scale
    """
    if scale is not None:
        margin_scale = scale
    B, H, W = z.shape
    if B == 0:
        return z
        
    device = z.device
    y_coords = torch.linspace(0, H - 1, H, device=device)
    x_coords = torch.linspace(0, W - 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    hw = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0) / 2.0
    hh = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0) / 2.0
    
    # Normalized coordinate offset from center
    norm_dx = (grid_x.unsqueeze(0) - cx.unsqueeze(-1).unsqueeze(-1)).abs() / (hw.unsqueeze(-1).unsqueeze(-1))
    norm_dy = (grid_y.unsqueeze(0) - cy.unsqueeze(-1).unsqueeze(-1)).abs() / (hh.unsqueeze(-1).unsqueeze(-1))
    
    # Chebychev / rectangular boundary distance
    # Core (< 0.85 half-box): 0 penalty. Perimeter / outside (> 0.85): smooth penalty
    excess_dist = (torch.maximum(norm_dx, norm_dy) - 0.85).clamp_min(0.0) / margin_scale
    penalty = 1.5 * excess_dist.clamp_max(2.0)
    return z - penalty
