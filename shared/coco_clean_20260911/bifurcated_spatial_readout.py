"""BSR-Head: Bifurcated Spatial-Readout Head.

Grounded on RESEARCH_FACTS.md (S080, S074, S032).
Bifurcates mask readout into:
1. Coefficient Filtering Branch: 32D residual for surplus false-positive removal.
2. Spatial Prototype Residual Branch: non-linear prototype mapping for missing target pixel recovery.
3. Adaptive Gate Router: infers instance-level routing g_i in [0, 1] strictly from inference-visible features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GateRouter(nn.Module):
    """Predicts instance routing gate g_i in [0, 1].
    
    Input: normalized instance feature xn (73-dim) + base coefficients c (32-dim) = 105-dim.
    """
    def __init__(self, in_dim=105, hidden=64, init_bias=-1.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        # Initialize final layer with small weights and negative bias (initial gate ~ 0.18)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, xn, c):
        feat = torch.cat([xn, c], dim=-1)
        logit = self.net(feat)
        return torch.sigmoid(logit)  # [B, 1]


class CoefficientBranch(nn.Module):
    """S032-compatible 32D coefficient residual branch.
    
    Input: normalized instance feature xn (73-dim).
    Output: delta_c (32-dim).
    """
    def __init__(self, in_dim=73, hidden=128, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 128)
        )
        # Small normal initialization ensures initial output is ~1e-4 while allowing gradients to flow to all layers
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, xn):
        return self.net(xn).reshape(-1, 4, 32).mean(1)  # [B, 32]


class SpatialResidualBranch(nn.Module):
    """Non-linear prototype spatial residual branch.
    
    Generates point-wise / pixel-wise logit corrections conditioned on instance representation.
    """
    def __init__(self, p_dim=32, h_dim=73, hidden=64):
        super().__init__()
        self.p_proj = nn.Linear(p_dim, hidden)
        self.h_proj = nn.Linear(h_dim, hidden, bias=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        # Small normal initialization ensures initial delta is ~1e-4 while allowing gradients to flow
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward_sampled(self, p_sampled, xn):
        """Forward on sampled points [B, N_pts, 32]."""
        p_feat = self.p_proj(p_sampled)  # [B, N_pts, hidden]
        h_feat = self.h_proj(xn).unsqueeze(1)  # [B, 1, hidden]
        delta_z = self.mlp(p_feat + h_feat).squeeze(-1)  # [B, N_pts]
        return delta_z

    def forward_grid(self, p_proto, xn_inst):
        """Forward on 2D prototype grid.
        
        Args:
            p_proto: [32, H, W]
            xn_inst: [B, 73]
        Returns:
            delta_z: [B, H, W]
        """
        H, W = p_proto.shape[1], p_proto.shape[2]
        p_perm = p_proto.permute(1, 2, 0)  # [H, W, 32]
        p_feat = self.p_proj(p_perm)  # [H, W, hidden]
        h_feat = self.h_proj(xn_inst)  # [B, hidden]
        
        # Broadcast add: [B, H, W, hidden]
        fused = p_feat.unsqueeze(0) + h_feat.unsqueeze(1).unsqueeze(1)
        delta_z = self.mlp(fused).squeeze(-1)  # [B, H, W]
        return delta_z


class BifurcatedSpatialHead(nn.Module):
    """Unified Bifurcated Spatial-Readout Head (BSR-Head)."""
    def __init__(self, in_dim=73, p_dim=32, hidden=128):
        super().__init__()
        self.coeff_branch = CoefficientBranch(in_dim=in_dim, hidden=hidden, out_dim=p_dim)
        self.spatial_branch = SpatialResidualBranch(p_dim=p_dim, h_dim=in_dim, hidden=64)
        self.gate_router = GateRouter(in_dim=in_dim + p_dim, hidden=64, init_bias=-1.5)

    def forward_sampled(self, xn, c, p_sampled):
        """Sampled-point forward during training.
        
        Args:
            xn: [B, 73] normalized instance features
            c: [B, 32] base coefficient vector
            p_sampled: [B, 512, 32] sampled prototype values
        Returns:
            z_final: [B, 512]
            info: dict containing gate, delta_c, delta_z_spatial
        """
        delta_c = self.coeff_branch(xn)  # [B, 32]
        c_refined = c + delta_c  # [B, 32]
        z_coeff = (p_sampled * c_refined.unsqueeze(1)).sum(-1)  # [B, 512]
        
        delta_z_spatial = self.spatial_branch.forward_sampled(p_sampled, xn)  # [B, 512]
        gate = self.gate_router(xn, c)  # [B, 1]
        
        z_final = z_coeff + gate * delta_z_spatial
        return z_final, dict(gate=gate, delta_c=delta_c, delta_z_spatial=delta_z_spatial, z_coeff=z_coeff)

    def forward_inference(self, xn, c, p_proto):
        """Full 2D inference forward.
        
        Args:
            xn: [B, 73]
            c: [B, 32]
            p_proto: [32, H, W]
        Returns:
            z_final: [B, H, W]
            gate: [B, 1]
        """
        B = len(c)
        if B == 0:
            return torch.zeros((0, p_proto.shape[1], p_proto.shape[2]), device=c.device), torch.zeros((0, 1), device=c.device)
            
        delta_c = self.coeff_branch(xn)  # [B, 32]
        c_refined = c + delta_c  # [B, 32]
        
        # [B, H, W] from linear combination
        H, W = p_proto.shape[1], p_proto.shape[2]
        z_coeff = torch.einsum('bc,chw->bhw', c_refined, p_proto)
        
        delta_z_spatial = self.spatial_branch.forward_grid(p_proto, xn)  # [B, H, W]
        gate = self.gate_router(xn, c)  # [B, 1]
        
        z_final = z_coeff + gate.unsqueeze(-1) * delta_z_spatial
        return z_final, gate


def bsr_loss(z_final, y, factor, info, lambda_spatial=0.001):
    """BSR-Head training loss.
    
    Combines BCE+Dice with spatial L2 regularization.
    """
    bce = F.binary_cross_entropy_with_logits(z_final, y, reduction='none').mean(1) * factor
    prob = z_final.sigmoid()
    dice = 1.0 - (2.0 * (prob * y).sum(1) + 1.0) / (prob.sum(1) + y.sum(1) + 1.0)
    task_loss = (bce + dice).mean()
    
    # Regularization: spatial residual shrinkage weighted by gate
    spatial_reg = (info['gate'] * info['delta_z_spatial'].square().mean(1, keepdim=True)).mean()
    total_loss = task_loss + lambda_spatial * spatial_reg
    return total_loss, dict(task_loss=float(task_loss.detach()), spatial_reg=float(spatial_reg.detach()))
