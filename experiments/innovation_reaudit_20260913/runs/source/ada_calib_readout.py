"""Ada-Calib: Instance-Adaptive Dynamic Decision & Logit Calibration Head.

Grounded on RESEARCH_FACTS.md (S034, S035, and Fact 5):
Negative margin supervision pushes boundary pre-activation logits negatively,
creating a steep logit cliff. Fixing a static global threshold p=0.50 hurts recall.
Ada-Calib predicts an instance-specific scale gamma_i and bias beta_i from inference-visible
features, dynamically adapting the decision boundary without manual threshold tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from bifurcated_spatial_readout import BifurcatedSpatialHead


class InstanceCalibrator(nn.Module):
    """Predicts instance-level logit scale gamma and bias beta.
    
    Input: [B, in_dim] normalized features (e.g. 73-dim xn or 105-dim xn+c).
    Output:
        gamma: [B, 1] scale factor in [0.45, 2.22]
        beta:  [B, 1] bias term in [-1.5, 1.5]
    """
    def __init__(self, in_dim=105, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)
        )
        # Zero-initialization: starts with gamma = 1.0, beta = 0.0 (identity mapping)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat):
        out = self.net(feat)  # [B, 2]
        log_gamma = torch.clamp(out[:, 0:1], -0.8, 0.8)
        gamma = torch.exp(log_gamma)
        beta = 1.5 * torch.tanh(out[:, 1:2])
        return gamma, beta


class AdaCalibHead(nn.Module):
    """Wraps a mask head (e.g. BSR-Head or S032) with InstanceCalibrator."""
    def __init__(self, base_head=None, in_dim=73, p_dim=32):
        super().__init__()
        self.base_head = base_head if base_head is not None else BifurcatedSpatialHead(in_dim=in_dim, p_dim=p_dim)
        self.calibrator = InstanceCalibrator(in_dim=in_dim + p_dim, hidden=64)

    def forward_sampled(self, xn, c, p_sampled):
        z_raw, info = self.base_head.forward_sampled(xn, c, p_sampled)
        feat = torch.cat([xn, c], dim=-1)
        gamma, beta = self.calibrator(feat)
        
        z_calib = gamma * z_raw + beta
        info['gamma'] = gamma
        info['beta'] = beta
        info['dynamic_thresh'] = torch.sigmoid(-beta / gamma)
        return z_calib, info

    def forward_inference(self, xn, c, p_proto):
        z_raw, gate = self.base_head.forward_inference(xn, c, p_proto)
        if len(c) == 0:
            return z_raw, gate
        feat = torch.cat([xn, c], dim=-1)
        gamma, beta = self.calibrator(feat)
        
        # [B, 1, 1] for 2D broadcasting
        z_calib = gamma.unsqueeze(-1) * z_raw + beta.unsqueeze(-1)
        return z_calib, gate
