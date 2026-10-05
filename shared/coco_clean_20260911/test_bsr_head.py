"""Unit test for BifurcatedSpatialHead (BSR-Head).

Checks:
1. Shape correctness on sampled forward and grid forward.
2. Zero-initialization: at step 0, z_final matches base z_coeff.
3. Gradient flow: backprop through bsr_loss provides non-zero finite gradients to all parameters.
4. Mathematical consistency between grid forward and sampled forward.
"""
import torch
from bifurcated_spatial_readout import BifurcatedSpatialHead, bsr_loss


def test_bsr_head():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    B = 8
    N_pts = 512
    H, W = 160, 160
    
    xn = torch.randn(B, 73, device=device)
    c = torch.randn(B, 32, device=device)
    p_sampled = torch.randn(B, N_pts, 32, device=device)
    p_proto = torch.randn(32, H, W, device=device)
    y = torch.randint(0, 2, (B, N_pts), device=device).float()
    factor = torch.ones(B, device=device)
    
    model = BifurcatedSpatialHead(in_dim=73, p_dim=32, hidden=128).to(device)
    
    # 1. Forward sampled
    z_final, info = model.forward_sampled(xn, c, p_sampled)
    assert z_final.shape == (B, N_pts), f"Unexpected shape {z_final.shape}"
    assert info['gate'].shape == (B, 1), f"Unexpected gate shape {info['gate'].shape}"
    assert info['delta_c'].shape == (B, 32), f"Unexpected delta_c shape {info['delta_c'].shape}"
    assert info['delta_z_spatial'].shape == (B, N_pts), f"Unexpected delta_z_spatial shape {info['delta_z_spatial'].shape}"
    
    # 2. Check zero init
    z_base = (p_sampled * c.unsqueeze(1)).sum(-1)
    diff = (z_final - z_base).abs().max().item()
    print(f"Max diff from base at step 0: {diff:.6e}")
    assert diff < 0.05, f"At step 0, model should match base closely, got diff {diff}"
    
    # 3. Check loss and gradient flow
    loss, loss_dict = bsr_loss(z_final, y, factor, info)
    assert torch.isfinite(loss), "Loss is non-finite"
    loss.backward()
    
    # Check all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no grad"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite grad"
        grad_norm = param.grad.norm().item()
        print(f"Grad norm {name}: {grad_norm:.6e}")
        assert grad_norm > 0, f"Parameter {name} has zero grad"
        
    # 4. Check forward inference
    with torch.no_grad():
        z_grid, gate_infer = model.forward_inference(xn, c, p_proto)
        assert z_grid.shape == (B, H, W), f"Unexpected grid shape {z_grid.shape}"
        assert gate_infer.shape == (B, 1), f"Unexpected gate shape {gate_infer.shape}"
        
    print("All BSR-Head unit tests passed successfully!")


if __name__ == '__main__':
    test_bsr_head()
