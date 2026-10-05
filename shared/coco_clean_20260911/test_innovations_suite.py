"""Unit tests for Innovations 2, 3, and 4."""
import torch
from ada_calib_readout import AdaCalibHead, InstanceCalibrator
from contrastive_spatial_loss import prototype_orthogonality_loss, pairwise_contrastive_loss, apply_centerness_prior
from bifurcated_spatial_readout import BifurcatedSpatialHead


def test_innovations():
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
    boxes = torch.tensor([[10, 10, 50, 50]] * B, device=device).float()
    
    # Test 1: AdaCalibHead
    model = AdaCalibHead(in_dim=73, p_dim=32).to(device)
    z_calib, info = model.forward_sampled(xn, c, p_sampled)
    assert z_calib.shape == (B, N_pts)
    assert 'gamma' in info and 'beta' in info and 'dynamic_thresh' in info
    print(f"AdaCalib initial gamma mean: {info['gamma'].mean().item():.3f}, beta mean: {info['beta'].mean().item():.3f}")
    assert (info['gamma'] - 1.0).abs().max() < 1e-4, "Initial gamma should be 1.0"
    assert info['beta'].abs().max() < 1e-4, "Initial beta should be 0.0"
    
    # Test 2: Orthogonality loss
    ortho_loss = prototype_orthogonality_loss(p_sampled)
    assert ortho_loss > 0 and torch.isfinite(ortho_loss)
    print(f"Prototype orthogonality loss: {ortho_loss.item():.4f}")
    
    # Test 3: Pairwise contrastive loss
    image_ids = [101, 101, 102, 102, 103, 103, 104, 104]
    contrast_loss = pairwise_contrastive_loss(z_calib, y, image_ids, margin=1.5)
    assert torch.isfinite(contrast_loss)
    print(f"Pairwise contrastive loss: {contrast_loss.item():.4f}")
    
    # Test 4: Centerness prior
    z_grid, _ = model.forward_inference(xn, c, p_proto)
    z_attenuated = apply_centerness_prior(z_grid, boxes, (H, W))
    assert z_attenuated.shape == (B, H, W)
    print("Centerness prior test passed!")
    
    # Test 5: Backprop combined loss
    total_loss = z_calib.mean() + 0.01 * ortho_loss + 0.1 * contrast_loss
    total_loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
    print("All combined backward gradients verified successfully!")


if __name__ == '__main__':
    test_innovations()
