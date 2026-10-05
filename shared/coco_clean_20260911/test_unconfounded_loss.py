import torch
import torch.nn as nn
from ultralytics.utils.loss import E2ELoss
from unconfounded_dilated_loss import DualBranchUnconfoundedLoss

def run_test():
    print("Testing DualBranchUnconfoundedLoss equivalence and behavior...")
    # Initialize a small mock or real model
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    yolo = YOLO(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911\weights\yolo26m-seg.pt")
    yolo.model.args = get_cfg()
    model = yolo.model.cuda().train()
    
    # Create a synthetic batch
    bs = 2
    nc = 80
    imgsz = torch.tensor([640, 640], device="cuda")
    dummy_imgs = torch.randn(bs, 3, 640, 640, device="cuda")
    
    # Ground truth targets: 4 instances
    # batch_idx, cls, x1, y1, x2, y2
    batch = {
        "batch_idx": torch.tensor([0, 0, 1, 1], device="cuda"),
        "cls": torch.tensor([0, 1, 0, 2], device="cuda").unsqueeze(1),
        "bboxes": torch.tensor([
            [100., 100., 200., 200.],
            [300., 300., 450., 450.],
            [50., 50., 150., 150.],
            [200., 200., 350., 350.]
        ], device="cuda"),
        "masks": torch.zeros((bs, 160, 160), device="cuda"), # overlap mask format
        "sem_masks": torch.zeros((bs, 160, 160), device="cuda"),
        "img": dummy_imgs
    }
    # Put some GT mask pixels
    batch["masks"][0, 25:50, 25:50] = 1 # instance 0
    batch["masks"][0, 75:112, 75:112] = 2 # instance 1
    batch["masks"][1, 12:37, 12:37] = 1 # instance 2
    batch["masks"][1, 50:87, 50:87] = 2 # instance 3
    
    # Forward pass
    with torch.no_grad():
        preds = model(dummy_imgs)
        
    # 1. Test official E2ELoss vs Unconfounded with alpha=0.0
    from ultralytics.utils.loss import v8SegmentationLoss
    stock_crit = E2ELoss(model, loss_fn=v8SegmentationLoss)
    unconf_crit_0 = DualBranchUnconfoundedLoss(model, alpha_train=0.0, lambda_margin=0.0)
    
    l_stock, _ = stock_crit(preds, batch)
    l_unconf_0, _ = unconf_crit_0(preds, batch)
    
    diff_0 = torch.max(torch.abs(l_stock - l_unconf_0)).item()
    print(f"Max component difference at alpha=0, lambda=0: {diff_0:.6e}")
    print(f"Stock loss components: {l_stock.tolist()}")
    print(f"Unconf loss components: {l_unconf_0.tolist()}")
    assert diff_0 < 1e-5, f"Stock equivalence failed: {diff_0}"
    print("Assertion 1 PASSED: Strict equivalence to official E2ELoss at alpha=0, lambda=0!")
    
    # 2. Test with alpha=0.1, lambda=1.0
    unconf_crit_dil = DualBranchUnconfoundedLoss(model, alpha_train=0.1, lambda_margin=1.0)
    l_unconf_dil, _ = unconf_crit_dil(preds, batch)
    print(f"Loss with margin supervision (alpha=0.1, lambda=1.0):")
    print(f"  Stock:     {l_stock.tolist()}")
    print(f"  Dilated:   {l_unconf_dil.tolist()}")
    # Compare seg loss component (index 1)
    assert l_unconf_dil[1].item() > l_stock[1].item(), "Margin loss should add positive penalty to seg loss!"
    print("Assertion 2 PASSED: Margin loss adds positive penalty to seg loss without confounding base loss!")
    
    print("\nAll Unconfounded Loss tests PASSED successfully!")

if __name__ == "__main__":
    run_test()
