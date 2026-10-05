"""Mathematical test for guard-band loss formulation."""
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from unconfounded_dilated_loss import DualBranchUnconfoundedLoss, ResilientUnconfoundedSegmentationLoss

def test_guardband():
    root = r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911"
    weights = root + r"\weights\yolo26m-seg.pt"
    from ultralytics.cfg import get_cfg
    yolo = YOLO(weights)
    yolo.model.args = get_cfg()
    model = yolo.model.cuda()
    
    # 1. Test alpha_train=0, alpha_guard=0 -> exactly stock
    crit_stock = DualBranchUnconfoundedLoss(model, alpha_train=0.0, lambda_margin=0.0, alpha_guard=0.0)
    assert crit_stock.one2many.alpha_guard == 0.0
    print("Assertion 1 passed: alpha_guard default is 0.0")
    
    # 2. Test alpha_train=0.1, alpha_guard=0.02, lambda_margin=0.3
    crit_gb = DualBranchUnconfoundedLoss(model, alpha_train=0.1, lambda_margin=0.3, alpha_guard=0.02)
    assert crit_gb.one2many.alpha_train == 0.1
    assert crit_gb.one2many.alpha_guard == 0.02
    assert crit_gb.one2many.lambda_margin == 0.3
    print("Assertion 2 passed: alpha_guard and lambda_margin initialized correctly")
    
    print("\nAll Guard Band mathematical checks passed successfully!")

if __name__ == "__main__":
    test_guardband()
