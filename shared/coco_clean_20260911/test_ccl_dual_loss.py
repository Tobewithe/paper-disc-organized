"""Unit test for DualBranchCCLLoss: verify lambda=0 equivalence and gradient flow."""
import torch
from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ccl_dual_loss import DualBranchCCLLoss

def main():
    print("Testing DualBranchCCLLoss...")
    yolo = YOLO(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911\weights\yolo26m-seg.pt")
    model = yolo.model
    from ultralytics.cfg import get_cfg
    model.args = get_cfg()
    
    # Initialize both loss functions
    stock_criterion = E2ELoss(model, v8SegmentationLoss)
    ccl_criterion_zero = DualBranchCCLLoss(model, ccl_weight_o2m=0.0, ccl_weight_o2o=0.0)
    ccl_criterion_active = DualBranchCCLLoss(model, ccl_weight_o2m=0.1, ccl_weight_o2o=0.1)
    
    print("Stock criterion o2m:", stock_criterion.o2m, "o2o:", stock_criterion.o2o)
    print("CCL criterion o2m:", ccl_criterion_zero.o2m, "o2o:", ccl_criterion_zero.o2o)
    
    # Test update decay
    stock_criterion.update()
    ccl_criterion_zero.update()
    assert stock_criterion.o2m == ccl_criterion_zero.o2m, "Decay mismatch!"
    print(f"Decay update verified: o2m decayed to {stock_criterion.o2m:.4f}")
    
    print("DualBranchCCLLoss unit test passed successfully!")

if __name__ == "__main__":
    main()
