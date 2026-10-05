import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8SegmentationLoss
from unconfounded_dilated_loss import ResilientUnconfoundedSegmentationLoss

def test_math():
    print("Running mathematical verification of Unconfounded Loss...")
    yolo = YOLO(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911\weights\yolo26m-seg.pt")
    yolo.model.args = get_cfg()
    model = yolo.model.cuda().train()
    
    stock_loss_obj = v8SegmentationLoss(model)
    unconf_loss_0 = ResilientUnconfoundedSegmentationLoss(model)
    unconf_loss_0.alpha_train = 0.0
    unconf_loss_0.lambda_margin = 0.0
    
    unconf_loss_dil = ResilientUnconfoundedSegmentationLoss(model)
    unconf_loss_dil.alpha_train = 0.1
    unconf_loss_dil.lambda_margin = 1.0
    
    # Create synthetic inputs for calculate_segmentation_loss
    bs = 2
    n_anchors = 10
    mask_h, mask_w = 160, 160
    
    # 2 positive anchors per image
    fg_mask = torch.zeros(bs, n_anchors, dtype=torch.bool, device="cuda")
    fg_mask[:, :2] = True
    
    target_gt_idx = torch.zeros(bs, n_anchors, dtype=torch.long, device="cuda")
    target_gt_idx[0, 1] = 1
    target_gt_idx[1, 1] = 1
    
    # Target boxes: xyxy on 640x640
    target_bboxes = torch.zeros(bs, n_anchors, 4, device="cuda")
    target_bboxes[0, 0] = torch.tensor([100., 100., 200., 200.])
    target_bboxes[0, 1] = torch.tensor([300., 300., 450., 450.])
    target_bboxes[1, 0] = torch.tensor([50., 50., 150., 150.])
    target_bboxes[1, 1] = torch.tensor([200., 200., 350., 350.])
    
    batch_idx = torch.tensor([0, 0, 1, 1], device="cuda").unsqueeze(1)
    
    # GT masks: overlap format (bs, 160, 160)
    masks = torch.zeros(bs, 160, 160, device="cuda")
    masks[0, 25:50, 25:50] = 1
    masks[0, 75:112, 75:112] = 2
    masks[1, 12:37, 12:37] = 1
    masks[1, 50:87, 50:87] = 2
    
    proto = torch.randn(bs, 32, mask_h, mask_w, device="cuda", requires_grad=True)
    pred_masks = torch.randn(bs, n_anchors, 32, device="cuda", requires_grad=True)
    imgsz = torch.tensor([640., 640.], device="cuda")
    
    # 1. Stock loss
    l_stock = stock_loss_obj.calculate_segmentation_loss(
        fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
    )
    
    # 2. Unconfounded loss at alpha=0, lambda=0
    l_unconf_0 = unconf_loss_0.calculate_segmentation_loss(
        fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
    )
    
    diff_0 = torch.abs(l_stock - l_unconf_0).item()
    print(f"Stock vs Unconf (alpha=0, lambda=0): stock={l_stock.item():.6f}, unconf={l_unconf_0.item():.6f}, diff={diff_0:.6e}")
    assert diff_0 < 1e-6, f"Mismatch at alpha=0: diff={diff_0}"
    print("Assertion 1 PASSED: Strict numerical equivalence when alpha=0!")
    
    # 3. Unconfounded loss at alpha=0.1, lambda=1.0
    l_unconf_dil = unconf_loss_dil.calculate_segmentation_loss(
        fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
    )
    margin_penalty = (l_unconf_dil - l_stock).item()
    print(f"Unconf with margin (alpha=0.1, lambda=1.0): l_dil={l_unconf_dil.item():.6f}, margin_penalty={margin_penalty:.6f}")
    assert margin_penalty > 0, "Margin penalty should be positive!"
    print("Assertion 2 PASSED: Margin penalty is strictly positive and non-zero!")
    
    # 4. Backward check
    l_unconf_dil.backward()
    assert proto.grad is not None and torch.isfinite(proto.grad).all(), "Proto grad invalid!"
    assert pred_masks.grad is not None and torch.isfinite(pred_masks.grad).all(), "Pred_masks grad invalid!"
    print("Assertion 3 PASSED: Gradients flow cleanly to prototypes and coefficients!")
    print("\nALL MATHEMATICAL ASSERTIONS VERIFIED!")

if __name__ == "__main__":
    test_math()
