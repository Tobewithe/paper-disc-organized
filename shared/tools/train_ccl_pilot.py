import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model

def box_iou_tensor(box1, box2):
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter_area
    return inter_area / (union + 1e-6)

def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

class CCLv8SegmentationLoss(v8SegmentationLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ccl_weight = 0.5
        self.ccl_margin = 0.1

    def loss(self, preds, batch):
        loss_array, loss_items = super().loss(preds, batch)
        
        pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        (fg_mask, target_gt_idx, _, _, _), _, _ = self.get_assigned_targets_and_loss(preds, batch)
        
        batch_size = pred_masks.shape[0]
        ccl_loss = torch.tensor(0.0, device=self.device)
        valid_pairs = 0
        
        for b in range(batch_size):
            batch_idx = batch["batch_idx"].view(-1)
            b_mask = batch_idx == b
            if not b_mask.any(): continue
            
            gt_boxes_xywh = batch["bboxes"][b_mask]
            gt_boxes_xyxy = xywh2xyxy(gt_boxes_xywh)
            ious = box_iou_tensor(gt_boxes_xyxy, gt_boxes_xyxy)
            
            fg = fg_mask[b]
            if not fg.any(): continue
            
            p_masks = pred_masks[b][fg]
            t_gt_idx = target_gt_idx[b][fg]
            
            unique_gts = t_gt_idx.unique()
            if len(unique_gts) < 2: continue
            
            gt_avg_coeffs = {}
            for gt_id in unique_gts:
                mask_for_gt = (t_gt_idx == gt_id)
                gt_avg_coeffs[gt_id.item()] = p_masks[mask_for_gt].mean(dim=0)
                
            for i in range(len(unique_gts)):
                for j in range(i+1, len(unique_gts)):
                    id1 = unique_gts[i].item()
                    id2 = unique_gts[j].item()
                    if id1 < len(ious) and id2 < len(ious) and ious[id1, id2] > 0.05:
                        c1 = gt_avg_coeffs[id1]
                        c2 = gt_avg_coeffs[id2]
                        sim = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0)).squeeze()
                        penalty = F.relu(sim - self.ccl_margin)
                        ccl_loss += penalty
                        valid_pairs += 1
                        
        if valid_pairs > 0:
            ccl_loss = (ccl_loss / valid_pairs) * self.ccl_weight
            
        loss_array[1] += ccl_loss * batch_size
        loss_items[1] += ccl_loss.detach()
        return loss_array, loss_items

import ultralytics.models.yolo.segment.train as seg_train
seg_train.v8SegmentationLoss = CCLv8SegmentationLoss
print("Successfully monkey-patched v8SegmentationLoss with CCLv8SegmentationLoss!")

def main():
    model = load_model(Path('C:/Dpan/document/model_datasets/artifacts/pigcv-task05/remote-runs/task05-train-yolo-full-20260810-02/yolo26_task05_final.pt'), 'cuda')
    
    data_yaml = 'C:/Users/Public/pigcv-task05/external-inference/faropigseg_v1/test/faro_test_pilot.yaml'
    Path(data_yaml).write_text(
        "path: C:/Dpan/document/model_datasets/datasets/FaroPigSeg\n"
        "train: test/images\n"
        "val: test/images\n"
        "names:\n"
        "  0: pig\n"
    )
    
    model.train(
        data=data_yaml,
        epochs=1,
        imgsz=640,
        batch=4,
        device='cuda',
        project='experiments/pilot_ccl_run',
        name='run',
        workers=0
    )

if __name__ == '__main__':
    main()
