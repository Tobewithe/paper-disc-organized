import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    script = """import os
import sys
import argparse
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# ----------------- TRUE CCL PATCH -----------------
import ultralytics.utils.loss
from ultralytics.utils.loss import v8SegmentationLoss

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
        self.ccl_weight = float(os.environ.get('CCL_WEIGHT', 0.0))
        self.ccl_margin = float(os.environ.get('CCL_MARGIN', 0.1))

    def loss(self, preds, batch):
        loss_array, loss_items = super().loss(preds, batch)
        
        # If baseline run, skip CCL entirely
        if self.ccl_weight <= 0.0001:
            return loss_array, loss_items
            
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
            if torch.rand(1).item() < 0.005:
                print(f"\\n[CCL ACTIVE] Contrastive Loss: {ccl_loss.item():.4f} for {valid_pairs} pairs.", flush=True)
            
        loss_array[1] += ccl_loss * batch_size
        loss_items[1] += ccl_loss.detach()
        return loss_array, loss_items

ultralytics.utils.loss.v8SegmentationLoss = CCLv8SegmentationLoss
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=float, default=0.0)
    parser.add_argument('--margin', type=float, default=0.1)
    args = parser.parse_args()
    
    os.environ['CCL_WEIGHT'] = str(args.weight)
    os.environ['CCL_MARGIN'] = str(args.margin)
    
    print(f"\\n{'='*50}\\nSTARTING ABLATION: W={args.weight}, M={args.margin}\\n{'='*50}\\n")
    
    model = YOLO("yolo26m-seg.pt")
    
    model.train(
        data="coco_dense.yaml",
        epochs=15,
        batch=16,
        workers=8,
        project="coco_dense_ablations",
        name=f"ccl_w{args.weight}_m{args.margin}",
        lr0=0.001,
        patience=5
    )

if __name__ == '__main__':
    main()
"""
    stdin, stdout, stderr = client.exec_command("cat > /root/autodl-tmp/tools/train_coco_ccl.py")
    stdin.write(script)
    stdin.close()
    
    # Kill and restart
    client.exec_command("pkill -f train_coco_ccl")
    client.exec_command("rm -f /root/autodl-tmp/run_cluster_fixed.log")
    client.exec_command("cd /root/autodl-tmp && nohup bash tools/run_coco_cluster.sh > run_cluster_fixed.log 2>&1 &")
    
    client.close()

if __name__ == '__main__':
    main()
