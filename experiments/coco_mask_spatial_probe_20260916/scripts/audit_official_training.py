"""Small structural probes of official 8.4.100; synthetic inputs, no optimizer steps."""
import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils.loss import BboxLoss, v8SegmentationLoss
from ultralytics.utils.tal import TaskAlignedAssigner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    torch.set_num_threads(4)
    assert ultralytics.__version__ == '8.4.100'
    yolo = YOLO(args.weights)
    train_args = yolo.ckpt['train_args']
    keys = ['epochs', 'optimizer', 'lr0', 'lrf', 'warmup_epochs', 'close_mosaic',
            'mask_ratio', 'overlap_mask', 'box', 'cls', 'dfl', 'batch', 'nbs',
            'imgsz', 'mosaic', 'mixup', 'copy_paste', 'cutmix']
    report = {
        'scope': 'Official source and synthetic gradient-route probes, not COCO effect estimation',
        'optimizer_steps': 0,
        'environment': {'ultralytics': ultralytics.__version__, 'torch': torch.__version__},
        'checkpoint_train_args': {k: train_args.get(k) for k in keys},
        'loaded_yolo_overrides': {k: yolo.overrides.get(k, '<absent>') for k in keys},
    }
    package = Path(ultralytics.__file__).parent
    for rel in ['utils/loss.py', 'utils/tal.py', 'utils/metrics.py', 'nn/modules/head.py',
                'nn/modules/block.py', 'engine/trainer.py', 'engine/model.py', 'optim/muon.py',
                'data/utils.py', 'data/augment.py']:
        dest = out / 'source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((package / rel).read_bytes())

    # Use actual checkpoint arguments. Small random image/rectangle only tests graph reachability.
    device = torch.device('cuda:0')
    model = yolo.model.to(device).float().train()
    model.args = SimpleNamespace(**train_args)
    for p in model.parameters():
        p.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    head = model.model[-1]
    report['head'] = {'type': type(head).__name__, 'reg_max': head.reg_max, 'end2end': head.end2end}
    criterion = model.init_criterion()
    batch = {
        'img': torch.rand(1, 3, 128, 128, device=device),
        'batch_idx': torch.zeros(1, device=device),
        'cls': torch.zeros(1, 1, device=device),
        'bboxes': torch.tensor([[.5, .5, .5, .5]], device=device),
        'masks': torch.zeros(1, 128, 128, device=device),
        'sem_masks': torch.zeros(1, 128, 128, device=device),
    }
    batch['masks'][:, 32:96, 32:96] = 1
    preds = model(batch['img'])
    targets = {
        'backbone_first': next(model.model[0].parameters()),
        'prototype_fusion_from_P4': next(head.proto.feat_refine[0].parameters()),
        'prototype_output': next(head.proto.cv3.parameters()),
        'semantic_head': next(head.proto.semseg.parameters()),
        'o2m_box': tuple(head.cv2.parameters()),
        'o2m_class': tuple(head.cv3.parameters()),
        'o2m_coefficient': tuple(head.cv4.parameters()),
        'o2o_box': tuple(head.one2one_cv2.parameters()),
        'o2o_class': tuple(head.one2one_cv3.parameters()),
        'o2o_coefficient': tuple(head.one2one_cv4.parameters()),
    }
    targets = {k: (v,) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
    flat_targets = [p for group in targets.values() for p in group]
    vectors = {}
    routes = {}
    positives = {}
    for branch in ['one2many', 'one2one']:
        branch_loss = getattr(criterion, branch)
        assigned = branch_loss.get_assigned_targets_and_loss(preds[branch], batch)[0]
        positives[branch] = int(assigned[0].sum())
        assert positives[branch] > 0
        vector, _ = branch_loss.loss(preds[branch], batch)
        vectors[branch] = vector
        routes[branch] = {}
        for i, name in enumerate(['box', 'instance_mask', 'classification', 'l1_dfl_slot', 'semantic']):
            grads = torch.autograd.grad(vector[i], flat_targets, retain_graph=True, allow_unused=True)
            routes[branch][name] = {}
            offset = 0
            for key, group in targets.items():
                group_grads = [g for g in grads[offset:offset + len(group)] if g is not None]
                routes[branch][name][key] = (
                    float(torch.stack([g.float().square().sum() for g in group_grads]).sum().sqrt())
                    if group_grads else None)
                offset += len(group)
    assert routes['one2one']['instance_mask']['o2o_coefficient'] > 0
    assert routes['one2one']['instance_mask']['backbone_first'] is None
    combined, logged = criterion(preds, batch)
    assert torch.allclose(logged, vectors['one2one'].detach())
    assert torch.allclose(combined, .8 * vectors['one2many'] + .2 * vectors['one2one'])
    report['synthetic_positive_counts'] = positives
    report['synthetic_gradient_norms'] = routes
    report['logged_loss_is_one2one'] = True
    report['branch_weights'] = {
        'start': [criterion.o2m, criterion.o2o],
        'last_training_epoch': [criterion.decay(model.args.epochs - 1), 1 - criterion.decay(model.args.epochs - 1)],
    }
    report['synthetic_loss_vectors'] = {k: v.detach().cpu().tolist() for k, v in vectors.items()}

    # Direct mask derivative: outside the GT crop is zero for this instance loss.
    proto = torch.zeros(1, 8, 8, requires_grad=True)
    gt = torch.zeros(1, 8, 8)
    gt[:, 3:5, 3:5] = 1
    mask_loss = v8SegmentationLoss.single_mask_loss(
        gt, torch.ones(1, 1), proto, torch.tensor([[2., 2., 6., 6.]]), torch.tensor([.25]))
    gradient = torch.autograd.grad(mask_loss, proto)[0]
    support = torch.zeros_like(gradient, dtype=torch.bool)
    support[:, 2:6, 2:6] = True
    report['crop_gradient'] = {
        'outside_absolute_sum': float(gradient[~support].abs().sum()),
        'inside_absolute_sum': float(gradient[support].abs().sum()),
    }
    assert report['crop_gradient']['outside_absolute_sum'] == 0

    # reg_max=1 uses normalized L1 in the legacy dfl slot.
    box_loss = BboxLoss(reg_max=1)
    iou_loss, l1_loss = box_loss(
        torch.full((1, 1, 4), .6), torch.tensor([[[.4, .4, 1.6, 1.6]]]),
        torch.tensor([[1., 1.]]), torch.tensor([[[0., 0., 2., 2.]]]),
        torch.ones(1, 1, 1), torch.tensor(1.), torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([16., 16.]), torch.tensor([[8.]]))
    report['reg_max_1'] = {'dfl_module': box_loss.dfl_loss is not None, 'l1_slot': float(l1_loss)}
    assert abs(float(l1_loss) - .2) < 1e-6

    # Tiny-GT eligibility expands, without changing the target geometry.
    assigner = TaskAlignedAssigner(topk=10, num_classes=80, stride=[8, 16, 32])
    boxes = torch.tensor([[[30., 30., 34., 34.]]])
    eligible = assigner.select_candidates_in_gts(
        torch.tensor([[26., 32.], [32., 32.], [40., 32.]]), boxes, torch.ones(1, 1, 1))
    report['tiny_gt'] = {'gt_box': boxes.tolist(), 'eligible': eligible.tolist(), 'first_anchor_outside_original_gt': True}
    assert eligible.flatten().tolist() == [True, True, False]
    # Isolate collision resolution; not an estimate of actual collision frequency.
    idx, fg, new_mask = assigner.select_highest_overlaps(
        torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]]),
        torch.tensor([[[.5, .1], [.6, .1], [.8, .9]]]), 3,
        torch.tensor([[[.1, .01], [.2, .01], [.3, .4]]]))
    report['collision_stage_example'] = {'winner_gt_indices': idx.tolist(), 'positive_counts_per_gt': new_mask.sum(-1).tolist()}

    polygons = [np.array([[1, 1], [6, 1], [6, 6], [1, 6]], dtype=np.float32),
                np.array([[3, 3], [5, 3], [5, 5], [3, 5]], dtype=np.float32)]
    individual = polygons2masks((8, 8), polygons, 1, downsample_ratio=1)
    canvas, order = polygons2masks_overlap((8, 8), polygons, downsample_ratio=1)
    report['overlap_canvas'] = {
        'original_areas': individual.sum((1, 2)).tolist(), 'sorted_indices': order.tolist(),
        'encoded_areas': [int((canvas == i + 1).sum()) for i in range(2)],
    }
    report['limitations'] = [
        'Synthetic input only establishes implementation behavior; no COCO loss-conflict frequency or AP claim.',
        'Gradient norms are illustrative, not task importance estimates.',
        'All head scales/parameters sampled; backbone and prototype use representative tensors; source establishes detach boundary.',
        'BN statistics fixed; no optimizer, no checkpoint mutation, no training or method comparison.',
    ]
    (out / 'SUMMARY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['head', 'synthetic_positive_counts', 'logged_loss_is_one2one',
                                          'branch_weights', 'crop_gradient', 'reg_max_1', 'tiny_gt', 'overlap_canvas']}, indent=2))


if __name__ == '__main__':
    main()
