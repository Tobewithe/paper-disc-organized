"""Frozen model inference, predicted-body bank and GT-free ROI alignment."""
import contextlib
import io
from pathlib import Path
import sys

import cv2
import numpy as np
from pycocotools import mask as cm
import torch
from torch.nn import functional as F

from model import crop_tensor
from rt import dump, read, sha, state_hash


def external(cfg):
    for folder in (Path(cfg['parent_root'])/'scripts', Path(cfg['object_root'])/'scripts'):
        if str(folder) not in sys.path: sys.path.insert(0, str(folder))


def load_native(cfg):
    external(cfg)
    import ultralytics
    from task_model import load
    from ultralytics.utils import ops
    assert ultralytics.__version__ == '8.4.100'
    assert torch.__version__ == '2.5.1+cu124'
    model = load(Path(cfg['parent_root'])/cfg['native_checkpoint']).eval().requires_grad_(False)
    head = model.model[-1]
    assert head.end2end and head.nc == 1 and head.nm == 32 and head.max_det == 300
    assert not head.agnostic_nms
    captured = {}
    hooks = [model.model[i].register_forward_hook(
        lambda layer, inputs, output, index=i: captured.__setitem__(index, output)) for i in (2, 16)]
    return model, captured, hooks, ops


def encode(mask):
    value = cm.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {'size': value['size'], 'counts': value['counts'].decode('ascii')}


def prediction(image_id, d, mask):
    x0, y0, x1, y1 = [float(x) for x in d[:4]]
    return dict(image_id=image_id, category_id=1, score=float(d[4]), bbox=[x0, y0, x1-x0, y1-y0],
                segmentation=encode(mask))


def unit_projection(rows, columns, seed, device='cpu'):
    matrix = torch.randn(rows, columns, generator=torch.Generator().manual_seed(seed))
    return F.normalize(matrix, dim=1).to(device)


def fit_map(array, size):
    yy, xx = np.where(array[:, :, 0] > .25)
    assert len(xx) >= 4, 'Degenerate predicted body projection'
    cropped = array[yy.min():yy.max()+1, xx.min():xx.max()+1]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)


@torch.no_grad()
def make_bank(root, run, cfg):
    external(cfg)
    from object_model import ObjectModel
    from geometry import raster
    obj = Path(cfg['object_root'])
    folder = Path(read(obj/'DATA_LOCATION.json')['directory'])
    manifest = read(folder/'MANIFEST.json')
    template = dict(np.load(obj/'artifacts/template_inference.npz'))
    ocfg = read(obj/'CONFIG_V0.json')
    payload = torch.load(obj/'artifacts/structured_last.pt', map_location='cpu', weights_only=False)
    assert payload['optimizer_steps'] == 4000 and payload['arm'] == 'structured'
    model = ObjectModel(template, ocfg, structured=True).cuda().eval().requires_grad_(False)
    model.load_state_dict(payload['model'], strict=True)
    before = state_hash(model)
    feature_projection = unit_projection(8, 64, cfg['seed']+101, 'cuda')
    selected = [r for r in manifest['samples'] if r['split'] == 'train' and r['appearance'] == 0 and r['occluded'] == 0]
    assert len(selected) == cfg['template_sources'] == 432
    maps, identities = [], []
    faces = torch.as_tensor(template['faces'], device='cuda', dtype=torch.long)
    for i, row in enumerate(selected):
        assert sha(folder/row['file']) == row['sha256']
        rgb = torch.from_numpy(np.load(folder/row['file'])['rgb'].copy()).cuda().permute(2, 0, 1)[None].float()/255
        camera = torch.tensor(row['camera'], device='cuda', dtype=torch.float32)
        pred = model(rgb, camera[None])
        features = F.normalize(pred['features'][0]@feature_projection.T, dim=-1)
        rendered = raster(pred['vertices'][0], faces, camera, row['focal'], 128,
                          {'canonical': model.canonical*2, 'feature': features})
        mask = rendered['mask'].float()
        array = torch.cat([mask[:, :, None], rendered['attributes']['canonical']*mask[:, :, None],
                           rendered['attributes']['feature']*mask[:, :, None]], -1).cpu().numpy()
        base = fit_map(array, 32)
        for rotation in range(cfg['template_rotations']):
            padded = np.zeros((64, 64, 12), np.float32); padded[16:48, 16:48] = base
            transform = cv2.getRotationMatrix2D((31.5, 31.5), rotation*45., 1.)
            rotated = cv2.warpAffine(padded, transform, (64, 64), flags=cv2.INTER_LINEAR)
            rotated = fit_map(rotated, 32)
            for reflection in range(cfg['template_reflections']):
                value = rotated[:, ::-1].copy() if reflection else rotated.copy()
                maps.append(value.transpose(2, 0, 1))
                identities.append(dict(sample_id=row['sample_id'], family=row['family'], pose=row['pose'],
                                       view=row['view'], rotation_degrees=rotation*45, reflection=bool(reflection)))
        if (i+1) % 72 == 0: print('BODY_BANK', i+1, len(selected), flush=True)
    maps = torch.from_numpy(np.stack(maps)).half()
    coarse = F.interpolate(maps[:, :1].float(), (16, 16), mode='area').flatten(1)
    pixel_permutation = torch.randperm(32*32, generator=torch.Generator().manual_seed(cfg['seed']+301))
    bank = dict(maps=maps, coarse=coarse, area=coarse.sum(1), pixel_permutation=pixel_permutation,
                yolo_projection=unit_projection(8, 512, cfg['seed']+201),
                feature_projection=feature_projection.cpu(), identities=identities,
                source_model_sha256=sha(obj/'artifacts/structured_last.pt'), source_split='synthetic_train_only',
                camera_role='known source synthetic cameras; no claimed real calibration')
    assert state_hash(model) == before
    path = run/'object_bank.pt'
    assert not path.exists()
    torch.save(bank, path)
    receipt = dict(templates=len(maps), source_observations=len(selected), channels=12, roi_size=32,
                   source_split='train', bank_sha256=sha(path), source_model_sha256=bank['source_model_sha256'],
                   object_state_hash=before, source_sample_ids=[r['sample_id'] for r in selected],
                   source_targets_used_for_projection=False, file=str(path))
    dump(run/'BANK_MANIFEST.json', receipt); dump(root/'BANK_POINTER.json', receipt)
    del model
    return bank


def gpu_bank(bank):
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in bank.items()}


@torch.no_grad()
def retrieve(native_crop, bank):
    occupancy = F.interpolate((native_crop > 0).float()[:, None], (16, 16), mode='area').flatten(1)
    overlap = occupancy@bank['coarse'].T
    score = overlap/(occupancy.sum(1, keepdim=True)+bank['area'][None]-overlap).clamp_min(1e-6)
    value, index = score.max(1)
    return index, value


@torch.no_grad()
def native_frame(model, captured, ops, image, bank, cfg):
    captured.clear()
    output, proto = model(image)[0]
    proto = proto[0] if isinstance(proto, tuple) else proto
    all_det = output[0]
    head_ids = torch.arange(len(all_det), device='cuda')[all_det[:, 4] > cfg['native_confidence']]
    det = all_det[head_ids].clone(); det[:, :4].clamp_(0, 640)
    logits = (det[:, 6:]@proto[0].float().flatten(1)).reshape(-1, 160, 160)
    masks = ops.process_mask(proto[0], det[:, 6:], det[:, :4], (640, 640), upsample=True)
    keep = masks.flatten(1).any(1).bool()
    det, logits, masks, head_ids = det[keep], logits[keep], masks[keep], head_ids[keep]
    wh = det[:, 2:4]-det[:, :2]
    eligible = torch.nonzero((det[:, 4] >= cfg['application_confidence']) &
                             (wh.min(1).values >= cfg['application_min_box_side']))[:, 0]
    boxes = det[eligible, :4]
    n = len(eligible)
    if n:
        # Each detection uses its own raw logit map, without GT-driven selection.
        logit_crop = F.grid_sample(logits[eligible, None],
                                  __import__('model').roi_grid(boxes, 32), align_corners=False)[:, 0]
        features = torch.cat([captured[2], F.interpolate(captured[16], (160, 160), mode='bilinear', align_corners=False)], 1)
        projected = torch.einsum('kc,bchw->bkhw', bank['yolo_projection'], features)
        projected = F.normalize(projected, dim=1)
        rgb_roi, features_roi = crop_tensor(image, boxes, 32), crop_tensor(projected, boxes, 32)
        t = (torch.arange(32, device='cuda')+.5)/16-1
        yy, xx = torch.meshgrid(t, t, indexing='ij')
        coords = torch.stack([xx, yy])[None].expand(n, -1, -1, -1)
        common = torch.cat([rgb_roi, logit_crop[:, None].clamp(-10, 10)/5, features_roi, coords], 1)
        index, alignment = retrieve(logit_crop, bank)
    else:
        common = image.new_zeros((0, 14, 32, 32)); logit_crop = image.new_zeros((0, 32, 32))
        index = torch.zeros(0, dtype=torch.long, device='cuda'); alignment = image.new_zeros(0)
    return dict(det=det, logits=logits, masks=masks.bool(), head_ids=head_ids, eligible=eligible,
                common=common.half(), native_crop=logit_crop, bank_index=index, alignment=alignment)


def verify_real(cfg, run):
    parent = Path(cfg['parent_root'])
    rows = [r for r in read(parent/'ALL_DATASET.json')['rows'] if r['domain'] == 'real' and r['split'] in ('train', 'dev')]
    counts = {split: sum(r['split'] == split for r in rows) for split in ('train', 'dev')}
    assert counts == {'train': 1272, 'dev': 432}, counts
    groups = {split: {r['group'] for r in rows if r['split'] == split} for split in counts}
    assert not groups['train'] & groups['dev']
    for row in rows:
        assert sha(row['image']) == row['image_sha256'], row['image']
        assert sha(row['label']) == row['label_sha256'], row['label']
    dump(run/'REAL_INPUT_CHECK.json', dict(counts=counts, image_and_label_files_verified=2*len(rows),
                manifest_sha256=sha(parent/'ALL_DATASET.json'), gt_sha256=sha(parent/'DEV_COCO.json'),
                train_dev_group_disjoint=True, real_test_used=False))
    return rows


def training_targets(frame, row, cfg):
    boxes = frame['det'][frame['eligible'], :4]
    gt = torch.tensor(row['bboxes'], device='cuda', dtype=torch.float32)*640
    xy0, xy1 = gt[:, :2]-gt[:, 2:]/2, gt[:, :2]+gt[:, 2:]/2
    a = boxes[:, 2:]-boxes[:, :2]
    inter = (torch.minimum(boxes[:, None, 2:], xy1)-torch.maximum(boxes[:, None, :2], xy0)).clamp_min(0).prod(2)
    ious = inter/(a.prod(1)[:, None]+gt[:, 2:].prod(1)[None]-inter).clamp_min(1e-8)
    best, ids = ious.max(1)
    positive = best >= cfg['training_box_gt_join_iou']
    target = boxes.new_zeros((len(boxes), 32, 32))
    if positive.any():
        indices = torch.nonzero(positive)[:, 0]
        masks = torch.from_numpy(np.stack([cm.decode(row['rles'][int(ids[j])]) for j in indices])).cuda().float()
        from model import roi_grid
        target[indices] = F.grid_sample(masks[:, None], roi_grid(boxes[indices], 32), align_corners=False)[:, 0]
    return target, positive, ids, best


@torch.no_grad()
def prepare_cache(root, run, cfg, model, captured, ops, bank, rows, smoke):
    cache = root/'cache'/run.name; cache.mkdir(parents=True, exist_ok=False)
    train, dev = [], []
    native_predictions, identities, source_rows = [], [], []
    actual_rows = ([r for r in rows if r['split'] == 'train'][:8]+[r for r in rows if r['split'] == 'dev'][:2]) if smoke else rows
    zero_replay = 0
    for i, row in enumerate(actual_rows):
        rgb = cv2.imread(row['image']); assert rgb is not None
        image = torch.from_numpy(rgb[:, :, ::-1].transpose(2, 0, 1).copy()).cuda()[None].float()/255
        frame = native_frame(model, captured, ops, image, bank, cfg)
        det, eligible = frame['det'], frame['eligible']
        if len(det):
            replay = F.interpolate(ops.crop_mask(frame['logits'], det[:, :4]*.25)[:, None],
                                   (640, 640), mode='bilinear', align_corners=False)[:, 0] > 0
            assert torch.equal(replay, frame['masks']), 'Native decoder replay mismatch'
            zero_replay += len(det)
        if row['split'] == 'train':
            if len(eligible):
                target, positive, gt_ids, join = training_targets(frame, row, cfg)
                item = dict(common=frame['common'].cpu(), native=frame['native_crop'].cpu(), target=target.half().cpu(),
                            bank_index=frame['bank_index'].cpu(), positive=positive.cpu())
                file = cache/f'train_{len(train):04d}.pt'; torch.save(item, file)
                train.append(dict(file=str(file), image_id=row['image_id'], source_id=row['source_id'], group=row['group'],
                                  rows=len(eligible), positives=int(positive.sum()),
                                  native_output_ids=frame['head_ids'][eligible].cpu().tolist(),
                                  gt_local_ids=gt_ids.cpu().tolist(), box_gt_iou=join.cpu().tolist(), sha256=sha(file)))
        else:
            file = cache/f'dev_{len(dev):04d}.pt'
            item = dict(common=frame['common'].cpu(), native=frame['native_crop'].cpu(),
                        bank_index=frame['bank_index'].cpu(), eligible=eligible.cpu(),
                        base_logits=frame['logits'][eligible].cpu(), det=det[:, :6].cpu(),
                        head_ids=frame['head_ids'].cpu(), alignment=frame['alignment'].cpu(), row=row)
            torch.save(item, file)
            frame_preds = [prediction(row['image_id'], d, m) for d, m in zip(det.cpu().numpy(), frame['masks'].cpu().numpy())]
            native_predictions += frame_preds
            identities += [dict(image_id=row['image_id'], output_detection_id=int(head), detection_ordinal=k,
                                checkpoint_sha256=read(root/'INVENTORY.json')['external_files'][str(Path(cfg['parent_root'])/cfg['native_checkpoint'])])
                           for k, head in enumerate(frame['head_ids'].cpu().tolist())]
            dev.append(dict(file=str(file), image_id=row['image_id'], source_id=row['source_id'], group=row['group'],
                            rows=len(det), eligible_rows=len(eligible), sha256=sha(file)))
            source_rows.append(row)
        if (i+1) % 100 == 0: print('NATIVE_CACHE', i+1, len(actual_rows), flush=True)
    result = dict(train=train, dev=dev, zero_replay_rows=zero_replay, cache=str(cache),
                  train_source_images=sum(r['split'] == 'train' for r in actual_rows),
                  train_images_with_eligible_rows=len(train), dev_images=len(dev),
                  training_objects=sum(r['rows'] for r in train),
                  training_positive_objects=sum(r['positives'] for r in train),
                  eligible_dev_objects=sum(r['eligible_rows'] for r in dev))
    dump(run/'CACHE_MANIFEST.json', result)
    dump(run/'native_predictions.json', native_predictions); dump(run/'OUTPUT_IDENTITIES.json', identities)
    return result, native_predictions


def coco_score(gt_file, predictions, kind='segm'):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_file))
        source = [{k: v for k, v in p.items() if k != ('bbox' if kind == 'segm' else 'segmentation')} for p in predictions]
        dt = gt.loadRes(source)
        ev = COCOeval(gt, dt, kind); ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return dict(mask_ap=float(ev.stats[0])*100, mask_ap50=float(ev.stats[1])*100,
                mask_ap75=float(ev.stats[2])*100, mask_small_ap=float(ev.stats[3])*100,
                stats=[float(v)*100 for v in ev.stats]), ev
