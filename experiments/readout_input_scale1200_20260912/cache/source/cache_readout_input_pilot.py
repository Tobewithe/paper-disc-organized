"""Build S023 real-COCO/native640 cache with pinned Ultralytics source.

Fresh selected annotations go through official convert_coco and YOLODataset.
No custom polygon rasterizer or synthetic target supplies training labels.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import contextlib
import io
import json
import shutil
import time
import zipfile
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import ultralytics
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import LetterBox
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.converter import convert_coco, coco91_to_coco80_class, merge_multi_segment
from ultralytics.utils import ops, nms

from frozen_mechanism_probe import ownership
from native_label_pipeline_probe import format_native, mask_of
from structure_candidate_trace import TraceCapture, save_csv
from readout_input_probe import sample_regions, sha, stable_seed, write_json

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def native_source_ids(gt, iid, label_path, label):
    """Audit official conversion text and subsequent parser dedup/order by ID."""
    im = gt.imgs[iid]
    classes = coco91_to_coco80_class()
    ids, expected_lines, seen = [], [], set()
    for a in gt.imgToAnns[iid]:
        if a.get('iscrowd', 0):
            continue
        box = np.asarray(a['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2
        box[[0, 2]] /= im['width']
        box[[1, 3]] /= im['height']
        if min(box[2:]) <= 0:
            continue
        cls = classes[a['category_id'] - 1]
        key = (cls, *box.tolist())
        if key in seen:
            continue
        seen.add(key)
        seg = a['segmentation']
        require(isinstance(seg, list) and len(seg) > 0 and
                all(isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0 for p in seg),
                f'Unsupported ordinary polygon: {a["id"]}; no box-label fallback')
        points = np.concatenate(merge_multi_segment(seg)) if len(seg) > 1 else np.asarray(seg[0]).reshape(-1, 2)
        points = (points / np.asarray([im['width'], im['height']])).reshape(-1).tolist()
        line = [cls, *points]
        expected_lines.append((('%g ' * len(line)).rstrip() % tuple(line)))
        ids.append(a['id'])
    lines = label_path.read_text().splitlines() if label_path.exists() else []
    require(lines == expected_lines, f'Official conversion/ID text mismatch {iid}')
    segs = [np.asarray(line.split()[1:], dtype=np.float32).reshape(-1, 2) for line in lines]
    cls = np.asarray([line.split()[0] for line in lines], dtype=np.float32).reshape(-1, 1)
    boxes = ops.segments2boxes(segs) if lines else np.zeros((0, 4), np.float32)
    parsed = np.concatenate([cls, boxes], 1)
    _, keep = np.unique(parsed, axis=0, return_index=True)
    if len(keep) == len(lines):
        keep = np.arange(len(lines))
    np.testing.assert_array_equal(label['cls'], cls[keep])
    np.testing.assert_array_equal(label['bboxes'], boxes[keep])
    for actual, expected in zip(label['segments'], [segs[k] for k in keep]):
        np.testing.assert_array_equal(actual, expected)
    require(len(label['segments']) == len(keep), 'Parser segment count mismatch')
    return [ids[k] for k in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--data-root', type=Path, default=ROOT / 'data')
    ap.add_argument('--protocol', type=Path, default=ROOT / 'readout_input_pilot_protocol.json')
    ap.add_argument('--selection', type=Path)
    ap.add_argument('--train-zip', type=Path, default=Path('/autodl-pub/data/COCO2017/train2017.zip'))
    args = ap.parse_args()
    require(ultralytics.__version__ == '8.4.143', 'Pinned Ultralytics8.4.143 source required')
    require(torch.cuda.is_available(), 'CUDA required for real cache')
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config_path = args.protocol.resolve()
    config = json.loads(config_path.read_text())
    weight = ROOT / 'weights/yolo26m-seg.pt'
    require(sha(weight) == '16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5', 'Weight changed')
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ['images', 'conversion_input', 'source']:
        (args.out / name).mkdir()
    for name in ['readout_input_probe.py', 'cache_readout_input_pilot.py',
                 'native_label_pipeline_probe.py', 'structure_candidate_trace.py', 'frozen_mechanism_probe.py']:
        shutil.copy2(ROOT / name, args.out / 'source' / name)
    shutil.copy2(config_path, args.out / 'source' / config_path.name)
    annpath = args.data_root / 'annotations/instances_train2017.json'
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annpath))
    ranked = sorted(gt.imgs, key=lambda iid: stable_seed('readout-input:20260912', iid))
    nfit, ntest = (2, 2) if args.smoke else (config['train_images'], config['transfer_images'])
    fit_ids, transfer_ids = ranked[:nfit], ranked[nfit:nfit + ntest]
    if args.selection:
        fixed=json.loads(args.selection.read_text())
        fit_ids,transfer_ids=fixed['fit'],fixed['transfer']
        require(len(fit_ids)==nfit and len(transfer_ids)==ntest,'Prespecified split sizes differ')
    chosen = fit_ids + transfer_ids
    require(len(chosen) == len(set(chosen)), 'Image split overlap')
    write_json(args.out / 'selection.json', dict(fit=fit_ids, transfer=transfer_ids,
               full_pool=len(gt.imgs), selection='Raw train2017 image hash, no density/model selection', smoke=args.smoke))
    write_json(args.out / 'protocol.json', dict(config=config, annotation_sha256=sha(annpath),
               selection_sha256=sha(args.out / 'selection.json'), weight_sha256=sha(weight),
               source_hashes={p.name: sha(p) for p in (args.out / 'source').iterdir()},
               execution_state='CACHE_BUILDING', assignment='Frozen bbox50, not historical TAL',
               data_root=str(args.data_root.resolve()),
               runtime=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,opencv=cv2.__version__,
                            numpy=np.__version__,gpu=torch.cuda.get_device_name(),source=str(ultralytics.__file__))))
    missing = {f'train2017/{gt.imgs[i]["file_name"]}' for i in chosen
               if not (args.data_root / 'images/train2017' / gt.imgs[i]['file_name']).exists()}
    if missing:
        with zipfile.ZipFile(args.train_zip) as archive:
            for member in sorted(missing):
                dest = args.data_root / 'images' / member
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, dest.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                require(dest.stat().st_size == archive.getinfo(member).file_size, 'Incomplete JPEG extraction')
    subset = dict(info=gt.dataset.get('info', {}), images=[gt.imgs[i] for i in chosen],
                  categories=list(gt.cats.values()), annotations=[a for i in chosen for a in gt.imgToAnns[i]])
    write_json(args.out / 'conversion_input/instances_probe.json', subset)
    with contextlib.redirect_stdout(io.StringIO()):
        convert_coco(str(args.out / 'conversion_input'), str(args.out / 'converted'), use_segments=True)
    image_dir = args.out / 'converted/images/probe'
    image_dir.mkdir()
    for iid in chosen:
        image = args.data_root / 'images/train2017' / gt.imgs[iid]['file_name']
        os.link(image, image_dir / image.name)
    cfg = get_cfg(overrides=dict(task='segment', imgsz=640, mask_ratio=1, overlap_mask=True,
                                rect=False, cache=False, workers=0, fraction=1.0))
    data = dict(names={k: gt.cats[c]['name'] for k, c in enumerate(sorted(gt.cats))}, nc=80, channels=3)
    with contextlib.redirect_stdout(io.StringIO()):
        ds = build_yolo_dataset(cfg, str(image_dir), 1, data, mode='val', rect=False)
    index = {int(Path(l['im_file']).stem): k for k, l in enumerate(ds.labels)}
    require(set(index) == set(chosen), 'Native loader dropped images')
    model = YOLO(str(weight))
    model.model.eval().requires_grad_(False)
    head = model.model.model[-1]
    require(not head.end2end, 'Expected verified one-to-many branch')
    maps, outputs, weights, hooks = {}, {}, {}, []
    for level, branch in enumerate(head.cv4):
        def hook(module, inp, out, level=level):
            maps[level] = inp[0].detach().clone()
            outputs[level] = out.detach().clone()
            weights[level] = (module.weight.detach().clone(), module.bias.detach().clone())
        hooks.append(branch[-1].register_forward_hook(hook))
    rows, witness = [], []
    start = time.monotonic()
    for number, iid in enumerate(chosen, 1):
        split = 'fit' if iid in fit_ids else 'transfer'
        lab = ds.labels[index[iid]]
        label_path = args.out / 'converted/labels/probe' / f'{iid:012d}.txt'
        ids = native_source_ids(gt, iid, label_path, lab)
        common = deepcopy(lab)
        common.pop('shape', None)
        common['img'] = cv2.imread(lab['im_file'])
        require(common['img'] is not None, 'Unreadable JPEG')
        hh, ww = common['img'].shape[:2]
        common = ds.update_labels_info(common)
        common['ori_shape'], common['ratio_pad'] = (hh, ww), (1., 1.)
        common = LetterBox(new_shape=(640, 640), auto=False, scaleup=True)(common)
        native = format_native(common, ids, 1, True)
        image = args.data_root / 'images/train2017' / gt.imgs[iid]['file_name']
        with torch.inference_mode():
            model.predict(str(image), predictor=TraceCapture, imgsz=640, rect=False, conf=.001,
                          iou=.7, max_det=300, half=False, retina_masks=False, device=0, verbose=False)
            cap, raw, pre = model.predictor.capture, model.predictor.dense, model.predictor.before_empty
            require(raw.shape == (1, 116, 8400), 'Unexpected dense output')
            allnms, idx = nms.non_max_suppression(raw.clone(), conf_thres=.001, iou_thres=.7,
                    nc=80, max_det=8400, return_idxs=True, end2end=False)
            require(torch.equal(allnms[0][:300], pre), 'NMS source replay failed')
            pre_mask = ops.process_mask(cap['proto'], pre[:, 6:], pre[:, :4], (640, 640), upsample=True)
            keep = idx[0].flatten()[:300][pre_mask.flatten(1).any(1).bool()]
            require(torch.equal(raw[0, 84:, keep].T, cap['coeff']), 'Candidate identity replay failed')
            hs = torch.cat([maps[k][0].flatten(1).T for k in range(3)])
            cc = torch.cat([outputs[k][0].flatten(1).T for k in range(3)])
            lv = torch.cat([torch.full((outputs[k].shape[-2] * outputs[k].shape[-1],), k,
                                       device='cuda', dtype=torch.long) for k in range(3)])
            h, levels = hs[keep], lv[keep]
            require(torch.equal(cc[keep], cap['coeff']), 'Hook source coeff mismatch')
            replay_error = 0.
            for k in range(3):
                w, b = weights[k]
                local = (levels == k).nonzero().flatten()
                if len(local):
                    replay_error = max(replay_error, float((F.linear(h[local], w[:, :, 0, 0], b) - cap['coeff'][local]).abs().max()))
            require(replay_error < 2e-4, 'Last-layer replay failed')
            region = sample_regions([maps[k] for k in range(3)], levels, cap['boxes'], (640, 640))
            p = cap['proto'].float()
            p640 = F.interpolate(p[None], (640, 640), mode='bilinear', align_corners=False)[0]
            mapping = ownership(gt, iid, cap['detections'])
        arrays = dict(h=h.cpu().numpy(), region=region.cpu().numpy(), level=levels.cpu().numpy(),
                      source_index=keep.cpu().numpy(), coeff=cap['coeff'].cpu().numpy(),
                      boxes=cap['boxes'].cpu().numpy(), detections=cap['detections'].cpu().numpy(),
                      proto=p.cpu().numpy(), shape=np.asarray(cap['shape']), input_shape=np.asarray(cap['input_shape']))
        tids, pidx, pp, yy, factors, positions = [], [], [], [], [], []
        sample_error = 0.
        for ann in gt.imgToAnns[iid]:
            if ann.get('iscrowd', 0):
                continue
            aid = ann['id']
            state = dict(image_id=iid, split=split, annotation_id=aid, parts=len(ann['segmentation'])
                         if isinstance(ann['segmentation'], list) else -1)
            if aid not in native['position']:
                rows.append(dict(**state, status='official_label_removed'))
                continue
            if aid not in mapping:
                rows.append(dict(**state, status='bbox50_unmatched'))
                continue
            j = mapping[aid]
            norm = torch.as_tensor(native['boxes'][native['position'][aid]:native['position'][aid] + 1], device='cuda')
            box = ops.xywh2xyxy(norm) * 640
            area = float(norm[0, 2:].prod())
            support = ops.crop_mask(torch.ones((1, 640, 640), device='cuda'), box)[0].bool()
            pool = support.flatten().nonzero().flatten()
            if not len(pool) or area <= 0:
                rows.append(dict(**state, status='native_empty_support'))
                continue
            rng = np.random.default_rng(stable_seed('native-pixels', iid, aid))
            pos = pool[torch.as_tensor(rng.integers(len(pool), size=config['pixels']), device='cuda')]
            px = p640.flatten(1)[:, pos].T
            y = torch.as_tensor(mask_of(native, aid), device='cuda').flatten()[pos]
            sampled = px @ cap['coeff'][j]
            full = F.interpolate((cap['coeff'][j] @ p.flatten(1)).reshape(1, 1, *p.shape[-2:]),
                                  (640, 640), mode='bilinear', align_corners=False).flatten()[pos]
            err = float((sampled - full).abs().max())
            require(err < 2e-4, 'Prototype interpolation/sample replay failed')
            sample_error = max(sample_error, err)
            tids.append(aid); pidx.append(j); pp.append(px.cpu().numpy()); yy.append(y.cpu().numpy())
            factors.append(len(pool) / (640 * 640 * area)); positions.append(pos.cpu().numpy())
            rows.append(dict(**state, status='matched', prediction_index=j, support_pixels=len(pool),
                             positive_pixels=int(mask_of(native, aid)[support.cpu().numpy()].sum())))
        ns = config['pixels']
        arrays.update(annotation_ids=np.asarray(tids, np.int64), prediction_indices=np.asarray(pidx, np.int64),
                      sample_p=np.stack(pp) if pp else np.empty((0, ns, 32), np.float32),
                      sample_y=np.stack(yy) if yy else np.empty((0, ns), np.uint8),
                      loss_factor=np.asarray(factors, np.float32),
                      sample_positions=np.stack(positions) if positions else np.empty((0, ns), np.int64))
        path = args.out / 'images' / f'{iid}.npz'
        np.savez_compressed(path, **arrays)
        wr = dict(image_id=iid, split=split, native_ids=ids, targets=len(tids),
                  head_replay_max_abs=replay_error, sample_replay_max_abs=sample_error,
                  image_sha256=sha(image), label_sha256=sha(label_path) if label_path.exists() else None,
                  cache_sha256=sha(path), source_indices_unique=len(keep) == len(torch.unique(keep)))
        witness.append(wr)
        write_json(args.out / 'images' / f'{iid}.json', wr)
        if number % 8 == 0 or number == len(chosen):
            print(json.dumps(dict(stage='cache', images=number, total=len(chosen), seconds=time.monotonic()-start)), flush=True)
    for handle in hooks:
        handle.remove()
    save_csv(args.out / 'gt_status.csv', rows)
    write_json(args.out / 'witness.json', witness)
    write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', images=len(chosen),
               fit_images=nfit, transfer_images=ntest, targets=sum(x['status']=='matched' for x in rows),
               seconds=time.monotonic()-start,
               hashes={str(p.relative_to(args.out)): sha(p) for p in args.out.rglob('*')
                       if p.is_file() and 'converted' not in p.relative_to(args.out).parts}))


if __name__ == '__main__':
    main()
