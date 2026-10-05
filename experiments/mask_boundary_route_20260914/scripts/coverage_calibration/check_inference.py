"""Replay actual GT-free adapters against bank decisions and time the full pipeline."""
from pathlib import Path
import argparse
import json
import csv
import time
from collections import defaultdict
from common import setup, atomic, progress, load_predictions


def main():
    parser = argparse.ArgumentParser()
    for arg in ['protocol','output','models','selection','evaluation']:
        parser.add_argument('--'+arg, required=True)
    args = parser.parse_args(); p, out = setup(args.protocol, args.output)
    import numpy as np
    import cv2
    import torch
    from pycocotools import mask as mu
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from calibrator import make_predictor
    from mask_calibration import export_masks
    ids = json.loads((Path(p['val_bank'])/'image_ids.json').read_text())
    ids = [ids[i] for i in np.linspace(0, len(ids)-1, 24, dtype=int)]
    images = [cv2.imread(str(Path(p['images']) / f'{i:012d}.jpg')) for i in ids]
    assert all(im is not None for im in images)
    flags = np.load(Path(args.evaluation)/'candidate_decisions.npz')
    positions = {tuple(key): i for i, key in enumerate(flags['keys'])}
    base, _ = load_predictions(Path(p['val_bank'])/'predictions_official_zero.json')
    _, trial = load_predictions(Path(p['val_bank'])/'predictions_smooth_gated.json')
    by_image = defaultdict(list)
    for r in base:
        if r['image_id'] in set(ids[:6]): by_image[r['image_id']].append(r)
    boxes = {}
    for r in csv.DictReader((Path(p['val_bank'])/'candidate_records.csv').open(newline='',encoding='utf-8')):
        if int(r['image_id']) in set(ids[:6]):
            boxes[(int(r['image_id']), int(r['candidate_index']))] = [float(r['box_'+k]) for k in ['x1','y1','x2','y2']]
    categories = json.loads(Path(p['val_annotations']).read_text())['categories']
    name_to_id = {c['name']: c['id'] for c in categories}
    model = YOLO(p['weights']); assert model.model.model[-1].end2end is True
    names = {int(k): name_to_id[v] for k, v in model.names.items()}
    factories = {'baseline': SegmentationPredictor}
    for name in ['frozen_rcmc','direct','decomposed','decomposed_protected']:
        factories[name] = make_predictor(args.models, p['frozen_rcmc'], args.selection, name, chunk=24)
    opts = dict(imgsz=640, conf=.001, max_det=300, device=0, batch=1, half=False,
                verbose=False, save=False, retina_masks=False)
    parity = {}
    for name, factory in factories.items():
        model.predictor = None; count = 0; max_box_error = 0.; pixel_errors = 0
        for iid, im in zip(ids[:6], images[:6]):
            result = model.predict(im, predictor=factory, **opts)[0]
            masks = export_masks(result.masks.data.byte(), im.shape[:2]).cpu().numpy().astype(bool)
            # Bank exports omit empty original-grid masks; preserve raw order for remaining slots.
            nonempty = masks.any(axis=(1,2)); masks = masks[nonempty]
            actual_boxes = result.boxes.data.cpu().numpy()[nonempty]
            reference = sorted(by_image[iid], key=lambda r: r['candidate_index'])
            assert len(masks) == len(reference), (name, iid, len(masks), len(reference))
            for actual, box, row in zip(masks, actual_boxes, reference):
                key = (iid,row['candidate_index']); index = positions[key]
                chosen = trial[key] if flags[name][index] else row
                expected = mu.decode(chosen['segmentation']).astype(bool)
                pixel_errors += int(np.count_nonzero(actual != expected))
                err = float(np.max(np.abs(box[:4]-np.asarray(boxes[key])))); max_box_error = max(max_box_error,err)
                assert abs(float(box[4])-row['score']) < 1e-6 and names[int(box[5])] == row['category_id']
                count += 1
            del result
        assert pixel_errors == 0 and max_box_error < 1e-3, (name, pixel_errors,max_box_error)
        parity[name] = {'images':6,'masks':count,'pixel_mismatches':pixel_errors,'max_box_error':max_box_error}
        atomic(out/'parity.json',parity);progress(out,'parity',variant=name,**parity[name])
    del base,trial,flags; import gc;gc.collect()
    timings = []; modes = list(factories); peaks = {}
    for repeat in range(3):
        order = modes[repeat:] + modes[:repeat]
        for name in order:
            model.predictor = None
            for im in images[:4]: model.predict(im, predictor=factories[name], **opts)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            for iid, im in zip(ids, images):
                torch.cuda.synchronize(); start = time.perf_counter()
                result = model.predict(im, **opts)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                timings.append({'repeat':repeat,'variant':name,'image_id':iid,'milliseconds':elapsed,
                                'outputs':len(result[0].boxes)})
                del result
            peaks[name] = max(peaks.get(name,0),torch.cuda.max_memory_allocated()/2**20)
            atomic(out/'timings_partial.json',timings)
            progress(out,'timing',repeat=repeat,variant=name,calls=len(timings))
    stats = {}
    for name in modes:
        values = np.asarray([r['milliseconds'] for r in timings if r['variant']==name])
        stats[name] = {'calls':len(values),'mean_ms':float(values.mean()),'median_ms':float(np.median(values)),
                       'p95_ms':float(np.quantile(values,.95)), 'peak_gpu_allocated_mib':peaks[name],
                       'block_mean_ms':[float(np.mean([r['milliseconds'] for r in timings if r['variant']==name and r['repeat']==j])) for j in range(3)]}
    atomic(out/'timings.json',timings)
    atomic(out/'SUMMARY.json',{'parity':parity,'timing':stats,'image_ids':ids,
        'fixed_versions':p['environment'],'gpu':torch.cuda.get_device_name(),
        'scope':'6 fixed images for exact mask parity; 24 evenly spaced fixed images x 3 cyclic-order passes per variant, CUDA synchronized, batch 1 FP32',
        'included':'preprocess, YOLO forward, normal postprocessing, feature extraction, CPU transfers, required predictor heads and mask selection',
        'excluded':'image disk IO, model/predictor setup and warmup, RLE export and COCO evaluation',
        'limitations':['single laptop hardware and operating state','test GT is used for offline COCO metrics only, not adapter inputs or policy','timings are sampled pipeline costs, not full-dataset throughput']})
    progress(out,'completed',parity=parity,timing=stats)


if __name__ == '__main__': main()
