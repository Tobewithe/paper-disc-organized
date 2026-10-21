from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import Run, iou_matrix, read_json, setup_paths, sha256, write_json


def extract(root: Path, config: dict, smoke=False, demo_only=False, only_names=None, pointer_name=None):
    setup_paths(root)
    from ultralytics import YOLO, __version__
    assert __version__ == "8.4.100", __version__
    manifest = read_json(root / "input_manifest.json")
    sequences = manifest["sequences"]
    if only_names is not None:
        sequences = [s for s in sequences if s["name"] in only_names]
    if demo_only:
        sequences = [s for s in sequences if s["split"] == "demo"]
    if smoke:
        sequences = [next(s for s in sequences if s["name"] == "pig_3"), next(s for s in sequences if s["name"] == "horse_4")]
    cfg = dict(config, stage="feature_smoke" if smoke else "frozen_detector_cache", gt_used_by_detector=False,
               feature="P3/P4 predicted-mask weighted mean and P3 standard deviation; fixed YOLO features",
               model_sha256={k: sha256(root / v) for k, v in manifest["models"].items()},
               frame_limit=8 if smoke else None)
    with Run(root, "feature_smoke" if smoke else "features_demo_repair" if demo_only else "features", cfg) as run:
        models = {}
        stats = []
        for seq in sequences:
            model_key = seq["model"]
            if model_key not in models:
                models[model_key] = YOLO(str(root / manifest["models"][model_key]))
            model = models[model_key]
            # Initialize predictor/fusion first; hook actual executed head input afterward.
            warm = np.zeros((640, 640, 3), np.uint8)
            model.predict(warm, imgsz=config["imgsz"], conf=config["conf"], max_det=config["max_det"],
                          device=0, verbose=False)
            holder = {}
            def capture(_module, args):
                holder["maps"] = [x.detach().float() for x in args[0][:2]]
            handle = model.model.model[-1].register_forward_pre_hook(capture)
            cap = cv2.VideoCapture(str(root / seq["video"]))
            assert cap.isOpened(), seq
            cache = run.path / "cache" / seq["name"]
            cache.mkdir(parents=True)
            records, all_feats, seconds = [], [], []
            limit = min(seq["frames"], 8) if smoke else seq["frames"]
            t0 = time.perf_counter()
            for frame_idx in range(1, limit + 1):
                ok, img = cap.read()
                if not ok:
                    raise RuntimeError(f"Unexpected end of {seq['name']} at {frame_idx}/{limit}")
                start = time.perf_counter()
                r = model.predict(img, imgsz=config["imgsz"], conf=config["conf"], iou=config["iou"],
                                  max_det=config["max_det"], device=0,
                                  classes=seq["classes"], retina_masks=False, verbose=False)[0]
                boxes = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                n = len(boxes)
                if n:
                    masks = r.masks.data.float()
                    map3, map4 = holder["maps"]
                    features = []
                    pooled_masks = F.interpolate(masks[:, None], size=map3.shape[-2:], mode="area")[:, 0]
                    for feature_map in [map3, map4]:
                        mask = F.interpolate(masks[:, None], size=feature_map.shape[-2:], mode="area")[:, 0]
                        sums = mask.sum((1, 2)).clamp(min=1)
                        avg = torch.einsum("nhw,chw->nc", mask, feature_map[0]) / sums[:, None]
                        features.append(avg)
                        if feature_map is map3:
                            var = torch.einsum("nhw,chw->nc", mask, feature_map[0] ** 2) / sums[:, None] - avg ** 2
                            features.append(var.clamp(min=0).sqrt())
                    features = torch.cat(features, dim=1).cpu().numpy().astype(np.float32)
                    count = pooled_masks.sum(0)
                    conflict = ((pooled_masks * (count[None] - pooled_masks).clamp(0, 1)).sum((1, 2)) /
                                pooled_masks.sum((1, 2)).clamp(min=1)).cpu().numpy()
                    mask_frac = masks.mean((1, 2)).cpu().numpy()
                    polygons = [p[::max(1, len(p)//80)].round(1).tolist() for p in r.masks.xy]
                else:
                    dim = holder["maps"][0].shape[1] * 2 + holder["maps"][1].shape[1]
                    features = np.empty((0, dim), np.float32)
                    conflict, mask_frac, polygons = np.empty(0), np.empty(0), []
                pair_iou = iou_matrix(boxes, boxes)
                if n:
                    np.fill_diagonal(pair_iou, 0)
                overlap = pair_iou.max(1) if n else np.empty(0)
                quality = np.column_stack([conf, overlap, conflict, mask_frac]).astype(np.float32)
                offset = sum(len(f) for f in all_feats)
                all_feats.append(features)
                records.append(dict(frame=frame_idx, time=(frame_idx - 1)/seq["fps"], boxes=boxes.tolist(),
                                    scores=conf.tolist(), classes=cls.tolist(), polygons=polygons,
                                    quality=quality.tolist(), feature_start=offset, feature_count=n))
                seconds.append(time.perf_counter() - start)
                if smoke:
                    annotated = r.plot()
                    cv2.imwrite(str(cache / f"smoke_{frame_idx:04d}.jpg"), annotated)
                if frame_idx % 60 == 0:
                    print(f"FEATURES {seq['name']} {frame_idx}/{limit} {sum(len(f) for f in all_feats)} detections", flush=True)
            cap.release()
            handle.remove()
            np.savez_compressed(cache / "features.npz", x=np.concatenate(all_feats, axis=0))
            write_json(cache / "detections.json", records)
            stats.append(dict(sequence=seq["name"], frames=limit, detections=sum(len(f) for f in all_feats),
                              seconds=time.perf_counter()-t0, mean_frame_ms=1000*np.mean(seconds),
                              p95_frame_ms=1000*np.percentile(seconds, 95),
                              feature_dim=all_feats[0].shape[1], cache_sha256=sha256(cache / "detections.json")))
            print("SEQUENCE_DONE", stats[-1], flush=True)
        write_json(run.path / "SUMMARY.json", dict(sequences=stats, cuda=torch.cuda.get_device_name(),
                                                   peak_memory_bytes=torch.cuda.max_memory_allocated()))
        write_json(run.path / "DECISION.json", dict(status="engineering_complete", next_action="inspect_smoke_then_freeze_full_cache" if smoke else "train_identity_projection_on_official_training_sequences", scientific_claim=False))
        if not smoke:
            write_json(root / (pointer_name or ("demo_features_pointer.json" if demo_only else "features_pointer.json")), dict(run_id=run.id, cache=str(run.path.relative_to(root)/"cache")))
    return run.id
