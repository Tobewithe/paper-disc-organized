"""Small CPU engineering checks of pixel bookkeeping and actual Boundary COCOeval.

Synthetic masks establish formulas/semantics only. No real model result, gain,
GT rematching of final8 data, GPU call or method-selection evidence is produced.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from comparison_io import InputLock, check, encoded_rle, sha, write_json
import multiview_readout as readout


def rle(mask, mask_utils):
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return encoded | {"counts": encoded["counts"].decode("ascii")}


def evaluate(coco, COCOeval, records, kind, out):
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        result = coco.loadRes(readout.segmentation_records(records))
        evaluator = COCOeval(coco, result, iouType=kind, dilation_ratio=.02) if kind == "boundary" else COCOeval(coco, result, iouType=kind)
        evaluator.params.imgIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    check(len(result.anns) == len(records), "Empty prediction ordinal was removed")
    check(result.anns[1]["area"] == 1600 and result.anns[2]["area"] == 0, "Segmentation result area incorrectly follows supplied bbox/area")
    np.savez_compressed(out/(kind+"_ACCUMULATED.npz"), precision=evaluator.eval["precision"], recall=evaluator.eval["recall"],
        iou_thresholds=evaluator.params.iouThrs, max_detections=evaluator.params.maxDets, area_ranges=evaluator.params.areaRng)
    (out/(kind+".txt")).write_text(stream.getvalue(), encoding="utf-8")
    return evaluator, {name: float(value) if value >= 0 else None for name, value in zip(readout.METRIC_NAMES, evaluator.stats)}


def run(args):
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    check(not (out/"ENGINEERING_COMPLETE.json").exists() and not (out/"ENGINEERING_FAILURE.json").exists(), "Engineering history exists; use a new Run")
    lock = InputLock()
    for path in (Path(__file__), Path(readout.__file__), Path(__file__).with_name("comparison_io.py"), args.protocol):
        lock.add(path)
    pixel_metrics, pixel_source = readout.load_pixel_metrics(); lock.add(pixel_source)
    write_json(out/"ENGINEERING_INPUTS.json", {"scope": "small synthetic CPU checks, no method efficacy claim", "protocol_sha256": sha(args.protocol),
        "python": sys.version, "executable": sys.executable, "numpy": np.__version__, "GPU_or_model_execution": False,
        "synthetic_inputs": True, "source_sha256": {"readout": sha(readout.__file__), "check": sha(__file__), "pixel_metrics": sha(pixel_source)}})
    tests = {}
    gt = np.zeros((12, 12), dtype=bool); gt[2:4, 2:4] = True
    pred = gt.copy(); pred[3, 3] = False; pred[6:9, 6:9] = True
    values = pixel_metrics(pred, gt)
    expected = {"tp": 3, "fp": 9, "fn": 1, "gt_pixel_area": 4, "prediction_pixel_area": 12,
        "mask_iou": 3/13, "target_coverage": .75, "prediction_purity": .25, "fp_per_gt_area": 2.25, "fn_per_gt_area": .25}
    check(values == expected, "Closed pixel-count example does not satisfy metric definitions")
    empty = pixel_metrics(np.zeros_like(gt), gt)
    check(empty["mask_iou"] == empty["target_coverage"] == 0 and empty["prediction_purity"] is None and empty["fn_per_gt_area"] == 1, "Empty prediction missingness/coverage differs")
    zero = pixel_metrics(np.zeros_like(gt), np.zeros_like(gt))
    check(zero["mask_iou"] is None and zero["target_coverage"] is None and zero["prediction_purity"] is None, "Zero denominator fabricated quality")
    changed = readout.record_metrics(empty, values)
    check(changed["delta_prediction_purity"] is None and changed["delta_target_coverage"] == -.75, "Paired metric difference used unequal valid cohorts")
    tests["pixel_formula_closed_counts"] = tests["empty_prediction_purity_null"] = tests["zero_denominator_null"] = tests["paired_defined_denominators"] = True
    # Two detections may share a GT; averaging remains candidate based, not one GT.
    candidates = readout.Means(fields=("target_coverage", "prediction_purity"))
    candidates.add({"target_coverage": 1., "prediction_purity": 1.})
    candidates.add({"target_coverage": 0., "prediction_purity": None})
    table = candidates.finish()
    check(table["row_count"] == 2 and table["fields"]["target_coverage"]["mean"] == .5 and table["fields"]["prediction_purity"]["valid_count"] == 1
        and table["fields"]["prediction_purity"]["missing_count"] == 1, "Repeated-GT candidate or purity denominator lost")
    image_macro = readout.Means(fields=("target_coverage",))
    image_macro.add({"target_coverage": 1.}); image_macro.add({"target_coverage": 0.})
    unequal_candidates = readout.Means(fields=("target_coverage",))
    for value in (1., 1., 1., 1., 0.):
        unequal_candidates.add({"target_coverage": value})
    check(image_macro.finish()["fields"]["target_coverage"]["mean"] == .5
        and unequal_candidates.finish()["fields"]["target_coverage"]["mean"] == .8, "Candidate and equal-image weighting were conflated")
    tests["candidate_duplicate_GT_denominator"] = tests["image_macro_equal_image_weights"] = True
    provenance = readout.check_boundary_provenance(args.boundary_vendor.resolve(), args.boundary_provenance.resolve(), sha(args.boundary_provenance), lock)
    check(provenance["upstream_revision"] is None, "Unknown vendor revision was fabricated")
    sys.path.insert(0, str(args.boundary_vendor.resolve()))
    from boundary_iou.coco_instance_api.coco import COCO as BoundaryCOCO
    from boundary_iou.coco_instance_api.cocoeval import COCOeval as BoundaryEval
    from boundary_iou.utils.boundary_utils import mask_to_boundary
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils
    import cv2
    cv2.setNumThreads(1)
    border = mask_to_boundary(np.ones((4, 4), dtype=np.uint8), .02)
    check(int(border.sum()) == 12 and int(border[1:3, 1:3].sum()) == 0, "Boundary padding lost image-border foreground")
    scene = np.zeros((96, 96), dtype=np.uint8); scene[20:60, 20:60] = 1
    shifted = np.zeros_like(scene); shifted[20:60, 21:61] = 1
    crowd = np.zeros_like(scene); crowd[80:85, 80:85] = 1
    records = [{"image_id": 1, "category_id": 1, "score": .9, "bbox": [0, 0, 1, 1], "area": 1, "segmentation": rle(shifted, mask_utils)},
        {"image_id": 1, "category_id": 1, "score": .1, "bbox": [0, 0, 1, 1], "area": 1, "segmentation": rle(np.zeros_like(scene), mask_utils)}]
    original = copy.deepcopy(records)
    clean = readout.segmentation_records(records)
    check(all(set(row) == {"image_id", "category_id", "score", "segmentation"} for row in clean), "Segmentation evaluator received bbox/area contamination")
    annotations = {"info": {}, "images": [{"id": 1, "height": 96, "width": 96}], "categories": [{"id": 1, "name": "synthetic"}],
        "annotations": [{"id": 11, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [20, 20, 40, 40], "area": 1600, "segmentation": rle(scene, mask_utils)},
            {"id": 12, "image_id": 1, "category_id": 1, "iscrowd": 1, "bbox": [80, 80, 5, 5], "area": 25, "segmentation": rle(crowd, mask_utils)}]}
    gt_path = out/"SYNTHETIC_COCO.json"; write_json(gt_path, annotations)
    np.savez_compressed(out/"SYNTHETIC_MASKS.npz", GT=scene, prediction=shifted, crowd=crowd)
    with contextlib.redirect_stdout(io.StringIO()):
        normal = COCO(str(gt_path)); alternate = BoundaryCOCO(str(gt_path))
    fixed = [{"image_id": 1, "detection_index": 0, "annotation_id": 11, "category_id": 1, "baseline_score": .9,
        "baseline_mask_iou": 1560/1640, "baseline_success": True},
        {"image_id": 1, "detection_index": 1, "annotation_id": 11, "category_id": 1, "baseline_score": .1,
            "baseline_mask_iou": 0., "baseline_success": False}]
    indexed = readout.fixed_pairs_for_image(1, fixed, records, normal)
    check(len(indexed) == 2 and len({row["annotation_id"] for row in indexed.values()}) == 1,
        "Fixed repeated GT associations were turned into one candidate or rematched")
    normal_eval, normal_metrics = evaluate(normal, COCOeval, records, "segm", out)
    alternate_dir = out/"vendor_normal"; alternate_dir.mkdir()
    _, alternate_metrics = evaluate(alternate, BoundaryEval, records, "segm", alternate_dir)
    check(normal_metrics == alternate_metrics, "Vendor ordinary COCOeval does not reproduce actual ordinary evaluator")
    alternate.get_boundary, alternate.dilation_ratio = True, .02
    with contextlib.redirect_stdout(io.StringIO()):
        alternate.createIndex()
    boundary_eval, boundary_metrics = evaluate(alternate, BoundaryEval, records, "boundary", out)
    check(records == original, "Evaluation mutated input prediction identities/area/bbox")
    check(boundary_metrics["AP"] <= normal_metrics["AP"] and boundary_eval.ious[(1, 1)][0, 0] < normal_eval.ious[(1, 1)][0, 0], "Boundary result did not follow its declared boundary/min-IoU semantics")
    check(normal_eval.params.maxDets == boundary_eval.params.maxDets == [1, 10, 100], "Normal/Boundary counting budget differs")
    tests["normal_vendor_COCO_parity"] = tests["segmentation_loadRes_mask_area"] = tests["predictions_unchanged_including_empty"] = True
    tests["normal_and_boundary_independent_matching"] = tests["boundary_border_padding"] = tests["vendor_revision_unknown_retained"] = True
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and (getattr(module, "__name__", "").startswith(("boundary_iou", "pycocotools")) or module is cv2):
            lock.add(path)
    write_json(out/"SOURCE_LOCK.json", {"files": lock.finish(), "all_before_after_sha256_equal": True})
    report = {"status": "passed", "passed": True, "engineering_only": True, "synthetic_masks": True,
        "tests": tests, "synthetic_normal_metrics": normal_metrics, "synthetic_boundary_metrics": boundary_metrics,
        "formal_seven_arm_metrics": None, "method_efficacy_claimed": False, "GPU_or_model_execution": False,
        "source_lock_sha256": sha(out/"SOURCE_LOCK.json"), "memory_parent_process": readout.memory_observation()}
    write_json(out/"ENGINEERING_COMPLETE.json", report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out-dir", "protocol", "boundary-vendor", "boundary-provenance"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir/"ENGINEERING_FAILURE.json", {"status": "failed", "passed": False,
            "error": repr(error), "traceback": traceback.format_exc(), "GPU_or_model_execution": False})
        raise


if __name__ == "__main__":
    main()
