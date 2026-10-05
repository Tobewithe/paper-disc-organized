"""S067: outcome-stratified mask repair opportunity on all COCO val2017.

This is an explicit GT-assisted diagnostic, not a method evaluation. A mask
replacement can extend outside a predicted box. Therefore poor-box repair
does NOT estimate what fixing only the box would achieve.
"""
from pathlib import Path
import argparse
import contextlib
import csv
import gc
import io
import json
import shutil
import time
from collections import defaultdict

from mask_error_ap_probe import (ANNOTATION, CENSUS, ROOT, COCO, dump, read_gz,
                                 evaluate, save_csv, sha, paired_stats)

SOURCE = ROOT / "diagnostics/mask_error_ap_20260913"
ARMS = ["original", "supported_goodbox_mask_gt", "limited_goodbox_mask_gt",
        "badbox_mask_gt", "strict_supported_mask_gt"]


def select_scope(row, arm):
    if arm == "original" or row["matched"] != "True":
        return False
    box, mask = float(row["box_iou"]), float(row["mask_iou"])
    if mask >= .75:
        return False
    if arm == "badbox_mask_gt":
        return box < .75
    if box < .75:
        return False
    support = float(row["box_gt_pixel_coverage"]) >= .95
    if arm == "limited_goodbox_mask_gt":
        return not support
    if arm == "supported_goodbox_mask_gt":
        return support
    if arm == "strict_supported_mask_gt":
        return support and row["any_kept_mask75"] == "False" and row["official_mask75"] == "False"
    raise ValueError(arm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with CENSUS.open(encoding="utf-8-sig") as f:
        metadata = {int(r["annotation_id"]): r for r in csv.DictReader(f)}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANNOTATION))
    ids = sorted(gt.imgs)
    selected = {arm: [aid for aid, row in metadata.items() if select_scope(row, arm)] for arm in ARMS}
    assert len(metadata) == 36335 and len(ids) == 5000
    assert [len(selected[x]) for x in ARMS[1:]] == [4417, 1158, 6584, 4260]
    assert set(selected[ARMS[1]]).isdisjoint(selected[ARMS[2]])
    assert set(selected[ARMS[1]]).isdisjoint(selected[ARMS[3]])
    dump(out / "protocol.json", dict(
        experiment="S067_FAILURE_DIMENSION_MASK_REPAIR", status="FROZEN_BEFORE_EVALUATION",
        question="How much task AP is recoverable by ideal masks separately in each fixed-slot failure state?",
        decision="Prioritize which failure states warrant mechanism tracing; never equate GT repair with achievable method gain.",
        images=5000, ordinary_gt=36335, arms=ARMS,
        selected_counts={k: len(v) for k, v in selected.items()},
        attribution="S048 same-class official bbox50 1:1 assignment; Box75 x Mask75 on identical original slot.",
        intervention="Replace ONLY selected original masks with own full COCO GT RLE; preserve every score/class/order/candidate. Selection uses GT; no model training.",
        support="95% support is original-coordinate floating-box raster proxy, NOT exact decoder support.",
        warning="GT replacement may extend outside predicted box. Badbox arm is ideal mask conditional on bad box, NOT box-only repair. No final assignment cannot be repaired this way; unknown, not zero AP potential. Arms overlap at strict/supported and all gains are nonadditive.",
        grouping="All GT matched by COCOeval before E4 or ICI stratification; E4 low0/middle(0,.2)/high>=.2.",
        statistics="Paired image-cluster bootstrap for R75 only; no AP significance test. Reuses explored validation data.",
        source_census_sha256=sha(CENSUS), script_sha256=sha(__file__),
        parent_receipt_sha256=sha(SOURCE / "COMPLETE.json")))
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("mask_error_ap_probe.py"), out / "mask_error_ap_probe.py")
    save_csv(out / "selected_instances.csv", [dict(arm=arm, annotation_id=aid,
             image_id=int(metadata[aid]["image_id"]), slot=int(metadata[aid]["prediction_slot"]))
             for arm in ARMS[1:] for aid in selected[arm]])
    summaries, records = [], {}
    for arm in ARMS:
        patches = defaultdict(dict)
        for aid in selected[arm]:
            row = metadata[aid]
            iid, slot = int(row["image_id"]), int(row["prediction_slot"])
            assert slot not in patches[iid]
            rle = gt.annToRLE(gt.anns[aid])
            if isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].decode("ascii")
            patches[iid][slot] = (rle, aid)
        predictions, changed = [], 0
        for iid in ids:
            saved = read_gz(SOURCE / "images" / f"{iid}.json.gz")
            for slot, pred in enumerate(saved["original"]):
                if slot in patches[iid]:
                    rle, aid = patches[iid][slot]
                    row = metadata[aid]
                    assert pred["category_id"] == int(row["category_id"])
                    assert abs(pred["score"] - float(row["score"])) < 1e-12
                    pred = dict(pred, segmentation=rle)
                    changed += 1
                # bbox is intentionally absent in mask predictions, so COCO's
                # mask area (not stale box area) controls size-range ignores.
                assert "bbox" not in pred
                predictions.append(pred)
        assert changed == len(selected[arm]) and len(predictions) == 446097
        dump(out / "progress.json", dict(stage="COCO_EVALUATE", arm=arm, completed=len(summaries), changed=changed))
        print(json.dumps(dict(stage="COCO_EVALUATE", arm=arm, changed=changed)), flush=True)
        summary, rec = evaluate(gt, predictions, ids, metadata, arm)
        if arm == "original":
            reference = json.loads((SOURCE / "original.json").read_text())["summary"]
            for key in ["mask_ap", "mask_ap50", "mask_ap75", "hit75_all", "hit75_high"]:
                assert abs(summary[key] - reference[key]) < 1e-12, (key, summary[key], reference[key])
        summary["selected_gt"] = changed
        summary["delta_ap_points"] = 100 * (summary["mask_ap"] - summaries[0]["mask_ap"]) if summaries else 0.
        save_csv(out / f"{arm}_gt.csv", rec)
        dump(out / f"{arm}.json", dict(summary=summary))
        records[arm] = rec
        summaries.append(summary)
        save_csv(out / "task_summary.csv", summaries)
        print(json.dumps(summary), flush=True)
        del predictions
        gc.collect()
    # Existing image-cluster implementation indexes draws with replacement;
    # it does not deduplicate resampled image IDs via an isin() filter.
    stats = {arm: paired_stats(records["original"], records[arm]) for arm in ARMS[1:]}
    dump(out / "paired_stats.json", stats)
    dump(out / "COMPLETE.json", dict(status="COMPLETE", seconds=time.monotonic()-start,
         images=5000, gt=36335, arms=ARMS, baseline_parity=True,
         task_summary_sha256=sha(out / "task_summary.csv")))
    dump(out / "progress.json", dict(stage="COMPLETE", seconds=time.monotonic()-start))


if __name__ == "__main__":
    main()
