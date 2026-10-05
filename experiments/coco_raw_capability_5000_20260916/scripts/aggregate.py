"""Join raw geometry with official COCOeval; never change raw geometry by scores.

May follow a running upstream run. No aggregate conclusions are emitted until
all 5,000 shards and the upstream COMPLETE record exist. CPU only.
"""
import argparse
import contextlib
import gc
import io
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def area_group(area):
    # Disjoint descriptive strata; COCO's own area metrics use its native ranges.
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(a.root / "assets/datasets/coco/annotations/instances_val2017.json"))
    ids = sorted(coco.imgs)
    assert len(ids) == 5000
    cat_ids = sorted(coco.cats)
    # Official COCO80 class order is sorted original category ID.
    cat_cls = {c: i for i, c in enumerate(cat_ids)}
    all_rows, predictions = [], []
    matching = {t: Counter() for t in ("0.5", "0.75")}
    totals = Counter()
    stages = ("raw_mask75", "raw_good_gt_score_gt001", "head_class_mask75", "conf_class_mask75", "export_class_mask75", "budget_class_mask75")
    for number, image_id in enumerate(ids):
        path = a.source / "images" / f"{image_id:012d}.json"
        while not path.exists():
            status = json.loads((a.source / "run.json").read_text(encoding="utf-8")).get("status")
            if status in ("failed", "error", "cancelled", "interrupted"):
                raise RuntimeError(f"Upstream {status}; missing {image_id}")
            time.sleep(2)
        # The JSON is written last for each image; retry a transient partial write.
        for attempt in range(20):
            try:
                shard = json.loads(path.read_text(encoding="utf-8"))
                break
            except json.JSONDecodeError:
                time.sleep(.1)
        else:
            raise RuntimeError(f"Incomplete shard {path}")
        with np.load(path.with_suffix(".npz")) as z:
            b = z["box_ious"]
            m = z["mask_ious"]
            scores = z["scores"]
            ti, tc, ts = z["top_ids"], z["top_classes"], z["top_scores"]
            assert np.array_equal(z["gt_ids"], [r["annotation_id"] for r in shard["gt_rows"]])
        good = np.nan_to_num(m, nan=-1) >= .75
        # All threshold-crossing edges were certified by the raw evaluator.
        local_preds = shard["predictions"]
        totals.update(raw_candidates=shard["raw_count"], decoded_geometry=shard["geometry_decoded"],
                      normal_predictions=len(local_preds), duplicate_raw_exports=len(local_preds)-len({p["raw_id"] for p in local_preds}))
        budget = set()
        for cat in {p["category_id"] for p in local_preds}:
            ix = [i for i, p in enumerate(local_preds) if p["category_id"] == cat]
            ix.sort(key=lambda i: (-local_preds[i]["score"], i))
            budget.update(ix[:100])
        for gi, row in enumerate(shard["gt_rows"]):
            ci, cat = cat_cls[row["category_id"]], row["category_id"]
            good_ids = np.flatnonzero(good[:, gi])
            class_top = tc == ci
            row["area_group"] = area_group(row["area"])
            row["raw_mask75"] = bool(len(good_ids))
            row["raw_mask75_count"] = int(len(good_ids))
            row["raw_good_argmax_class_correct"] = bool((scores[good_ids].argmax(1) == ci).any()) if len(good_ids) else False
            row["raw_good_gt_score_max"] = float(scores[good_ids, ci].max()) if len(good_ids) else None
            row["raw_good_gt_score_gt001"] = bool(row["raw_mask75"] and row["raw_good_gt_score_max"] > .001)
            row["head_class_mask75"] = bool(good[ti[class_top], gi].any())
            row["conf_class_mask75"] = bool(good[ti[class_top & (ts > .001)], gi].any())
            valid = [i for i, p in enumerate(local_preds) if p["category_id"] == cat and good[p["raw_id"], gi]]
            row["export_class_mask75"] = bool(valid)
            row["budget_class_mask75"] = bool(set(valid) & budget)
            row["same_gt_positive_has_mask75"] = bool(row.get("assigned_mask_max") is not None and row["assigned_mask_max"] >= .75)
            # State-representative pixel arithmetic, not a network-cause diagnosis.
            w = row["best_mask_with_box75"] or row["best_mask"]
            ga = row["mask_area"]
            row["pixel_reference_role"] = "best_mask_with_box75" if row["best_mask_with_box75"] else "best_mask"
            row["pixel_reference"] = w
            row["fp_removal_iou"] = w["tp"] / ga
            row["fn_fill_iou"] = ga / (ga + w["fp"])
            row["pixel_repair75"] = ("already_pass" if w["mask_iou"] >= .75 else
                "either_fp_or_fn_suffices" if row["fp_removal_iou"] >= .75 and row["fn_fill_iou"] >= .75 else
                "only_fp_removal_suffices" if row["fp_removal_iou"] >= .75 else
                "only_fn_fill_suffices" if row["fn_fill_iou"] >= .75 else "both_needed")
            row["candidate_support_below75"] = w["crop_support"] < .75
            all_rows.append(row)
        for threshold in matching:
            matching[threshold].update(shard["matching"][threshold])
        predictions.extend(local_preds)
        if (number+1) % 100 == 0 or number == 4999:
            progress = dict(images=number+1, gt=len(all_rows), predictions=len(predictions), elapsed_s=round(time.monotonic()-started, 2), stage="joining")
            dump(a.out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    while not (a.source / "COMPLETE.json").exists():
        time.sleep(1)
    complete = json.loads((a.source / "COMPLETE.json").read_text())
    assert complete["images"] == 5000 and complete["gt"] == len(all_rows) == 36335
    assert len({r["annotation_id"] for r in all_rows}) == len(all_rows)
    row_by_id = {r["annotation_id"]: r for r in all_rows}
    dump(a.out / "predictions_with_identity.json", predictions)
    coco_stats = {}
    dt_audits = {}
    for task in ("bbox", "segm"):
        print(f"COCOeval {task}: {len(predictions)} predictions", flush=True)
        dump(a.out / "progress.json", dict(images=5000, stage="cocoeval_"+task, elapsed_s=round(time.monotonic()-started, 2)))
        # Deliberately omit bbox in segm loadRes, preserving mask-based area.
        field = "bbox" if task == "bbox" else "segmentation"
        dets = [{k: p[k] for k in ("image_id", "category_id", "score", field)} for p in predictions]
        dt = coco.loadRes(dets)
        ev = COCOeval(coco, dt, task)
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        names = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100", "ARS", "ARM", "ARL")
        stats = dict(zip(names, [float(v) for v in ev.stats]))
        audits = {}
        for entry in ev.evalImgs:
            if entry is None or entry["aRng"] != ev.params.areaRng[0] or entry["maxDet"] != 100:
                continue
            for gt_id, ignore, matched in zip(entry["gtIds"], entry["gtIgnore"], entry["gtMatches"][5]):
                if gt_id in row_by_id:
                    assert not ignore
                    row = row_by_id[gt_id]
                    row[task+"_coco75"] = bool(matched)
                    row[task+"_prediction_id"] = int(matched) if matched else None
                    row[task+"_raw_id"] = predictions[int(matched)-1]["raw_id"] if matched else None
            for j, det_id in enumerate(entry["dtIds"]):
                audits[int(det_id)] = dict(score=float(entry["dtScores"][j]),
                    ignored=bool(entry["dtIgnore"][5,j]), matched_gt=int(entry["dtMatches"][5,j]))
        for row in all_rows:
            assert task+"_coco75" in row
        scored = sorted((v for v in audits.values() if not v["ignored"]), key=lambda v: -v["score"])
        score_array = np.array([v["score"] for v in scored])
        tp = np.cumsum([v["matched_gt"] > 0 for v in scored])
        precision = tp / np.arange(1, len(tp)+1)
        # A single inclusive global score threshold must include all tied scores.
        ends = np.r_[score_array[:-1] != score_array[1:], True]
        feasible = np.flatnonzero((precision >= .9) & ends)
        ix = int(feasible[np.argmax(tp[feasible])]) if len(feasible) else None
        stats["micro_R75_at_P90"] = float(tp[ix]/len(all_rows)) if ix is not None else 0.
        stats["micro_P90_score_threshold"] = float(score_array[ix]) if ix is not None else None
        stats["R75_TP_count"] = sum(r[task+"_coco75"] for r in all_rows)
        stats["R75_micro"] = stats["R75_TP_count"] / len(all_rows)
        stats["ordinary_evaluated_predictions"] = len(scored)
        stats["ignored_predictions"] = sum(v["ignored"] for v in audits.values())
        stats["budget_excluded_predictions"] = len(predictions)-len(audits)
        coco_stats[task] = stats
        dt_audits[task] = audits
        dump(a.out / (task+"_cocoeval.json"), stats)
        del ev, dt, dets
        gc.collect()
    failure = Counter()
    for row in all_rows:
        if row["segm_coco75"]:
            why = "normal_success"
            assert row["raw_mask75"] and row["budget_class_mask75"]
        elif not row["raw_mask75"]:
            why = "raw_geometry_unavailable"
        elif not row["raw_good_gt_score_gt001"]:
            why = "all_good_masks_gt_class_score_le001"
        elif not row["head_class_mask75"]:
            why = "good_mask_class_pair_not_in_head_top300"
        elif not row["conf_class_mask75"]:
            why = "head_has_good_mask_but_conf_filters_it"
        elif not row["export_class_mask75"]:
            why = "export_loss"
        elif not row["budget_class_mask75"]:
            why = "coco_category_max100_budget"
        else:
            why = "coco_matching_competition"
        row["normal_mask75_state"] = why
        failure[why] += 1
    def summarize(rows):
        return dict(gt=len(rows), geometry_states=dict(Counter(r["geometry_state"] for r in rows)),
            raw_box75=sum(r["box_max"]>=.75 for r in rows), raw_mask75=sum(r["raw_mask75"] for r in rows),
            raw_box50=sum(r["box_max"]>=.5 for r in rows), raw_mask50=sum(r["mask_max"]>=.5 for r in rows),
            normal_box75=sum(r["bbox_coco75"] for r in rows), normal_mask75=sum(r["segm_coco75"] for r in rows),
            normal_states=dict(Counter(r["normal_mask75_state"] for r in rows)),
            stages={s:sum(r[s] for r in rows) for s in stages},
            mean_box_max=mean([r["box_max"] for r in rows]),mean_mask_max=mean([r["mask_max"] for r in rows]),
            replay_o2o_no_positive=sum(r.get("one2one_positive_count",0)==0 for r in rows),
            replay_o2m_no_positive=sum(r.get("one2many_positive_count",0)==0 for r in rows),
            replay_own_mask75=sum(r["same_gt_positive_has_mask75"] for r in rows),
            pixel_repairs=dict(Counter(r["pixel_repair75"] for r in rows)),
            selected_support_below75=sum(r["candidate_support_below75"] for r in rows),
            witness_means={k:mean([r["pixel_reference"][k] for r in rows]) for k in ("mask_iou","box_iou","coverage","purity","boundary_iou","crop_support")})
    summary = dict(source_run=a.source.name, images=5000, totals=dict(totals), full=summarize(all_rows),
        raw_distinct_candidate_maximum_matching={t:dict(v) for t,v in matching.items()}, coco=coco_stats,
        by_area={key:summarize([r for r in all_rows if r["area_group"]==key]) for key in ("small","medium","large")},
        by_geometry={key:summarize([r for r in all_rows if r["geometry_state"]==key]) for key in sorted({r["geometry_state"] for r in all_rows})},
        by_category={str(cat):dict(name=coco.cats[cat]["name"],**summarize([r for r in all_rows if r["category_id"]==cat])) for cat in cat_ids},
        limitations=["GT-guided raw capability is not AP or a deployable method.",
          "Mask50/75 edges and maximum matching use the declared predict decoder, not native-retina val decoder.",
          "Original-GT TAL replay is neither historical assignment nor actual augmented mask-loss supervision.",
          "Pixel repair arithmetic does not identify neural causes or additive AP gains.",
          "All 5000 val images are exploratory; later learning uses train2017 and requires frozen evaluation."])
    dump(a.out / "SUMMARY.json", summary)
    with (a.out / "gt_joined.jsonl").open("w",encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    dump(a.out / "prediction_coco75_audit.json", dt_audits)
    dump(a.out / "COMPLETE.json", dict(images=5000,gt=len(all_rows),elapsed_s=round(time.monotonic()-started,2)))
    print(json.dumps({"complete":True,"full":summary["full"],"coco":coco_stats}),flush=True)


if __name__ == "__main__":
    main()
