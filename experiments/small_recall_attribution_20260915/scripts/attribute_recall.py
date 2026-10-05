"""Trace saved COCO predictions; no inference, training or custom GT matching."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse, contextlib, csv, gc, gzip, io, json, sys, traceback, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
import coco_metrics as cm


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def csv_read(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def eval_saved(gt, predictions, task):
    processed = cm.task_predictions(predictions, {c: c for c in gt.cats}, task, gt)
    with quiet():
        dt = gt.loadRes(processed)
        ev = COCOeval(gt, dt, task)
        ev.params.imgIds = sorted(gt.imgs)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def extract_matches(ev, ids):
    """Retain official area-dependent matches, including unique competition."""
    lookup = {v: i for i, v in enumerate(ids)}
    out = {}
    for area in ["all", "small"]:
        ai = ev.params.areaRngLbl.index(area)
        matches = np.zeros((len(ids), 10), dtype=np.int64)
        valid = np.zeros(len(ids), dtype=bool)
        for item in ev.evalImgs:
            if item is None or item["aRng"] != ev.params.areaRng[ai]:
                continue
            for j, ann_id in enumerate(item["gtIds"]):
                if ann_id in lookup and not item["gtIgnore"][j]:
                    i = lookup[ann_id]
                    valid[i] = True
                    matches[i] = item["gtMatches"][:, j].astype(np.int64)
        out[area + "_matches"] = matches
        out[area + "_valid"] = valid
    return out


def geometry(gt, bev, mev, ids):
    """Best-candidate opportunities; same saved row IDs align box and mask."""
    index = {v: i for i, v in enumerate(ids)}
    names = ["best_box_iou", "best_mask_iou", "mask_iou_at_best_box",
             "box_iou_at_best_mask", "best_box_score", "best_mask_score",
             "best_box_id", "best_mask_id", "same_class_top100_count",
             "coverage_at_best_mask", "purity_at_best_mask",
             "coverage_at_best_box", "purity_at_best_box"]
    out = {k: np.zeros(len(ids), dtype=np.float64) for k in names}
    for key, bg in bev._gts.items():
        if not bg:
            continue
        mg = mev._gts[key]
        assert [a["id"] for a in bg] == [a["id"] for a in mg]
        bd = sorted(bev._dts[key], key=lambda d: -d["score"])[:100]
        md = sorted(mev._dts[key], key=lambda d: -d["score"])[:100]
        assert [d["id"] for d in bd] == [d["id"] for d in md]
        if not bd:
            continue
        bi = np.asarray(bev.ious[key])
        mi = np.asarray(mev.ious[key])
        assert bi.shape == mi.shape == (len(bd), len(bg))
        for g, ann in enumerate(bg):
            if ann["id"] not in index:
                continue
            n = index[ann["id"]]
            ib, im = int(bi[:, g].argmax()), int(mi[:, g].argmax())
            values = [bi[ib, g], mi[im, g], mi[ib, g], bi[im, g],
                      bd[ib]["score"], md[im]["score"], bd[ib]["id"], md[im]["id"], len(bd)]
            for name, value in zip(names, values):
                out[name][n] = value
            # For noncrowd GT, invert IoU using actual RLE areas, not box areas.
            ga = float(mu.area(mg[g]["segmentation"]))
            for suffix, d in [("best_mask", im), ("best_box", ib)]:
                da = float(md[d]["area"])
                inter = float(mi[d, g]) * (ga + da) / (1 + float(mi[d, g]))
                out["coverage_at_" + suffix][n] = inter / ga if ga else 0
                out["purity_at_" + suffix][n] = inter / da if da else 0
    return out


def macro_recall(hit, valid, category):
    cats = sorted(set(category[valid]))
    return np.mean([hit[valid & (category == c)].mean() for c in cats])


def process_one(cfg, out, gt, ids, category, branch, arm):
    prefix = branch + "__" + arm
    folder = Path(cfg["source_evaluations"][arm])
    path = folder / "exports" / branch / "predictions.json.gz"
    print(now(), "READ", prefix, flush=True)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        predictions = json.load(f)
    evs, arrays, validation = {}, {}, []
    reference = csv_read(folder / "official_metrics.csv")
    for task in ["bbox", "segm"]:
        print(now(), "COCOeval", prefix, task, flush=True)
        ev = eval_saved(gt, predictions, task)
        ref = next(r for r in reference if r["branch"] == branch and r["task"] == task)
        diffs = {name: float(ev.stats[i]) - float(ref[name])
                 for i, name in enumerate(["ap", "ap50", "ap75", "aps", "apm", "apl",
                                          "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large"])}
        assert max(abs(d) for d in diffs.values()) < 1e-10, diffs
        data = extract_matches(ev, ids)
        for area, stat in [("all", 8), ("small", 9)]:
            reconstructed = macro_recall(data[area + "_matches"] > 0,
                                          data[area + "_valid"], category)
            assert abs(reconstructed - ev.stats[stat]) < 1e-12
        for k, v in data.items():
            arrays[task + "_" + k] = v
        evs[task] = ev
        validation.append(dict(arm=arm, branch=branch, task=task,
                               max_metric_difference=max(abs(d) for d in diffs.values()),
                               official_ar_small=float(ev.stats[9]),
                               official_ar_all=float(ev.stats[8])))
    arrays.update(geometry(gt, evs["bbox"], evs["segm"], ids))
    np.savez_compressed(out / (prefix + ".npz"), **arrays)
    save_json(out / (prefix + "_validation.json"), validation)
    print(now(), "EXTRACTED", prefix, flush=True)
    del predictions, evs, arrays
    gc.collect()


def paired_bootstrap(method, reference, valid, category, images, image_ids, weights):
    cats = sorted(set(category[valid]))
    cindex = {c: i for i, c in enumerate(cats)}
    iindex = {v: i for i, v in enumerate(image_ids)}
    numer = np.zeros((len(image_ids), len(cats)))
    denom = np.zeros_like(numer)
    for i in np.flatnonzero(valid):
        ix, cx = iindex[images[i]], cindex[category[i]]
        numer[ix, cx] += method[i].mean() - reference[i].mean()
        denom[ix, cx] += 1
    ns, ds = weights @ numer, weights @ denom
    ratios = np.divide(ns, ds, out=np.full_like(ns, np.nan), where=ds > 0)
    draws = np.nanmean(ratios, axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    return float(lo), float(hi)


def mechanism(old, i, threshold):
    if old["best_mask_iou"][i] >= threshold - 1e-10:
        return "qualifying_mask_already_present_but_unmatched"
    if old["best_box_iou"][i] >= threshold - 1e-10:
        return "mask_opportunity_added_with_existing_box_support"
    return "mask_opportunity_added_without_previous_same_threshold_box"


def summarize(cfg, out, gt, ids, category):
    annotations = [gt.anns[int(i)] for i in ids]
    images = np.array([a["image_id"] for a in annotations])
    image_ids = sorted(gt.imgs)
    weights = np.zeros((cfg["bootstrap"]["resamples"], len(image_ids)), dtype=np.float64)
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    for j in range(len(weights)):
        weights[j] = np.bincount(rng.integers(0, len(image_ids), len(image_ids)),
                                minlength=len(image_ids))
    contrasts, thresholds, categories, mechanisms, target_rows, spatial = [], [], [], [], [], []
    validation = []
    for branch in cfg["branches"]:
        arms = {arm: dict(np.load(out / (branch + "__" + arm + ".npz")))
                for arm in cfg["source_evaluations"]}
        for arm in arms:
            validation.extend(json.loads((out / (branch + "__" + arm + "_validation.json")).read_text()))
        for reference in ["baseline", "reg_only"]:
            old, new = arms[reference], arms["reg_scheduled"]
            contrast = "reg_scheduled minus " + reference
            for task in ["segm", "bbox"]:
                valid = old[task + "_small_valid"]
                assert np.array_equal(valid, new[task + "_small_valid"])
                mh, rh = new[task + "_small_matches"] > 0, old[task + "_small_matches"] > 0
                cats = sorted(set(category[valid]))
                ns = {c: int(np.sum(valid & (category == c))) for c in cats}
                delta = macro_recall(mh, valid, category) - macro_recall(rh, valid, category)
                lo, hi = paired_bootstrap(mh, rh, valid, category, images, image_ids, weights)
                contrasts.append(dict(branch=branch, contrast=contrast, task=task,
                                      small_gt=int(valid.sum()), categories=len(cats),
                                      reference_ar=macro_recall(rh, valid, category),
                                      method_ar=macro_recall(mh, valid, category),
                                      delta_points=100*delta, ci_low_points=100*lo, ci_high_points=100*hi,
                                      micro_delta_points=100*float((mh[valid].astype(float)-rh[valid]).mean())))
                for ti, threshold in enumerate(np.linspace(.5, .95, 10)):
                    gains = valid & mh[:, ti] & ~rh[:, ti]
                    losses = valid & rh[:, ti] & ~mh[:, ti]
                    thresholds.append(dict(branch=branch, contrast=contrast, task=task,
                                           threshold=round(float(threshold), 2), gained=int(gains.sum()),
                                           lost=int(losses.sum()), net=int(gains.sum()-losses.sum()),
                                           reference_matched=int(rh[valid, ti].sum()),
                                           method_matched=int(mh[valid, ti].sum()),
                                           macro_delta_points=100*(macro_recall(mh[:,ti],valid,category)-
                                                                    macro_recall(rh[:,ti],valid,category))))
                    if task != "segm":
                        continue
                    for direction, selected, absent in [("gained", gains, old), ("lost", losses, new)]:
                        grouped = {}
                        for i in np.flatnonzero(selected):
                            mech = mechanism(absent, i, threshold)
                            contribution = 100/(len(cats)*ns[category[i]]*10)
                            count, total = grouped.get(mech, (0, 0.))
                            grouped[mech] = count+1, total+contribution
                            if ti in [0, 5]:
                                row = dict(branch=branch, contrast=contrast, threshold=round(float(threshold),2),
                                           direction=direction, opportunity_state=mech,
                                           annotation_id=int(ids[i]), image_id=int(images[i]),
                                           category_id=int(category[i]), category_name=gt.cats[int(category[i])]["name"],
                                           gt_area=annotations[i]["area"],
                                           reference_box_matched=int(old["bbox_small_matches"][i,ti]>0),
                                           method_box_matched=int(new["bbox_small_matches"][i,ti]>0),
                                           reference_mask_score_id=int(old["segm_small_matches"][i,ti]),
                                           method_mask_score_id=int(new["segm_small_matches"][i,ti]))
                                for key in ["best_box_iou","best_mask_iou","mask_iou_at_best_box",
                                            "box_iou_at_best_mask","best_box_score","best_mask_score",
                                            "coverage_at_best_box","purity_at_best_box"]:
                                    row["reference_"+key]=float(old[key][i])
                                    row["method_"+key]=float(new[key][i])
                                target_rows.append(row)
                        for name,(count,total) in grouped.items():
                            mechanisms.append(dict(branch=branch,contrast=contrast,threshold=round(float(threshold),2),
                                                   direction=direction,opportunity_state=name,count=count,
                                                   ar_small_contribution_points=total*(1 if direction=="gained" else -1)))
                    if ti in [0,5]:
                        for direction, sel in [("gained",gains),("lost",losses)]:
                            for key in ["best_box_iou","best_mask_iou","mask_iou_at_best_box",
                                        "box_iou_at_best_mask","coverage_at_best_box","purity_at_best_box"]:
                                spatial.append(dict(branch=branch,contrast=contrast,threshold=round(float(threshold),2),
                                                    direction=direction,n=int(sel.sum()),metric=key,
                                                    reference=float(old[key][sel].mean()) if sel.any() else None,
                                                    method=float(new[key][sel].mean()) if sel.any() else None))
                for c in cats:
                    sel=valid & (category==c)
                    b,m=float(rh[sel].mean()),float(mh[sel].mean())
                    categories.append(dict(branch=branch,contrast=contrast,task=task,category_id=int(c),
                                           category_name=gt.cats[int(c)]["name"],small_gt=ns[c],
                                           reference_ar=b,method_ar=m,delta_points=100*(m-b),
                                           macro_contribution_points=100*(m-b)/len(cats)))
            # Export every small GT once for this comparison, keeping threshold-dependent matches in NPZ.
            rows=[]
            valid=old["segm_small_valid"]
            for i in np.flatnonzero(valid):
                a=annotations[i]
                row=dict(annotation_id=int(ids[i]),image_id=int(images[i]),category_id=int(category[i]),
                         category_name=gt.cats[int(category[i])]["name"],area=a["area"],
                         reference_ar=float((old["segm_small_matches"][i]>0).mean()),
                         method_ar=float((new["segm_small_matches"][i]>0).mean()))
                for task in ["bbox","segm"]:
                    for ti,t in [(0,50),(5,75)]:
                        row[f"reference_{task}_match{t}"]=int(old[task+"_small_matches"][i,ti]>0)
                        row[f"method_{task}_match{t}"]=int(new[task+"_small_matches"][i,ti]>0)
                for key in ["best_box_iou","best_mask_iou","mask_iou_at_best_box",
                            "coverage_at_best_box","purity_at_best_box"]:
                    row["reference_"+key]=float(old[key][i])
                    row["method_"+key]=float(new[key][i])
                rows.append(row)
            csv_write(out/(branch+"__vs_"+reference+"__small_instances.csv"),rows)
    for name,rows in [("recall_contrasts",contrasts),("threshold_transitions",thresholds),
                      ("category_contributions",categories),("opportunity_transitions",mechanisms),
                      ("changed_instances",target_rows),("changed_instance_quality",spatial)]:
        csv_write(out/(name+".csv"),rows)
    save_json(out/"validation.json",validation)
    # Additivity: contributions from gained and lost threshold events reconstruct COCO AR exactly.
    for row in contrasts:
        if row["task"]=="segm":
            total=sum(r["ar_small_contribution_points"] for r in mechanisms
                      if r["branch"]==row["branch"] and r["contrast"]==row["contrast"])
            assert abs(total-row["delta_points"]) < 1e-10,(total,row)
    save_json(out/"COMPLETE.json",dict(status="complete",finished_at=now(),images=5000,
                                      noncrowd_gt=len(ids),source_evaluations=cfg["source_evaluations"],
                                      training=False,inference=False,
                                      validation="All 12 task evaluations reproduce 12 official metrics each; per-GT reconstruction and contribution additivity pass.",
                                      interpretation="Opportunity decomposition is observational within retained predictions; it does not identify pre-NMS loss, causal mechanism or candidate lineage across models."))
    print("ATTRIBUTION_COMPLETE",json.dumps(contrasts),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--study",type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.study/"protocol.json").read_text(encoding="utf-8"))
    out=a.study/"runs"/cfg["run_id"];out.mkdir(parents=True,exist_ok=False)
    snapshot=out/"execution_source";snapshot.mkdir()
    for source in [Path(__file__),Path(cm.__file__),a.study/"protocol.json"]:
        shutil.copy2(source,snapshot/source.name)
    record=dict(run_id=cfg["run_id"],study_id=cfg["study_id"],started_at=now(),
                status="running",command=sys.argv,protocol="../../protocol.json",training=False)
    save_json(out/"run.json",record)
    try:
        with quiet():gt=COCO(cfg["annotation"])
        assert len(gt.imgs)==5000
        ids=np.array(sorted(i for i,ann in gt.anns.items() if not ann.get("iscrowd",0)))
        category=np.array([gt.anns[int(i)]["category_id"] for i in ids])
        np.savez_compressed(out/"annotations.npz",annotation_ids=ids,category=category,
                            image_ids=np.array([gt.anns[int(i)]["image_id"] for i in ids]),
                            area=np.array([gt.anns[int(i)]["area"] for i in ids]))
        for branch in cfg["branches"]:
            for arm in cfg["source_evaluations"]:
                process_one(cfg,out,gt,ids,category,branch,arm)
        summarize(cfg,out,gt,ids,category)
        record.update(status="completed",finished_at=now(),returncode=0)
    except BaseException:
        record.update(status="failed",finished_at=now(),returncode=1,error=traceback.format_exc())
        raise
    finally:
        save_json(out/"run.json",record)


if __name__=="__main__":
    main()
