"""One-shot, immutable-cache frequency audit on the authorized laptop."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time, hashlib, socket, uuid, datetime, traceback, shutil, csv, io, contextlib
from pathlib import Path
from collections import defaultdict
import importlib.metadata as md

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT / "_site"))
import numpy as np
import torch
import psutil
from pycocotools.coco import COCO
from pycocotools import mask as masks
from spectral import spectrum, tests, BANDS
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

SOURCE = ROOT / "experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170"
GT = ROOT / "assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json"
WEIGHT = ROOT / "assets/models/coco_clean_20260911/yolo26m-seg.pt"
PRED = SOURCE / "trace/native/predictions.json"
LEDGER = SOURCE / "ledger/PER_GT_FAILURE.jsonl"
TRACE = SOURCE / "trace/TRACE_MANIFEST.json"
MODEL_SHA = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
GT_SHA = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
PRED_SHA = "a8c3e7d0af9db4684653c9b7b412f734d6105dc1e1c98a20c0782cf8d2fd3151"
LEDGER_SHA = "8e563c9a3f32e6a3289b6649685c8160f557d74f19b35209b9f712f79c2e5884"


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def lines(p):
    with Path(p).open(encoding="utf-8-sig") as f:
        return [json.loads(x) for x in f if x.strip()]


def image_root():
    candidates = [ROOT / "assets/datasets/coco_native_s0_20261013/images/val2017",
                  ROOT / "assets/datasets/coco_native_s0_20261013/full/images/val2017",
                  Path("D:/coco_wire/data/images/val2017"), Path("D:/coco_wire/data/val2017")]
    for p in candidates:
        if (p / "000000000139.jpg").is_file():
            return p
    # A dataset yaml may point at the shared immutable images.
    import yaml
    y = ROOT / "assets/datasets/coco_native_s0_20261013/full/coco.yaml"
    d = yaml.safe_load(y.read_text(encoding="utf-8-sig"))
    for root in [Path(d["path"]), y.parent]:
        p = root / "images/val2017"
        if (p / "000000000139.jpg").is_file():
            return p
    raise FileNotFoundError("COCO val2017 image root not found")


def sources():
    return [STUDY / "PROTOCOL.md", *sorted((STUDY / "scripts").glob("*.py")),
            WEIGHT, GT, PRED, LEDGER, TRACE, SOURCE / "EVAL_CONTRACT.json",
            SOURCE / "run.json", SOURCE / "ZERO_DELTA.json", SOURCE / "image_ids.json"]


def core_check():
    assert socket.gethostname().upper() == "YANGLEI", "wrong executor"
    for p, expected in [(WEIGHT, MODEL_SHA), (GT, GT_SHA), (PRED, PRED_SHA), (LEDGER, LEDGER_SHA)]:
        assert sha(p) == expected, str(p)
    assert read(SOURCE / "run.json")["execution_status"] == "completed"
    contract = read(SOURCE / "EVAL_CONTRACT.json")
    assert contract["ultralytics_version"] == "8.4.100"
    assert contract["end2end"] and not contract["agnostic_nms"]
    # Verify saved framework identity without invoking inference or touching GPU.
    sys.path.insert(0, str(ROOT / "shared/evaluation"))
    from native_output_bridge import native_environment
    environment = native_environment()
    manifest = read(TRACE)
    assert len(manifest["images"]) == 5000
    assert len(set(r["image_id"] for r in manifest["images"])) == 5000
    return manifest, environment


def freeze():
    assert not (STUDY / "FREEZE_MANIFEST.json").exists(), "already frozen"
    result = tests()
    manifest, env = core_check()
    imroot = image_root()
    cache_bytes = image_bytes = 0
    for r in manifest["images"]:
        p = SOURCE / "trace" / r["cache"]
        im = imroot / f"{r['image_id']:012d}.jpg"
        assert p.is_file() and im.is_file(), r["image_id"]
        cache_bytes += p.stat().st_size
        image_bytes += im.stat().st_size
    ledger = lines(LEDGER)
    ordinary = [r for r in ledger if not r["is_ignored"]]
    paired = [r for r in ordinary if r["bbox75_match_output_id"] is not None]
    assert len(ordinary) == 36335
    manifest_out = dict(frozen_at=now(), files={p.relative_to(ROOT).as_posix(): sha(p) for p in sources()},
        baseline_run_id=SOURCE.name, images=5000, ordinary_GT=len(ordinary),
        box75_pairs=len(paired), no_box75=len(ordinary)-len(paired),
        image_root=str(imroot), cache_bytes=cache_bytes, image_bytes=image_bytes,
        environment=env, analysis_packages={p: md.version(p) for p in
           ["numpy", "scipy", "matplotlib", "torch", "pycocotools", "psutil", "Pillow"]},
        synthetic_tests=result, source_commit=read(STUDY/"SOURCE_CONTEXT.json")["code_commit"],
        resources=dict(cpu_threads=2, rss_max_GiB=6, max_seconds=7200, free_disk_min_GiB=12))
    write(STUDY / "FREEZE_MANIFEST.json", manifest_out)
    s = read(STUDY / "study.json")
    s.update(plan_status="frozen", execution_status="not_started")
    write(STUDY / "study.json", s)
    print(json.dumps({k: manifest_out[k] for k in ["images", "ordinary_GT", "box75_pairs", "no_box75",
                                                 "cache_bytes", "image_root", "analysis_packages"]}), flush=True)


def verify_freeze():
    frozen = read(STUDY / "FREEZE_MANIFEST.json")
    for rel, expected in frozen["files"].items():
        assert sha(ROOT / rel) == expected, "frozen drift " + rel
    return frozen


def cropped_proto(cache):
    p = cache["proto"].numpy().astype(np.float64)
    gy, gx = cache["ratio_pad"][0]
    px, py = cache["ratio_pad"][1]
    h, w = cache["original_hw"]
    y0, y1 = int(np.ceil(py / 4)), int(np.floor((py + round(h * gy)) / 4))
    x0, x1 = int(np.ceil(px / 4)), int(np.floor((px + round(w * gx)) / 4))
    assert 0 <= y0 < y1 <= p.shape[1] and 0 <= x0 < x1 <= p.shape[2]
    return p[:, y0:y1, x0:x1], [y0, y1, x0, x1]


def aggregate(rows, image_rows, run):
    # Every confidence interval resamples whole images. Missing/zero energy stays undefined.
    groups = {}
    for name, rr in [("all", rows), ("failure", [r for r in rows if r["own_mask75"] == 0]),
                     ("success", [r for r in rows if r["own_mask75"] == 1])]:
        for size in ["all", "S", "M", "L"]:
            sub = rr if size == "all" else [r for r in rr if r["size_group"] == size]
            if not sub:
                continue
            by_im = defaultdict(list)
            for r in sub:
                by_im[r["image_id"]].append(r)
            ids = sorted(by_im)
            rng = np.random.default_rng(1729)
            draws = rng.integers(0, len(ids), size=(2000, len(ids)))
            g = dict(pairs=len(sub), images=len(ids), zero_error=sum(r["error_pixels"] == 0 for r in sub))
            for prefix in ["error", "error_hann", "gt", "pred", "logit"]:
                a = []
                for i in ids:
                    vv = np.array([[r[f"{prefix}_{b}_fraction"] for b in BANDS] for r in by_im[i]])
                    valid = np.isfinite(vv).all(1)
                    a.append(vv[valid].mean(0) if valid.any() else np.full(3, np.nan))
                a = np.array(a)
                with np.errstate(invalid="ignore"):
                    means = np.nanmean(a, 0)
                    sampled = np.nanmean(a[draws], 1)
                g[prefix] = {"image_macro_fraction": means.tolist(),
                             "ci95": np.nanquantile(sampled, [.025, .975], axis=0).T.tolist(),
                             "valid_images": int(np.isfinite(a[:, 0]).sum())}
            gt_energy = np.array([sum(r[f"gt_{b}_energy"] for r in sub) for b in BANDS])
            pred_energy = np.array([sum(r[f"pred_{b}_energy"] for r in sub) for b in BANDS])
            g["pooled_pred_gt_energy_ratio"] = (pred_energy / gt_energy).tolist()
            g["zero_gt_energy_pairs_by_band"] = [sum(r[f"gt_{b}_energy"] < 1e-20 for r in sub) for b in BANDS]
            g["error_energy_pooled_fraction_including_DC"] = (
                np.array([sum(r["error_dc"] for r in sub)] +
                         [sum(r[f"error_{b}_energy"] for r in sub) for b in BANDS]) /
                sum(r["error_pixels"] for r in sub)).tolist()
            g["mean_iou_pair_weighted"] = float(np.mean([r["iou"] for r in sub]))
            groups[name + "_" + size] = g
    proto = {}
    for prefix in ["proto", "proto_hann", "proto_energy_weighted"]:
        arr = np.array([r[prefix] for r in image_rows])
        rng = np.random.default_rng(1729)
        means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean(0) for _ in range(2000)])
        proto[prefix] = dict(image_macro_fraction=arr.mean(0).tolist(),
                            ci95=np.quantile(means, [.025, .975], axis=0).T.tolist())
    summary = dict(groups=groups, prototypes=proto, images=len(image_rows), paired_instances=len(rows),
        training_steps=0, new_model_forward=0, unit="energy_fraction_0_to_1",
        frequency_unit="cycles_per_640_input_pixels", band_edges=[0, 8, 32, "Nyquist_support"],
        baseline_run_id=SOURCE.name, run_id=run.name)
    write(run / "SUMMARY.json", summary)
    return summary


def execute(scope):
    frozen = verify_freeze()
    assert tests()["status"] == "passed"
    if scope == "full":
        smoke = read(STUDY / "SMOKE_POINTER.json")
        assert read(STUDY / "runs" / smoke["run_id"] / "run.json")["execution_status"] == "completed"
    run_id = "RUN_FREQ_" + scope.upper() + "_" + uuid.uuid4().hex
    run = STUDY / "runs" / run_id
    lock = STUDY / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(json.dumps(dict(pid=os.getpid(), run_id=run_id, host=socket.gethostname())))
    run.mkdir(parents=True)
    start = time.monotonic()
    meta = dict(study_id=read(STUDY / "study.json")["study_id"], run_id=run_id,
        stage_id=scope, executor=socket.gethostname(), pid=os.getpid(), started_at=now(),
        execution_status="running", scientific_status="unadjudicated", command=sys.argv,
        baseline_run_id=SOURCE.name, protocol_sha256=sha(STUDY / "PROTOCOL.md"),
        freeze_sha256=sha(STUDY / "FREEZE_MANIFEST.json"), code_commit=frozen["source_commit"],
        training_steps=0, new_model_forward=0, gpu_use=False)
    write(run / "run.json", meta)
    write(STUDY / "LATEST_RUN.json", dict(run_id=run_id, scope=scope))
    shutil.copy2(STUDY / "PROTOCOL.md", run / "PROTOCOL.md")
    contract = read(SOURCE / "EVAL_CONTRACT.json")
    write(run / "EVAL_CONTRACT.json", dict(source_contract=contract, analysis_protocol_sha=meta["protocol_sha256"],
        candidate_policy="fixed_output_detection", matching="unchanged_original_official_bbox75",
        frequency_unit="cycles_per_640_input_pixels", band_edges=[0, 8, 32],
        aggregation="within_image_instance_mean_then_equal_image_mean",
        bootstrap_replicates=2000, bootstrap_seed=1729, analysis_packages=frozen["analysis_packages"]))
    try:
        trace = read(TRACE)
        ims = sorted(trace["images"], key=lambda r: r["image_id"])
        if scope == "smoke":
            ims = ims[:16]
        ids = {r["image_id"] for r in ims}
        ledger = [r for r in lines(LEDGER) if r["image_id"] in ids]
        pairs = [r for r in ledger if not r["is_ignored"] and r["bbox75_match_output_id"] is not None]
        grouped = defaultdict(list)
        for r in pairs:
            grouped[r["image_id"]].append(r)
        print("LOAD_PREDICTIONS", run_id, flush=True)
        predictions = read(PRED)
        needed = {int(r["bbox75_match_output_id"].split(":")[-1]) for r in pairs}
        pred = {i: p for i, p in enumerate(predictions) if i in needed}
        assert len(predictions) == 539128
        del predictions
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(str(GT))
        rows, image_rows = [], []
        radial_sums = {k: [] for k in ["proto", "logit", "gt", "pred", "error"]}
        channel_sums = np.zeros((32, 3))
        audit = dict(images=0, pairs=0, image_hash_checks=0, cache_hash_checks=0,
                     max_iou_error=0., max_parseval_relative_error=0., peak_rss_bytes=0, status="running")
        imroot = Path(frozen["image_root"])
        for num, rec in enumerate(ims, 1):
            assert time.monotonic() - start < 7200, "wall clock budget"
            assert shutil.disk_usage(STUDY).free > 12 * 1024**3, "free disk budget"
            rss = psutil.Process().memory_info().rss
            audit["peak_rss_bytes"] = max(audit["peak_rss_bytes"], rss)
            assert rss < 6 * 1024**3, "RSS budget"
            iid = rec["image_id"]
            cachep = SOURCE / "trace" / rec["cache"]
            imp = imroot / f"{iid:012d}.jpg"
            assert sha(cachep) == rec["cache_sha256"], "cache SHA"
            assert sha(imp) == rec["image_sha256"], "image SHA"
            cache = torch.load(cachep, map_location="cpu", weights_only=True)
            assert cache["image_id"] == iid and cache["prediction_indices"] == rec["prediction_indices"]
            assert cache["input_hw"] == rec["input_hw"] and cache["original_hw"] == rec["original_hw"]
            p, crop = cropped_proto(cache)
            ps = spectrum(p, (4, 4))
            ph = spectrum(p, (4, 4), True)
            assert np.isfinite(ps["fraction"]).all() and np.isfinite(ph["fraction"]).all()
            prow = dict(image_id=iid, proto=ps["fraction"].mean(0).tolist(),
                        proto_hann=ph["fraction"].mean(0).tolist(),
                        proto_energy_weighted=(ps["energy"].sum(0)/ps["total"].sum()).tolist(),
                        prototype_hw=list(p.shape[1:]), valid_crop=crop,
                        original_hw=rec["original_hw"], ratio_pad=cache["ratio_pad"],
                        radial_frequency_counts=ps["counts"].tolist())
            image_rows.append(prow)
            channel_sums += ps["fraction"]
            radial_sums["proto"].append(ps["radial"].mean(0))
            current_radial = defaultdict(list)
            ranks = {index: rank for rank, index in enumerate(cache["prediction_indices"])}
            for item in grouped[iid]:
                index = int(item["bbox75_match_output_id"].split(":")[-1])
                assert item["bbox75_match_output_id"].startswith(PRED_SHA + ":")
                prediction = pred[index]
                ann = coco.anns[item["annotation_id"]]
                assert prediction["image_id"] == ann["image_id"] == iid
                assert prediction["category_id"] == ann["category_id"] == item["gt_category_id"]
                g = coco.annToMask(ann).astype(np.float64)
                m = masks.decode(prediction["segmentation"]).astype(np.float64)
                assert list(m.shape) == rec["original_hw"] == list(g.shape)
                union = np.logical_or(g, m).sum()
                iou = float(np.logical_and(g, m).sum() / union) if union else 1.
                difference = abs(iou - item["mask_iou_of_bbox_match"])
                assert difference <= 1e-12, ("IoU", iid, item["annotation_id"], difference)
                audit["max_iou_error"] = max(audit["max_iou_error"], difference)
                e = m - g
                spacing = tuple(map(float, cache["ratio_pad"][0]))
                # Linear response only; official binary masks are immutable source RLE.
                z = np.einsum("k,khw->hw", cache["c0"][ranks[index]].numpy().astype(np.float64), p)
                specs = {name: spectrum(value, spacing if name != "logit" else (4, 4))
                         for name, value in [("gt", g), ("pred", m), ("error", e), ("logit", z)]}
                hs = {name+"_hann": spectrum(value, spacing if name != "logit" else (4, 4), True)
                      for name, value in [("gt", g), ("pred", m), ("error", e), ("logit", z)]}
                error_pixels = int((e ** 2).sum())
                assert abs(specs["error"]["total"][0]+specs["error"]["dc"][0]-error_pixels) <= 1e-7
                row = dict(image_id=iid, annotation_id=ann["id"], output_detection_id=item["bbox75_match_output_id"],
                    prediction_index=index, raw_id=int(cache["raw_ids"][ranks[index]]), category_id=ann["category_id"],
                    category=coco.cats[ann["category_id"]]["name"], gt_area=float(ann["area"]),
                    size_group="S" if ann["area"] < 32**2 else "M" if ann["area"] < 96**2 else "L",
                    iou=iou, own_mask75=int(iou >= .75), task_mask75=item["task_mask75"],
                    error_pixels=error_pixels, error_dc=float(specs["error"]["dc"][0]))
                for name, spec in {**specs, **hs}.items():
                    audit["max_parseval_relative_error"] = max(audit["max_parseval_relative_error"], spec["parseval"])
                    for b, band in enumerate(BANDS):
                        row[f"{name}_{band}_energy"] = float(spec["energy"][0, b])
                        row[f"{name}_{band}_fraction"] = float(spec["fraction"][0, b])
                for name in specs:
                    if np.isfinite(specs[name]["radial"]).all():
                        current_radial[name].append(specs[name]["radial"][0])
                rows.append(row)
            for name in ["logit", "gt", "pred", "error"]:
                if current_radial[name]:
                    radial_sums[name].append(np.mean(current_radial[name], 0))
            assert sha(cachep) == rec["cache_sha256"]
            assert sha(imp) == rec["image_sha256"]
            audit["image_hash_checks"] += 2
            audit["cache_hash_checks"] += 2
            audit["images"], audit["pairs"] = num, len(rows)
            if num % 50 == 0 or num == len(ims):
                elapsed = time.monotonic()-start
                write(run / "PROGRESS.json", dict(images=num, total_images=len(ims), pairs=len(rows),
                                                elapsed_seconds=elapsed, eta_seconds=elapsed/num*(len(ims)-num)))
                print(f"PROGRESS {num}/{len(ims)} pairs={len(rows)} elapsed={elapsed:.1f}s", flush=True)
        assert len(rows) == len(pairs)
        with (run / "PER_INSTANCE.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write(run / "PER_IMAGE.json", image_rows)
        write(run / "RADIAL.json", dict(bin_edges=np.linspace(0,160,49).tolist(), tail=">160",
            curves={k: np.mean(v,0).tolist() for k,v in radial_sums.items()},
            images_per_curve={k: len(v) for k,v in radial_sums.items()}))
        write(run / "PROTOTYPE_CHANNELS.json", dict(fractions=(channel_sums/len(ims)).tolist(), images=len(ims)))
        summary = aggregate(rows, image_rows, run)
        selections = []
        for size in ["S", "M", "L"]:
            for status in [0, 1]:
                candidates = sorted([r for r in rows if r["size_group"] == size and r["own_mask75"] == status],
                                    key=lambda r: (r["iou"], r["image_id"], r["annotation_id"]))
                if candidates:
                    r = candidates[len(candidates)//2]
                    selections.append({k: r[k] for k in ["image_id","annotation_id","prediction_index","size_group","own_mask75","iou","category"]})
        write(run / "EXAMPLES.json", selections)
        verify_freeze()
        audit.update(status="passed", source_files_unchanged=True, elapsed_seconds=time.monotonic()-start)
        write(run / "CONFIGURATION_REVIEW.json", audit)
        f = summary["groups"]["failure_all"]["error"]
        ci = f["ci95"][2]
        ratios = [v["pooled_pred_gt_energy_ratio"][2] for k,v in summary["groups"].items()
                  if k in ["failure_S","failure_M","failure_L"]]
        decision = ("high_frequency_priority" if ci[0] > .5 and all(v < 1 for v in ratios)
                    else "low_mid_frequency_priority" if ci[1] < .5 else "scale_conditioned_priority")
        write(run / "DECISION.json", dict(decision=decision, scientific_status="descriptive_not_causal",
            high_frequency_fraction=f["image_macro_fraction"][2], high_frequency_ci95=ci,
            method_benefit="not_applicable_no_intervention", next_experiment_started=False))
        meta.update(execution_status="completed", scientific_status="descriptive_not_causal",
                    ended_at=now(), exit_code=0, elapsed_seconds=time.monotonic()-start)
        write(run / "run.json", meta)
        if scope == "smoke":
            write(STUDY / "SMOKE_POINTER.json", dict(run_id=run_id, audit_status="passed"))
        s = read(STUDY/"study.json")
        s.update(execution_status="smoke_passed" if scope=="smoke" else "completed",
                 scientific_status="engineering_only" if scope=="smoke" else "descriptive_not_causal",
                 latest_run_id=run_id, artifact_status="numerical_complete", transfer_status="pending")
        write(STUDY/"study.json",s)
        print("COMPLETED", run_id, json.dumps(audit), flush=True)
    except BaseException:
        (run / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        meta.update(execution_status="failed", scientific_status="engineering_failure", ended_at=now(), exit_code=1)
        write(run / "run.json", meta)
        raise
    finally:
        if lock.exists() and read(lock).get("pid") == os.getpid():
            lock.unlink()


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "freeze":
        freeze()
    elif action in ["smoke", "full"]:
        execute(action)
    else:
        raise ValueError(action)
