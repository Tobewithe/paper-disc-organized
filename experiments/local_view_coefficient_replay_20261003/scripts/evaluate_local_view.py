"""Zero-training local-view/old-OGPS replay on the fixed screen dev candidates."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

if "--config" in sys.argv:
    _cfg = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text(encoding="utf-8-sig"))
    sys.path.insert(0, _cfg["source_python"])

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops

from online_runtime import dump, load_asset, load_index, load_json, resolve_runtime_config, sha256, tensor_sha
from local_view_source import LocalViewSource
from ogps_solver import (crop_pool_7o, pooled_design_matrix, solve_ogps, solve_ogps_logits,
                         RIDGE_LAMBDA, PROBABILITY_MIN, PROBABILITY_MAX)
import evaluation_metrics as em

ARMS = ("A", "BASE_SOLVE", "ORIG_RESELECT", "ORIG_RESELECT_SOLVE", "VIEW_FULL", "VIEW_GRID", "VIEW_SOLVE")
METRICS = ("iou", "mask75", "coverage", "auc", "fpr")
PAIRS = (("VIEW_SOLVE", "A"), ("VIEW_SOLVE", "BASE_SOLVE"), ("VIEW_SOLVE", "VIEW_FULL"),
         ("VIEW_SOLVE", "VIEW_GRID"), ("VIEW_SOLVE", "ORIG_RESELECT_SOLVE"),
         ("VIEW_FULL", "ORIG_RESELECT"), ("ORIG_RESELECT", "A"),
         ("ORIG_RESELECT_SOLVE", "A"), ("BASE_SOLVE", "A"), ("VIEW_FULL", "A"), ("VIEW_GRID", "A"))
ALLOWLIST = ("image_id", "source_split", "input_shape", "original_shape", "ratio_pad", "boxes", "proto", "c0", "input_uint8", "raw_ids")
em.ARMS = ARMS


def key(row):
    return tuple(row[n] for n in ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx"))


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Fixed 3600-second replay budget reached; report incomplete, never shrink membership")


def historical_A(path, items):
    required = {int(r["image_id"]) for r in items}
    result = {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            r = json.loads(line)
            if r["split"] != "dev" or int(r["image_id"]) not in required:
                continue
            identity = key(r)
            if identity in result:
                raise AssertionError("Duplicate original screen candidate")
            result[identity] = r
    if len(result) != sum(int(r["n"]) for r in items):
        raise AssertionError("Original screen A reference does not cover the fixed population")
    return result


def grid_to_canvas(target_logits, box, device):
    """Exact old direct renderer: pooled target LOGITS interpolated into box."""
    x1 = max(0, min(639, math.floor(float(box[0]))))
    y1 = max(0, min(639, math.floor(float(box[1]))))
    x2 = max(x1 + 1, min(640, math.ceil(float(box[2]))))
    y2 = max(y1 + 1, min(640, math.ceil(float(box[3]))))
    patch = F.interpolate(target_logits.reshape(1, 1, 8, 8).float(), (y2-y1, x2-x1),
                          mode="bilinear", align_corners=False)[0, 0]
    canvas = torch.full((640, 640), -30., device=device)
    canvas[y1:y2, x1:x2] = patch
    return canvas


@torch.no_grad()
def add_direct(row, name, logits, image, candidate_index, coco):
    """Fixed original predicted support and the exact same original-image inverse."""
    if logits.shape != (640, 640) or not torch.isfinite(logits).all():
        raise AssertionError("Direct arm must return finite continuous 640x640 logits")
    device = logits.device
    box = image["boxes"][candidate_index].to(device)
    support = ops.crop_mask(torch.ones((1, 640, 640), device=device), box[None])[0].bool()
    binary = ((logits > 0) & support).byte()[None]
    mask = em._scale_binary(binary, tuple(image["original_shape"]), image["ratio_pad"])[0]
    ann = coco.anns[row["annotation_id"]]
    if int(ann["image_id"]) != int(image["image_id"]):
        raise AssertionError("Original COCO identity changed")
    truth = torch.as_tensor(coco.annToMask(ann).astype(bool), device=device)
    padded_truth = em._padded_gt(truth, image["ratio_pad"], (640, 640))
    inter = int((mask & truth).sum())
    row[f"iou_{name}"] = inter / max(1, int((mask | truth).sum()))
    row[f"coverage_{name}"] = inter / max(1, int(truth.sum()))
    row[f"mask75_{name}"] = int(row[f"iou_{name}"] >= .75)
    row[f"auc_{name}"], row[f"fpr_{name}"] = em._pixel_auc_fpr(logits, padded_truth, support)
    row[f"empty_mask_{name}"] = not bool(mask.any())
    # Binary crop equivalence matters at fractional box edges.
    manual = ops.crop_mask(logits[None].clone(), box[None]).gt(0).byte()
    if not torch.equal(manual, binary):
        raise AssertionError("Fixed-box direct threshold/crop ordering differs")


def summarize(rows, out, info):
    predicates = dict(all=lambda r: True,
        box_good_mask_bad=lambda r: r["box_good_mask_bad"],
        original_success=lambda r: bool(r["mask75_A"]),
        original_failure=lambda r: not bool(r["mask75_A"]))
    tables, per_image = {}, []
    for name, predicate in predicates.items():
        selected = [r for r in rows if predicate(r)]
        groups = em.image_groups(selected)
        table = dict(images=len(groups), candidates=len(selected), candidate={}, image_macro={}, undefined={}, comparisons={})
        for metric in METRICS:
            table["candidate"][metric] = {a: em.avg([r.get(f"{metric}_{a}") for r in selected]) for a in ARMS}
            table["image_macro"][metric] = {a: em.avg([em.avg([r.get(f"{metric}_{a}") for r in g]) for g in groups]) for a in ARMS}
            table["undefined"][metric] = {a: sum(not em.finite(r.get(f"{metric}_{a}")) for r in selected) for a in ARMS}
        for arm, ref in PAIRS:
            comparison = {m: em.paired(groups, arm, ref, m, 20261003, 1000) for m in METRICS}
            repair = sum(r[f"mask75_{ref}"] == 0 and r[f"mask75_{arm}"] == 1 for r in selected)
            damage = sum(r[f"mask75_{ref}"] == 1 and r[f"mask75_{arm}"] == 0 for r in selected)
            comparison["crossings"] = dict(repair=repair, damage=damage, net=repair-damage)
            table["comparisons"][f"{arm}_minus_{ref}"] = comparison
        tables[name] = table
        for group in groups:
            item = dict(split="dev", image_id=group[0]["image_id"], group=name, candidates=len(group))
            for metric in METRICS:
                for arm in ARMS:
                    item[f"{metric}_{arm}"] = em.avg([r.get(f"{metric}_{arm}") for r in group])
                    item[f"{metric}_{arm}_defined"] = sum(em.finite(r.get(f"{metric}_{arm}")) for r in group)
                for arm, ref in PAIRS:
                    item[f"delta_{metric}_{arm}_minus_{ref}"] = em.avg([
                        r[f"{metric}_{arm}"]-r[f"{metric}_{ref}"] for r in group
                        if em.finite(r.get(f"{metric}_{arm}")) and em.finite(r.get(f"{metric}_{ref}"))])
            per_image.append(item)
    em.append_rows(out / "PER_IMAGE.jsonl", per_image)
    result = dict(schema="local-view-fixed-prototype-replay-v1", arms=list(ARMS), population=info["population"],
        tables=tables, evaluation=info, bootstrap=dict(draws=1000, unit="paired image", seed=20261003),
        training=False, no_new_blind_test=True, no_COCO_AP=True, automatic_followup=False,
        inference_scope="Fixed official GT-conditioned diagnostic candidates; no full deployment/AP claim")
    dump(out / "SUMMARY.json", em.clean(result))
    lines = ["# Frozen local-view coefficient replay", "",
        "Zero training. Frozen local-view and original-view reselection sources supply responses to direct rendering and the unchanged OGPS solver. Original P, target c0, predicted boxes and evaluation candidate identities remain fixed.", "",
        f"{'SMOKE ONLY: ' if info['smoke'] else ''}{info['population']['planned_images']} planned dev images, {tables['all']['images']} effective images, {len(rows)} candidates. These images were previously viewed; this is exploratory reuse, not a new blind test or COCO AP.", "",
        "All deltas below are percentage points. Macro estimates and confidence intervals use the same image-equal statistic; candidate intervals in SUMMARY also resample whole images. The 1000-bootstrap intervals are descriptive across these multiple exploratory comparisons.", "",
        "| Group | Comparison | Macro IoU Δ [95% CI] | Candidate IoU Δ | Net Mask75 | Repair/damage | Coverage Δ | AUC Δ | FPR Δ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def pp(v): return "undefined" if v is None else f"{v*100:+.4f}"
    for name, table in tables.items():
        for pair, record in table["comparisons"].items():
            d, crossing = record["iou"]["image_macro"], record["crossings"]
            lo, hi = d["ci95"]
            lines.append(f"| {name} | {pair} | {pp(d['delta'])} [{pp(lo)}, {pp(hi)}] | {pp(record['iou']['candidate']['delta'])} | {crossing['net']} | {crossing['repair']}/{crossing['damage']} | {pp(record['coverage']['image_macro']['delta'])} | {pp(record['auc']['image_macro']['delta'])} | {pp(record['fpr']['image_macro']['delta'])} |")
    lines += ["", "Arm definitions:", "",
        "- A: original full prototype and original coefficient, with the original normal-image decoder.",
        "- BASE_SOLVE: sigmoid(A8 c0) passed through the formal probability clipping and fixed ridge solver. Clipping can change c0, so this is not an identity arm.",
        "- VIEW_FULL: local-view continuous responses reprojected to the original 640 input, with source-defined baseline fill outside observed support; fixed original box crop and identical original-image inverse.",
        "- VIEW_GRID: pool sigmoid(VIEW_FULL) using the legacy 8×8 floor/ceil crop, form t8=logit(clamp(q8,.01,.99)), resize these logits into the same integer box, then crop/threshold and restore the original image.",
        "- VIEW_SOLVE: exactly the same q8/t8 passed through the unchanged FP64 ridge (lambda=.003) into the original full prototype. No local-view coefficients or prototypes replace original P/c0.", "",
        "- ORIG_RESELECT: reselect from the frozen original-image official top300 by predicted-box IoU, then source score and raw ID; decode that source coefficient with original full P, while retaining the target candidate's box and identity.",
        "- ORIG_RESELECT_SOLVE: the same reselected source's full 640 input logits are transformed by sigmoid, the same 8x8 probability pooling, clipping and solver as VIEW_SOLVE; the regularization center stays the original target c0.", "",
        "VIEW_FULL versus VIEW_GRID changes pooling, clipping and rendering jointly; it is not a pure resolution experiment. VIEW_SOLVE versus VIEW_GRID compares prototype projection with direct rendering of the same target. VIEW_SOLVE versus BASE_SOLVE checks added local-view information against self-projection/calibration effects.", "",
        "VIEW_SOLVE versus ORIG_RESELECT_SOLVE and VIEW_FULL versus ORIG_RESELECT control original-image source reselection. A local-view gain over A alone cannot distinguish image-view effects from changing the source raw position. Source selection is GT-free in both views; this comparison still includes differences in source predictions induced by the view.", "",
        "Raw class/score or GT never selects an evaluation subset after replay. Every fixed candidate remains, including empty masks and worsening predictions. GT is supplied only to metrics; the source receives an explicit GT-free field allowlist. Original successes/failures and the box-good/mask-bad stratum are defined by A.", "",
        "AUC uses continuous input-space logits with original COCO labels on the fixed original predicted-box support. Undefined AUC entries are counted, not removed from IoU. Model execution completed is not a scientific mechanism claim; no further training is launched.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def run(args):
    started = time.monotonic(); deadline = started + 3600
    cfg = resolve_runtime_config(load_json(args.config))
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Run only on the authorized Linux GPU server")
    if RIDGE_LAMBDA != .003 or (PROBABILITY_MIN, PROBABILITY_MAX) != (.01, .99):
        raise AssertionError("Legacy solver configuration changed")
    if int(cfg.get("max_seconds", 3600)) != 3600 or int(cfg.get("bootstrap", 1000)) != 1000:
        raise ValueError("Fixed 3600 seconds / 1000 image bootstrap required")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "PER_CANDIDATE.jsonl").exists() or (out / "COMPLETE.json").exists():
        raise RuntimeError("Use an independent Run; retain prior output")
    index = load_index(cfg)
    if len(index["dev"]) != 256:
        raise AssertionError("Fixed screen dev slice must contain 256 planned images")
    planned_dev = list(map(int, load_json(cfg["split"])["dev"]))
    if [int(r["image_id"]) for r in index["dev"]] != planned_dev:
        raise AssertionError("Cache INDEX deviates from the locked dev image order")
    items = [r for r in index["dev"] if r["n"]][:2] if args.smoke else index["dev"]
    population = dict(planned_images=len(items), effective_images=sum(bool(r["n"]) for r in items),
        candidates=sum(int(r["n"]) for r in items), no_positive_images=[r["image_id"] for r in items if not r["n"]])
    if len({r["image_id"] for r in items}) != len(items):
        raise AssertionError("Duplicate dev image")
    baseline_path = Path(cfg["original_screen_results"])
    previous = historical_A(baseline_path, items)
    inputs = dict(config_sha256=sha256(args.config), cache_identity=load_json(Path(cfg["cache"]) / "CACHE_IDENTITY.json"),
        cache_identity_sha256=sha256(Path(cfg["cache"]) / "CACHE_IDENTITY.json"),
        cache_index_sha256=sha256(Path(cfg["cache"]) / "INDEX.json"),
        split_sha256=sha256(cfg["split"]),
        original_weights_sha256=sha256(cfg["weights"]), baseline_reference_sha256=sha256(baseline_path),
        source_script_sha256=sha256(Path(__file__).with_name("local_view_source.py")),
        solver_script_sha256=sha256(Path(__file__).with_name("ogps_solver.py")),
        source_GT_free_allowlist=list(ALLOWLIST), no_training=True)
    dump(out / "INPUT_MANIFEST.json", inputs)
    dump(out / "EVALUATION_INDEX.json", dict(dev=items, smoke=args.smoke, population=population))
    source = LocalViewSource(cfg)
    dump(out / "ENVIRONMENT.json", source.import_info)
    if source.weights_sha256 != inputs["original_weights_sha256"]:
        raise AssertionError("Local-view source uses another weight")
    geometry = source.geometry_test_cases()
    if geometry.get("passed") is not True:
        raise AssertionError("Local-view geometry check failed")
    dump(out / "GEOMETRY_AUDIT.json", geometry)
    coco = COCO(str(cfg["annotations_train"]))
    rows, seen, numeric = [], set(), []
    max_reference_error = 0.
    for position, entry in enumerate(items):
        check_time(deadline)
        if not entry["n"]:
            continue
        x = load_asset(cfg, int(entry["image_id"]), verify=True)
        if x["split"] != "dev" or len(x["rows"]) != int(entry["n"]):
            raise AssertionError("Fixed image/candidate population changed")
        passed = {k: x[k] for k in ALLOWLIST}
        proto = x["proto"].to("cuda"); c0 = x["c0"].to("cuda"); boxes = x["boxes"].to("cuda")
        matrix = pooled_design_matrix(proto, boxes).double()
        base_logits = (matrix @ c0.double().unsqueeze(-1)).squeeze(-1)
        identity_solution = solve_ogps_logits(matrix, c0, base_logits)
        identity_error = float(identity_solution["delta"].abs().max())
        if identity_error != 0.:
            raise AssertionError("Unclipped logit self-target did not give exact zero coefficient increment")
        baseline = solve_ogps(matrix, c0, base_logits.sigmoid())
        if float(baseline["diagnostics"]["normal_equation_relative_backward_error"].max()) > 1e-10:
            raise ArithmeticError("BASE_SOLVE failed the same fixed backward-error tolerance")
        produced, view_coefficients, direct_rows = [], [], []
        reselected_coefficients, reselected_solve_coefficients = [], []
        for view in source.infer(passed, deadline_monotonic=deadline):
            check_time(deadline)
            j = int(view["candidate_index"])
            if j != len(produced) or j >= len(x["rows"]):
                raise AssertionError("Local source changed candidate order/membership")
            z = view["logit_input"].to("cuda").float()
            if z.shape != (640, 640) or not torch.isfinite(z).all():
                raise AssertionError("Nonfinite or wrong-shaped local-view logits")
            q8 = crop_pool_7o(z.sigmoid()[None], boxes[j]).reshape(8, 8)
            solved = solve_ogps(matrix[j], c0[j], q8)
            if float(solved["diagnostics"]["normal_equation_relative_backward_error"]) > 1e-10:
                raise ArithmeticError("Fixed FP64 solver failed its backward-error tolerance")
            target = solved["target_logits"]
            q_clipped = q8.double().clamp(.01, .99)
            torch.testing.assert_close(target, (q_clipped.log()-torch.log1p(-q_clipped)).flatten(), atol=0, rtol=0)
            reselected = view["same_view_coeff"].detach().to("cuda").float()
            if reselected.shape != (32,) or not torch.isfinite(reselected).all():
                raise AssertionError("Original-image reselection must supply a finite 32D coefficient")
            original_source_logits = em._logits(proto, reselected[None], (640, 640))[0]
            original_q8 = crop_pool_7o(original_source_logits.sigmoid()[None], boxes[j]).reshape(8, 8)
            original_solved = solve_ogps(matrix[j], c0[j], original_q8)
            if float(original_solved["diagnostics"]["normal_equation_relative_backward_error"]) > 1e-10:
                raise ArithmeticError("ORIG_RESELECT_SOLVE failed the fixed FP64 backward-error tolerance")
            original_clipped = original_q8.double().clamp(.01, .99)
            torch.testing.assert_close(original_solved["target_logits"],
                (original_clipped.log()-torch.log1p(-original_clipped)).flatten(), atol=0, rtol=0)
            # Retain only scalar direct metrics and 32D coefficients. Decode
            # coefficient arms per complete image with the original chunk=4,
            # so A uses the exact historical GEMM/decode batching as well.
            decoded = {"annotation_id": x["rows"][j]["annotation_id"]}
            add_direct(decoded, "VIEW_FULL", z, x, j, coco)
            grid_canvas = grid_to_canvas(target, boxes[j], z.device)
            add_direct(decoded, "VIEW_GRID", grid_canvas, x, j, coco)
            decoded["local_view"] = view["metadata"]
            decoded["original_view_reselection"] = view["same_view_source_meta"]
            decoded["source_input_sha256"] = tensor_sha(x["input_uint8"])
            decoded["source_asset_sha256"] = x["_asset_integrity"]["compressed_sha256"]
            decoded["source_observed_support_pixels"] = int(view["support_input"].sum())
            decoded["solver"] = {name: float(value) for name, value in solved["diagnostics"].items()}
            decoded["original_reselection_solver"] = {name: float(value) for name, value in original_solved["diagnostics"].items()}
            decoded["base_solver"] = {name: float(value[j]) for name, value in baseline["diagnostics"].items()}
            decoded["unclipped_self_delta_max_abs"] = identity_error
            direct_rows.append(decoded); view_coefficients.append(solved["coeff"].float()); produced.append(j)
            reselected_coefficients.append(reselected)
            reselected_solve_coefficients.append(original_solved["coeff"].float())
        if len(produced) != int(entry["n"]):
            raise AssertionError("Local-view source dropped a candidate")
        check_time(deadline)
        gpu = dict(x, proto=proto, c0=c0, boxes=boxes)
        coeff = dict(A=c0, BASE_SOLVE=baseline["coeff"].float(), VIEW_SOLVE=torch.stack(view_coefficients),
                     ORIG_RESELECT=torch.stack(reselected_coefficients),
                     ORIG_RESELECT_SOLVE=torch.stack(reselected_solve_coefficients))
        normal_rows = em.evaluate_image(gpu, coeff, coco, chunk_size=4)
        for j, decoded in enumerate(normal_rows):
            k = key(decoded)
            if k != key(x["rows"][j]) or k in seen or k not in previous:
                raise AssertionError("Candidate identity changed or is duplicated")
            old = previous[k]
            for metric in METRICS:
                actual, prior = decoded[f"{metric}_A"], old[f"{metric}_A"]
                if em.finite(actual) != em.finite(prior):
                    raise AssertionError("Baseline metric definedness changed")
                if em.finite(actual):
                    difference = abs(actual-prior); max_reference_error = max(max_reference_error, difference)
                    if difference > 1e-12:
                        raise AssertionError(f"Original A {metric} does not reproduce screen reference: {difference}")
            decoded.update(direct_rows[j])
            em.append_rows(out / "PER_CANDIDATE.jsonl", [decoded])
            rows.append(decoded); seen.add(k)
        numeric.append(dict(image_id=x["image_id"], candidates=len(produced), unclipped_identity_error=identity_error,
                            baseline_reference_max_error=max_reference_error))
        progress = dict(images_processed=position+1, planned_images=len(items), candidates=len(rows),
                        elapsed_s=time.monotonic()-started, training=False)
        dump(out / "PROGRESS.json", progress); print(json.dumps(progress), flush=True)
        del x, matrix, baseline, proto, c0, boxes
    if len(rows) != population["candidates"] or seen != set(previous):
        raise AssertionError("Fixed population evaluation incomplete")
    check_time(deadline); source.assert_unchanged()
    info = dict(smoke=args.smoke, population=population, no_training=True, source=source.import_info,
        original_source_unchanged=True, baseline_reference_max_error=max_reference_error,
        solver_lambda=.003, solver_dtype="float64", target_probability_clip=[.01,.99],
        grid_renderer="same clipped target logits as solver, bilinear to floor/ceil box, original support and inverse",
        no_new_blind_test=True, no_COCO_AP=True, automatic_followup=False, inputs=inputs)
    dump(out / "NUMERICAL_AUDIT.json", numeric); dump(out / "EVALUATION_AUDIT.json", info)
    summarize(rows, out, info); check_time(deadline)
    dump(out / "COMPLETE.json", dict(completed=True, passed=True, smoke=args.smoke,
        candidates=len(rows), training=False, scientific_scope="exploratory replay; not independent confirmation",
        elapsed_s=time.monotonic()-started, automatic_followup=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        payload = dict(completed=False, passed=False, error=repr(exc), traceback=traceback.format_exc(),
                       incomplete=True, timeout=isinstance(exc, TimeoutError), no_training=True)
        dump(Path(args.out) / "FAILURE.json", payload); dump(Path(args.out) / "COMPLETE.json", payload)
        raise


if __name__ == "__main__":
    main()
