"""Frozen U-checkpoint evidence-path audit; no training or parameter selection.

FULL, NATIVE, MEAN and ROLL share exactly the epoch-3 U model's native cv4.
MEAN removes variation between its 64 evidence values; it is NOT a global
mask bias because the unchanged prototype operator K is spatially dependent.
ROLL permutes those values, not K or the candidate's frozen geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

if "--config" in sys.argv:
    _cfg = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text(encoding="utf-8-sig"))
    if _cfg.get("source_python"):
        sys.path.insert(0, _cfg["source_python"])

import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics.utils import ops

from online_runtime import (
    FrozenReplay, dump, load_asset, load_index, load_json,
    resolve_runtime_config, sha256, tensor_sha,
)
from screen_models import build_model, build_operators
from point_selectors import UNIFORM_INDICES
from evaluate_screen import official_bce_values, official_gpu_payload
from runtime_utils import setup
import evaluation_metrics as em

ARMS = ("A", "FULL", "NATIVE", "MEAN", "ROLL")
METRICS = ("iou", "mask75", "coverage", "auc", "fpr", "bce")
PAIRS = tuple((arm, "A") for arm in ARMS[1:]) + tuple(
    ("FULL", arm) for arm in ("NATIVE", "MEAN", "ROLL")
)
IDENTITY_KEYS = (
    "split", "image_id", "annotation_id", "branch", "raw_id",
    "pyramid_level", "target_gt_idx",
)
DIAGNOSTICS = (
    "evidence_mean", "evidence_std", "evidence_rms", "spatial_energy_fraction",
    "native_delta_l2", "extra_delta_l2", "mean_extra_delta_l2", "roll_extra_delta_l2",
    "native_logit_rms16", "extra_logit_rms16", "mean_extra_logit_rms16",
    "roll_extra_logit_rms16", "full_delta_logit_rms16",
    "evidence_saturation_rate", "mean_center_logit_inner_product16",
    "center_logit_rms16", "extra_logit_rms640", "mean_extra_logit_rms640",
    "center_logit_rms640", "roll_extra_logit_rms640", "mean_center_logit_inner_product640",
)
em.ARMS = ARMS


def identity(row):
    return tuple(row[k] for k in IDENTITY_KEYS)


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        v = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(v.dtype).encode("ascii"))
        digest.update(str(tuple(v.shape)).encode("ascii"))
        digest.update(v.numpy().tobytes())
    return digest.hexdigest()


def assert_frozen(model):
    if model.training or any(module.training for module in model.modules()):
        raise AssertionError("Every checkpoint module must remain in evaluation mode")
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError("Frozen audit must neither require nor accumulate gradients")


def distribution(values):
    a = np.asarray([v for v in values if em.finite(v)], dtype=np.float64)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                quantiles_0_25_50_75_100=np.quantile(a, [0, .25, .5, .75, 1]).tolist() if len(a) else [])


def make_interventions(full, residual, current, selected, c0):
    """GT-free operations only; no labels, annotation IDs or quality groups."""
    native = current["c_current"]
    n = len(native)
    if residual.shape != (n, 64) or selected["K"].shape != (n, 32, 64):
        raise AssertionError("The existing U model must produce 64 evidence slots")
    expected = torch.tensor(UNIFORM_INDICES, device=residual.device, dtype=torch.long).expand(n, -1)
    if not torch.equal(selected["indices"], expected):
        raise AssertionError("Only the frozen row-major U grid can be reshaped to 8x8")
    if any(v.requires_grad for v in (full, residual, native, selected["K"])):
        raise AssertionError("The evidence audit must not build a gradient graph")
    k = selected["K"].float()
    r = residual.float()
    if not bool(torch.isfinite(r).all()):
        raise FloatingPointError("Nonfinite original evidence")
    correction = torch.bmm(k, r.unsqueeze(2)).squeeze(2)
    reconstructed = native + correction.to(native.dtype)
    torch.testing.assert_close(full, reconstructed, atol=3e-5, rtol=3e-5)
    rolled = r.reshape(n, 8, 8).roll(shifts=(4, 4), dims=(1, 2)).reshape(n, 64)
    if not torch.equal(r.sort(1).values, rolled.sort(1).values):
        raise AssertionError("ROLL must preserve each candidate's evidence multiset")
    mean_error = (r.double().mean(1) - rolled.double().mean(1)).abs()
    torch.testing.assert_close(r.double().mean(1), rolled.double().mean(1), atol=1e-12, rtol=0)
    constant = r.mean(1, keepdim=True).expand_as(r)
    mean_delta = torch.bmm(k, constant.unsqueeze(2)).squeeze(2)
    roll_delta = torch.bmm(k, rolled.unsqueeze(2)).squeeze(2)
    invalid = ~selected["valid"].bool()
    if bool(invalid.any()):
        if bool((r[invalid] != 0).any()) or bool((correction[invalid] != 0).any()):
            raise AssertionError("Originally invalid candidates must retain zero evidence correction")
    coefficients = dict(A=c0, FULL=full, NATIVE=native,
                        MEAN=native + mean_delta.to(native.dtype),
                        ROLL=native + roll_delta.to(native.dtype))
    for value in coefficients.values():
        if value.shape != (n, 32) or not bool(torch.isfinite(value).all()):
            raise AssertionError("All candidates and finite coefficients must remain in every arm")
    a = selected["A_full"].double()
    native_delta = native - c0
    deltas = dict(native=native_delta, extra=full-native, mean_extra=mean_delta,
                  roll_extra=roll_delta, full_delta=full-c0)
    r64 = r.double()
    rmean = r64.mean(1)
    renergy = r64.square().sum(1)
    centered = (r64-rmean[:, None]).square().sum(1)
    diagnostic_tensors = dict(
        evidence_mean=rmean, evidence_std=r64.std(1, unbiased=False),
        evidence_rms=r64.square().mean(1).sqrt(),
        evidence_saturation_rate=(r64.abs() >= 3.96).double().mean(1),
        spatial_energy_fraction=torch.where(renergy > 0, centered/renergy.clamp_min(1e-300), torch.zeros_like(renergy)),
        native_delta_l2=native_delta.double().norm(dim=1),
        extra_delta_l2=(full-native).double().norm(dim=1),
        mean_extra_delta_l2=mean_delta.double().norm(dim=1),
        roll_extra_delta_l2=roll_delta.double().norm(dim=1),
    )
    for name, delta in deltas.items():
        z = torch.bmm(a, delta.double().unsqueeze(2)).squeeze(2)
        rms = z.square().mean(1).sqrt()
        # Invalid operator inputs have no defined 16x16 diagnostic; preserve
        # their candidate and task metrics, mark this diagnostic undefined.
        rms = torch.where(selected["valid"].bool(), rms, torch.full_like(rms, float("nan")))
        diagnostic_tensors[name+"_logit_rms16"] = rms
    mean_z = torch.bmm(a, mean_delta.double().unsqueeze(2)).squeeze(2)
    center_z = torch.bmm(a, ((full-native)-mean_delta).double().unsqueeze(2)).squeeze(2)
    valid = selected["valid"].bool()
    diagnostic_tensors["center_logit_rms16"] = torch.where(valid, center_z.square().mean(1).sqrt(), torch.full_like(renergy, float("nan")))
    diagnostic_tensors["mean_center_logit_inner_product16"] = torch.where(valid, (mean_z*center_z).mean(1), torch.full_like(renergy, float("nan")))
    values = {name: value.detach().cpu().tolist() for name, value in diagnostic_tensors.items()}
    diagnostics = [{name: values[name][i] for name in values} for i in range(n)]
    for i, row in enumerate(diagnostics):
        row["operator_valid"] = bool(selected["valid"][i])
        row["evidence_exactly_zero"] = bool((r[i] == 0).all())
    checks = dict(full_reconstruction_max_abs=float((full-reconstructed).abs().max()) if n else 0.,
                  roll_mean_max_abs=float(mean_error.max()) if n else 0.,
                  roll_multiset_exact=True, uniform_indices_exact=True,
                  invalid_operator_candidates=int(invalid.sum()))
    return coefficients, diagnostics, checks


def full_logit_diagnostics(image, coefficients, diagnostics):
    """No GT: continuous delta-logits within the original prediction support."""
    native = coefficients["NATIVE"]
    proto = image["proto"].float()
    boxes = image["boxes"].to(proto.device).float()
    shape = tuple(image["input_shape"])
    deltas = dict(extra=coefficients["FULL"]-native,
                  mean_extra=coefficients["MEAN"]-native,
                  center=coefficients["FULL"]-coefficients["MEAN"],
                  roll_extra=coefficients["ROLL"]-native)
    for start in range(0, len(native), 4):
        stop = min(start+4, len(native))
        support = ops.crop_mask(torch.ones((stop-start, *shape), device=proto.device), boxes[start:stop]).bool()
        zs = {name: em._logits(proto, delta[start:stop], shape) for name, delta in deltas.items()}
        for offset, row_id in enumerate(range(start, stop)):
            selected = support[offset]
            count = int(selected.sum())
            diagnostics[row_id]["prediction_support_pixels640"] = count
            for name, z in zs.items():
                diagnostics[row_id][name+"_logit_rms640"] = float(z[offset][selected].double().square().mean().sqrt()) if count else None
            diagnostics[row_id]["mean_center_logit_inner_product640"] = float((zs["mean_extra"][offset][selected].double()*zs["center"][offset][selected].double()).mean()) if count else None


def compare_prior(rows, prior, errors):
    for row in rows:
        if row["split"] != "dev":
            continue
        key = identity(row)
        if key not in prior:
            raise AssertionError(f"Missing historical dev identity: {key}")
        old = prior[key]
        for arm, original_arm in (("A", "A"), ("FULL", "U")):
            for metric in METRICS:
                actual, expected = row[metric+"_"+arm], old[metric+"_"+original_arm]
                if em.finite(actual) != em.finite(expected):
                    raise AssertionError(f"Historical metric definedness changed: {key} {arm} {metric}")
                if em.finite(actual):
                    error = abs(float(actual)-float(expected))
                    errors[arm][metric] = max(errors[arm][metric], error)
                    if error > 1e-12:
                        raise AssertionError(f"Historical metric reproduction failed: {key} {arm} {metric}, error={error:.17g}; tolerance=1e-12")


def summarize(rows, out, audit, budget):
    predicates = dict(all=lambda r: True, failure=lambda r: r["mask75_A"] == 0,
                      success=lambda r: r["mask75_A"] == 1,
                      box_good_mask_bad=lambda r: bool(r["box_good_mask_bad"]))
    tables, image_rows = {}, []
    for split in ("fit", "dev"):
        for name, predicate in predicates.items():
            budget()
            rs = [r for r in rows if r["split"] == split and predicate(r)]
            groups = em.image_groups(rs)
            tab = dict(images=len(groups), candidates=len(rs), candidate={}, image_macro={},
                       undefined={}, comparisons={}, diagnostics={})
            for metric in METRICS:
                tab["candidate"][metric] = {a: em.avg([r.get(metric+"_"+a) for r in rs]) for a in ARMS}
                tab["image_macro"][metric] = {a: em.avg([em.avg([r.get(metric+"_"+a) for r in g]) for g in groups]) for a in ARMS}
                tab["undefined"][metric] = {a: sum(not em.finite(r.get(metric+"_"+a)) for r in rs) for a in ARMS}
            for arm, ref in PAIRS:
                comparison = {m: em.paired(groups, arm, ref, m, 20261004, 1000) for m in METRICS}
                repair = sum(r["mask75_"+ref] == 0 and r["mask75_"+arm] == 1 for r in rs)
                damage = sum(r["mask75_"+ref] == 1 and r["mask75_"+arm] == 0 for r in rs)
                comparison["crossings"] = dict(repair=repair, damage=damage, net=repair-damage,
                                               denominator_candidates=len(rs))
                comparison["absolute_iou_difference"] = dict(
                    candidate=distribution([abs(r["iou_"+arm]-r["iou_"+ref]) for r in rs]),
                    image_macro=distribution([em.avg([abs(r["iou_"+arm]-r["iou_"+ref]) for r in g]) for g in groups]))
                tab["comparisons"][arm+"_minus_"+ref] = comparison
            for diagnostic in DIAGNOSTICS:
                tab["diagnostics"][diagnostic] = dict(
                    candidate=distribution([r.get(diagnostic) for r in rs]),
                    image_macro=distribution([em.avg([r.get(diagnostic) for r in g]) for g in groups]))
            tab["invalid_operator_candidates"] = sum(not r["operator_valid"] for r in rs)
            tab["normal_image_mask_changes_vs_A"] = {
                arm: dict(changed_candidates=sum(r["original_pixel_changes_"+arm+"_vs_A"] > 0 for r in rs),
                          candidate_fraction=em.avg([int(r["original_pixel_changes_"+arm+"_vs_A"] > 0) for r in rs]),
                          total_changed_pixels=sum(r["original_pixel_changes_"+arm+"_vs_A"] for r in rs))
                for arm in ARMS[1:]}
            tables[split+":"+name] = tab
            for g in groups:
                row = dict(split=split, image_id=g[0]["image_id"], group=name, candidates=len(g))
                for metric in METRICS:
                    for arm in ARMS:
                        row[metric+"_"+arm] = em.avg([r.get(metric+"_"+arm) for r in g])
                        row[metric+"_"+arm+"_defined"] = sum(em.finite(r.get(metric+"_"+arm)) for r in g)
                    for arm, ref in PAIRS:
                        row["delta_"+metric+"_"+arm+"_minus_"+ref] = em.avg([
                            r[metric+"_"+arm]-r[metric+"_"+ref] for r in g
                            if em.finite(r.get(metric+"_"+arm)) and em.finite(r.get(metric+"_"+ref))])
                for arm, ref in PAIRS:
                    row["repair_"+arm+"_minus_"+ref] = sum(r["mask75_"+ref] == 0 and r["mask75_"+arm] == 1 for r in g)
                    row["damage_"+arm+"_minus_"+ref] = sum(r["mask75_"+ref] == 1 and r["mask75_"+arm] == 0 for r in g)
                for d in DIAGNOSTICS:
                    row[d] = em.avg([r.get(d) for r in g])
                image_rows.append(row)
    budget()
    em.append_rows(out/"PER_IMAGE.jsonl", image_rows)
    result = dict(schema="frozen-U-evidence-path-usage-v1", arms=list(ARMS), tables=tables, audit=audit,
                  bootstrap=dict(draws=1000, seed=20261004, unit="paired whole image", multiplicity="descriptive unadjusted intervals"),
                  no_training=True, no_parameter_selection=True, no_COCO_AP=True,
                  no_new_blind_test=True, automatic_followup=False)
    dump(out/"SUMMARY.json", em.clean(result))
    def fmt(value, scale=100):
        return "undefined" if value is None else f"{scale*value:+.6f}"
    def interval(result, scale=100):
        lo, hi = result["ci95"]
        return f"{fmt(result['delta'], scale)} [{fmt(lo, scale)}, {fmt(hi, scale)}]"
    dev = tables["dev:all"]["comparisons"]
    first = "; ".join(pair+" "+interval(dev[pair]["iou"]["image_macro"])+" pp"
                      for pair in ("FULL_minus_A", "FULL_minus_NATIVE", "FULL_minus_MEAN", "FULL_minus_ROLL"))
    lines = ["# Frozen U evidence-path usage audit", "", first+".", "",
             "One existing epoch-3 U checkpoint, no updates. NATIVE is the native cv4 inside that very checkpoint, not the independently trained N arm.", "",
             "A uses original c0; FULL uses the unchanged U forward; NATIVE removes its extra correction; MEAN replaces the 64 evidence values by their within-candidate mean; ROLL moves them by four rows and four columns on the verified 8x8 U grid. P, K, prediction boxes, scores and decoding stay fixed.", "",
             "## Absolute values", "", "IoU/Mask75/coverage/AUC/FPR are percentages; BCE is in its original loss units. Each cell is image macro / candidate mean.", "",
             "| Population | Arm | IoU | Mask75 | Coverage | AUC | FPR | BCE |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group, table in tables.items():
        for arm in ARMS:
            cells = [fmt(table["image_macro"][m][arm], 1 if m == "bce" else 100)+" / "+
                     fmt(table["candidate"][m][arm], 1 if m == "bce" else 100) for m in METRICS]
            lines.append("| "+" | ".join([group, arm, *cells])+" |")
    lines += ["", "## Paired comparisons", "",
              "Each interval resamples whole images, including candidate-mean intervals. BCE differences retain original loss units; other differences are percentage points.", "",
              "| Population | Comparison | Metric | Image macro delta [95% CI] | Candidate delta [95% CI] | Repair / damage / net Mask75 |",
              "|---|---|---|---:|---:|---:|"]
    for group, table in tables.items():
        for pair, comparison in table["comparisons"].items():
            crossing = comparison["crossings"]
            for metric in METRICS:
                scale = 1 if metric == "bce" else 100
                lines.append(f"| {group} | {pair} | {metric} | {interval(comparison[metric]['image_macro'], scale)} | {interval(comparison[metric]['candidate'], scale)} | {crossing['repair']} / {crossing['damage']} / {crossing['net']} |")
    lines += ["", "## Interpretation boundaries", "",
              "- MEAN is constant evidence before multiplication by K, not a constant mask-logit bias. Its projected correction may remain spatially nonuniform.",
              "- ROLL retains every candidate's evidence multiset and mean, but breaks its spatial correspondence with fixed K. A loss under ROLL demonstrates sensitivity to this intervention, not by itself useful segmentation or an identified failure cause.",
              "- The native branch and extra head were jointly trained. FULL versus NATIVE is a same-checkpoint path ablation, not an estimate of independently training a native-only model.",
              "- Fit is the first 128 planned fit images; dev is the existing 256 planned images (smoke uses only its first two effective images). These are reused research data, not an independent test.",
              "- All groups are defined from original A only. Invalid operators, empty corrected masks and undefined AUC cases stay in the candidate population; undefined metrics have separate counts.",
              "- Per-candidate mean/std/RMS, centered-evidence energy share, coefficient increments and A_full-projected 16x16 logit RMS are saved without using GT to construct the interventions.",
              "- The 16x16 operator diagnostic is not the normal-image mask metric. No threshold, sample, checkpoint or parameter is selected from these outcomes.",
              "- No new training, model design or automatic follow-up is authorized by this audit.", ""]
    (out/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    budget()
    return result


@torch.no_grad()
def run(args):
    started = time.monotonic()
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Model evaluation is allowed only on the authorized Linux CUDA server")
    cfg = resolve_runtime_config(load_json(args.config))
    original_cfg = resolve_runtime_config(load_json(cfg["original_config"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ("PER_CANDIDATE.jsonl", "COMPLETE.json", "SUMMARY.json")):
        raise RuntimeError("Preserve previous outcomes; use a new Run")
    deadline = started + (180 if args.smoke else 900)
    def budget():
        if time.monotonic() > deadline:
            raise TimeoutError("Frozen evidence audit exhausted its fixed wall-clock budget; no automatic retry")
    for key in ("weights", "cache", "assets", "annotations_train", "source_python"):
        if str(cfg[key]) != str(original_cfg[key]):
            raise AssertionError(f"Audit must retain the original runtime input: {key}")
    if load_json(cfg["split"]) != load_json(original_cfg["split"]):
        raise AssertionError("Copied split manifest differs from the original screen")
    if original_cfg.get("epochs") != 3 or original_cfg.get("sampled_cells") != 64:
        raise AssertionError("Only the existing epoch-3 / 64-point screen is eligible")
    index = load_index(cfg)
    if len(index["fit"]) != 1024 or len(index["dev"]) != 256 or index.get("val"):
        raise AssertionError("Original screen population changed")
    planned = dict(fit=index["fit"][:128], dev=index["dev"])
    if args.smoke:
        planned = dict(fit=[], dev=[entry for entry in index["dev"] if int(entry["n"]) > 0][:2])
        if len(planned["dev"]) != 2:
            raise AssertionError("Smoke requires two effective original dev images")
    expected = {split: dict(planned_images=len(entries), effective_images=sum(bool(e["n"]) for e in entries),
                           candidates=sum(int(e["n"]) for e in entries),
                           no_positive_image_ids=[int(e["image_id"]) for e in entries if not e["n"]])
                for split, entries in planned.items()}
    dump(out/"EVALUATION_INDEX.json", dict(planned=planned, expected=expected, smoke=args.smoke))
    prior = {}
    for line in Path(cfg["prior_results"]).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = identity(row)
        if key in prior:
            raise AssertionError("Historical result contains duplicate identities")
        if row["split"] != "dev":
            raise AssertionError("Expected the original fast-screen dev result file")
        prior[key] = row
    if len(prior) != sum(int(e["n"]) for e in index["dev"]):
        raise AssertionError("Historical dev result population differs from original index")
    replay = FrozenReplay(cfg)
    checkpoint = Path(cfg["checkpoint"])
    expected_checkpoint_sha = "d777ae8fd431092c9912fae0bb537e2d755b961729d5091b9e3f065d5a32406d"
    if cfg.get("checkpoint_sha256", expected_checkpoint_sha) != expected_checkpoint_sha or sha256(checkpoint) != expected_checkpoint_sha:
        raise AssertionError("Checkpoint is not the fixed audited U epoch-3 artifact")
    receipt = load_json(checkpoint.parent/"COMPLETE.json")
    if receipt.get("completed") is not True or receipt.get("epochs") != 3:
        raise AssertionError("Only the completed epoch-3 U checkpoint may be audited")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("mode") != "U" or ck.get("epoch") != 3 or checkpoint.name != "final.pt":
        raise AssertionError("Expected U/final.pt at the fixed epoch 3")
    if resolve_runtime_config(ck["config"]) != original_cfg:
        raise AssertionError("Saved checkpoint config differs from the frozen original config")
    if list(ck["feature_channels"]) != list(replay.feature_channels):
        raise AssertionError("Feature channel mismatch")
    setup(original_cfg["seed"])
    model = build_model("U", replay, original_cfg).to(replay.device).float().eval()
    model.load_state_dict(ck["state_dict"], strict=True)
    model.requires_grad_(False)
    assert_frozen(model)
    before = state_digest(model)
    del ck
    coco = COCO(str(cfg["annotations_train"]))
    files = {name: str(sys.modules[name].__file__) for name in
             ("online_runtime", "screen_models", "point_selectors", "point_head", "joint_head", "evaluation_metrics", "evaluate_screen")}
    environment = dict(replay.import_info, torch_version=torch.__version__, numpy_version=np.__version__,
                       gpu=torch.cuda.get_device_name(replay.device), checkpoint_sha256=sha256(checkpoint),
                       code_sha256={name: sha256(path) for name, path in files.items()}, code_paths=files,
                       audit_script_sha256=sha256(__file__), config_sha256=sha256(args.config),
                       original_config_sha256=sha256(cfg["original_config"]), prior_results_sha256=sha256(cfg["prior_results"]))
    dump(out/"ENVIRONMENT.json", environment)
    rows, seen, dev_seen = [], set(), set()
    errors = {a: {m: 0. for m in METRICS} for a in ("A", "FULL")}
    checks = dict(full_reconstruction_max_abs=0., roll_mean_max_abs=0.,
                  invalid_operator_candidates=0, uniform_indices_exact=True, roll_multiset_exact=True)
    timings = dict(forward=0., operators=0., metrics=0.)
    for split, entries in planned.items():
        for position, entry in enumerate(entries):
            budget()
            if not int(entry["n"]):
                continue
            image = load_asset(cfg, int(entry["image_id"]), verify=True)
            if image["split"] != split or len(image["rows"]) != int(entry["n"]):
                raise AssertionError("Prepared image population changed")
            keys = [identity(row) for row in image["rows"]]
            if len(keys) != len(set(keys)) or seen.intersection(keys):
                raise AssertionError("Candidate identities must remain unique")
            input_hashes = {k: tensor_sha(image[k]) for k in ("c0", "proto", "boxes", "raw_ids")}
            torch.cuda.synchronize()
            t = time.monotonic()
            selected = build_operators(image, "U", original_cfg, replay.device)
            operator_diagnostic = selected.pop("_operator_diagnostics", {})
            torch.cuda.synchronize()
            timings["operators"] += time.monotonic()-t
            t = time.monotonic()
            features = replay.replay([image])
            full, residual, current = model.forward_details(features, [selected])
            coefficients, diagnostics, local_checks = make_interventions(
                full[0], residual[0], current[0], selected, image["c0"].to(replay.device))
            torch.cuda.synchronize()
            timings["forward"] += time.monotonic()-t
            t = time.monotonic()
            gpu = official_gpu_payload(image, replay.device)
            full_logit_diagnostics(gpu, coefficients, diagnostics)
            bce = {arm: official_bce_values(gpu, value) for arm, value in coefficients.items()}
            decoded = em.evaluate_image(gpu, coefficients, coco, chunk_size=4)
            if [identity(row) for row in decoded] != keys:
                raise AssertionError("Normal decoder changed candidate identities or order")
            for i, row in enumerate(decoded):
                for arm in ARMS:
                    row["bce_"+arm] = float(bce[arm][i])
                row.update(diagnostics[i])
            compare_prior(decoded, prior, errors)
            for key, digest in input_hashes.items():
                if tensor_sha(image[key]) != digest:
                    raise AssertionError(f"Frozen input changed: {key}")
            assert_frozen(model)
            timings["metrics"] += time.monotonic()-t
            checks["full_reconstruction_max_abs"] = max(checks["full_reconstruction_max_abs"], local_checks["full_reconstruction_max_abs"])
            checks["roll_mean_max_abs"] = max(checks["roll_mean_max_abs"], local_checks["roll_mean_max_abs"])
            checks["invalid_operator_candidates"] += local_checks["invalid_operator_candidates"]
            em.append_rows(out/"OPERATOR_AUDIT.jsonl", [dict(split=split, image_id=int(entry["image_id"]),
                           inputs_sha256=input_hashes, diagnostics=operator_diagnostic, checks=local_checks)])
            em.append_rows(out/"PER_CANDIDATE.jsonl", decoded)
            rows.extend(decoded)
            seen.update(keys)
            if split == "dev":
                dev_seen.update(keys)
            if position % 20 == 0 or position+1 == len(entries):
                progress = dict(split=split, processed_planned_images=position+1, planned_images=len(entries),
                                candidates=len(rows), elapsed_s=time.monotonic()-started)
                dump(out/"PROGRESS.json", progress)
                print(json.dumps(progress), flush=True)
            del image, selected, features, full, residual, current, coefficients, gpu, bce, decoded
    budget()
    if len(rows) != sum(x["candidates"] for x in expected.values()):
        raise AssertionError("The fixed planned population was not fully evaluated")
    if not args.smoke and dev_seen != set(prior):
        raise AssertionError("Dev identities differ from the complete historical dev population")
    replay.assert_unchanged()
    assert_frozen(model)
    after = state_digest(model)
    if after != before:
        raise AssertionError("Frozen U checkpoint state changed during evaluation")
    audit = dict(expected=expected, source_weights_sha256=replay.weights_sha256,
                 checkpoint_sha256=sha256(checkpoint), model_state_before=before, model_state_after=after,
                 model_state_unchanged=True, gradients_disabled=True, replay_source_unchanged=True,
                 original_input_hashes_checked=True, historical_dev_reproduced=len(dev_seen),
                 historical_metric_tolerance=1e-12, historical_max_errors=errors,
                 intervention_checks=checks, elapsed_by_stage_seconds=timings,
                 no_training=True, no_new_TAL=True, no_parameter_selection=True,
                 smoke=args.smoke, maximum_seconds=180 if args.smoke else 900,
                 geometry="unchanged full-prototype process_mask(upsample=True) and original ratio_pad scale_masks",
                 regularization="No new solver, loss or regularization; existing U operator K reused",
                 native_control="native cv4 of this U checkpoint, never the separate N checkpoint")
    dump(out/"AUDIT.json", em.clean(audit))
    budget()
    summarize(rows, out, audit, budget)
    budget()
    dump(out/"COMPLETE.json", dict(completed=True, passed=True, code_checks_passed=True,
         smoke=args.smoke, candidates=len(rows), elapsed_s=time.monotonic()-started,
         automatic_followup=False, no_training=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        target = Path(args.out)
        target.mkdir(parents=True, exist_ok=True)
        dump(target/"FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        if not (target/"COMPLETE.json").exists():
            dump(target/"COMPLETE.json", dict(completed=False, passed=False, smoke=args.smoke, no_training=True))
        raise


if __name__ == "__main__":
    main()
