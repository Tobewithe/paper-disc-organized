"""Frozen SAM box responses -> existing OGPS -> original YOLO prototypes.

Five prespecified arms only. No model updates, GT prompts, output selection,
threshold tuning, teacher fitting or original candidate reassignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

if "--config" in sys.argv:
    _cfg = json.loads(Path(sys.argv[sys.argv.index("--config")+1]).read_text(encoding="utf-8-sig"))
    sys.path.insert(0, _cfg["source_python"])

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
import ultralytics
from ultralytics.utils import ops

from online_runtime import dump, load_asset, load_index, load_json, resolve_runtime_config, sha256, tensor_sha
from ogps_solver import (crop_pool_7o, pooled_design_matrix, solve_ogps, solve_ogps_logits,
                         RIDGE_LAMBDA, PROBABILITY_MIN, PROBABILITY_MAX)
import evaluation_metrics as em

ARMS = ("A", "BASE_SOLVE", "SAM_FULL", "SAM_GRID", "SAM_SOLVE")
METRICS = ("iou", "mask75", "coverage", "auc", "fpr")
PAIRS = tuple((a, "A") for a in ARMS[1:]) + (
    ("SAM_SOLVE", "BASE_SOLVE"), ("SAM_SOLVE", "SAM_FULL"),
    ("SAM_SOLVE", "SAM_GRID"), ("SAM_GRID", "SAM_FULL"))
KEYS = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
em.ARMS = ARMS


def identity(row):
    return tuple(row[k] for k in KEYS)


def state_sha(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        v = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(v.dtype).encode())
        digest.update(str(tuple(v.shape)).encode())
        digest.update(v.numpy().tobytes())
    return digest.hexdigest()


def assert_frozen(model):
    if any(module.training for module in model.modules()):
        raise AssertionError("SAM must remain entirely in evaluation mode")
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError("SAM parameters must have no gradient or update")


def verify_sam_assets(cfg):
    code = Path(cfg["sam_code"]).resolve()
    path = Path(cfg.get("asset_manifest", str(Path(cfg["server_root"])/"ASSET_MANIFEST.json")))
    manifest = load_json(path)
    expected = manifest["checkpoint_sha256"]
    actual = sha256(cfg["sam_checkpoint"])
    if actual != expected:
        raise AssertionError("SAM checkpoint differs from the frozen asset manifest")
    files = manifest["source_files_sha256"]
    if not isinstance(files, dict) or not files:
        raise AssertionError("SAM source files must have frozen SHA256 entries")
    for relative, digest in files.items():
        target = (code/relative).resolve()
        if not target.is_relative_to(code) or sha256(target) != digest:
            raise AssertionError(f"SAM source differs: {relative}")
    actual_files = {str(p.relative_to(code)).replace("\\", "/") for p in (code/"segment_anything").rglob("*.py")}
    if not actual_files.issubset(set(files)):
        raise AssertionError("Asset manifest must cover all imported SAM package Python files")
    commit = manifest["source_commit"]
    if commit != "dca509fe793f601edb92606367a655c15ac00fdf":
        raise AssertionError("Unexpected official SAM source revision")
    # Official raw files were relayed from this fixed commit without .git.
    # The manifest authenticates every imported source file's exact bytes.
    return dict(manifest_sha256=sha256(path), checkpoint_sha256=actual,
                code_commit=commit, source_files=len(files), source_verified=True)


class FrozenSAM:
    """Strictly GT-free source: only cached RGB and frozen prediction boxes."""
    def __init__(self, cfg):
        self.assets = verify_sam_assets(cfg)
        sys.path.insert(0, cfg["sam_code"])
        import segment_anything
        from segment_anything import sam_model_registry, SamPredictor
        if not Path(segment_anything.__file__).resolve().is_relative_to(Path(cfg["sam_code"]).resolve()):
            raise AssertionError("Imported SAM does not come from the pinned source")
        self.model = sam_model_registry["vit_b"](checkpoint=cfg["sam_checkpoint"]).cuda().float().eval().requires_grad_(False)
        if int(self.model.image_encoder.img_size) != 1024:
            raise AssertionError("Do not replace SAM's native 1024 encoder size with the 640 input frame")
        if float(self.model.mask_threshold) != 0.:
            raise AssertionError("Expected the unchanged official zero SAM mask threshold")
        self.predictor = SamPredictor(self.model)
        assert_frozen(self.model)
        self.before = state_sha(self.model)
        self.embedding_count = 0
        self.candidate_count = 0
        self.prompt_count = 0
        self.timings = dict(embedding_seconds=0., prompt_seconds=0.)
        self.image_audits = []

    @torch.no_grad()
    def infer(self, payload, budget):
        if set(payload) != {"image_id", "input_uint8", "boxes"}:
            raise AssertionError("SAM input allowlist excludes all GT and supervision fields")
        image = payload["input_uint8"]
        boxes = payload["boxes"].detach().to("cuda").float()
        if image.dtype != torch.uint8 or image.shape != (3, 640, 640):
            raise AssertionError("SAM expects the exact cached CHW uint8 RGB input")
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not bool(torch.isfinite(boxes).all()):
            raise AssertionError("Prediction boxes must be finite [N,4]; do not repair corrupted assets")
        rgb = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
        # Cache is already RGB and uint8. No channel swap or division by 255.
        budget(); torch.cuda.synchronize(); started = time.monotonic()
        self.predictor.set_image(rgb, image_format="RGB")
        torch.cuda.synchronize(); seconds = time.monotonic()-started
        self.timings["embedding_seconds"] += seconds
        self.embedding_count += 1
        if tuple(self.predictor.original_size) != (640, 640):
            raise AssertionError("SAM original_size must be the cached 640 input frame")
        valid = ((boxes[:, 2:]-boxes[:, :2]) > 0).all(1)
        valid &= ((boxes[:, 2:].clamp(0, 640)-boxes[:, :2].clamp(0, 640)) > 0).all(1)
        audit = dict(image_id=int(payload["image_id"]), candidates=len(boxes), valid_prompts=int(valid.sum()),
                     invalid_box_fallback=int((~valid).sum()), embedding_count=1, embedding_seconds=seconds,
                     original_size=list(self.predictor.original_size), transformed_size=list(self.predictor.input_size))
        self.image_audits.append(audit)
        for start in range(0, len(boxes), 8):
            budget()
            stop = min(start+8, len(boxes))
            ids = torch.where(valid[start:stop])[0]+start
            results = {}
            if ids.numel():
                transformed = self.predictor.transform.apply_boxes_torch(boxes[ids], self.predictor.original_size)
                torch.cuda.synchronize(); tick = time.monotonic()
                logits, quality, low_res = self.predictor.predict_torch(
                    point_coords=None, point_labels=None, boxes=transformed, mask_input=None,
                    multimask_output=False, return_logits=True)
                torch.cuda.synchronize(); self.timings["prompt_seconds"] += time.monotonic()-tick
                if logits.shape != (len(ids), 1, 640, 640) or low_res.shape != (len(ids), 1, 256, 256):
                    raise AssertionError("Use SAM postprocessed 640 logits, not the 256 output")
                if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(quality).all()):
                    raise FloatingPointError("SAM returned nonfinite outputs")
                if logits.requires_grad:
                    raise AssertionError("SAM inference must not construct a gradient graph")
                self.prompt_count += len(ids)
                results = {int(idx): (logits[k, 0], float(quality[k, 0])) for k, idx in enumerate(ids.tolist())}
            for j in range(start, stop):
                self.candidate_count += 1
                z, quality_score = results.get(j, (None, None))
                yield j, z, dict(source="SAM_vit_b", quality_score=quality_score,
                                 quality_used_for_selection=False, source_available=bool(valid[j]),
                                 fallback=None if bool(valid[j]) else "finite_degenerate_or_outside_box_use_original_response")
            if results:
                del results, logits, quality, low_res
        self.predictor.reset_image()
        assert_frozen(self.model)

    def finish(self):
        assert_frozen(self.model)
        after = state_sha(self.model)
        if after != self.before:
            raise AssertionError("SAM parameters or buffers changed")
        return dict(assets=self.assets, state_before=self.before, state_after=after,
                    state_unchanged=True, gradients_disabled=True, embedding_count=self.embedding_count,
                    candidates=self.candidate_count, prompted_candidates=self.prompt_count,
                    timings=self.timings, images=self.image_audits)


def grid_to_canvas(target_logits, box, device):
    """Copied unchanged from the prior local-view/OGPS direct renderer."""
    x1 = max(0, min(639, math.floor(float(box[0]))))
    y1 = max(0, min(639, math.floor(float(box[1]))))
    x2 = max(x1+1, min(640, math.ceil(float(box[2]))))
    y2 = max(y1+1, min(640, math.ceil(float(box[3]))))
    patch = F.interpolate(target_logits.reshape(1, 1, 8, 8).float(), (y2-y1, x2-x1),
                          mode="bilinear", align_corners=False)[0, 0]
    canvas = torch.full((640, 640), -30., device=device)
    canvas[y1:y2, x1:x2] = patch
    return canvas


def crop_support_audit(box):
    """Record legacy integer crop edges; do not change either pooling path."""
    def bounds(size):
        x1 = max(0, min(size-1, math.floor(float(box[0])*size/640)))
        y1 = max(0, min(size-1, math.floor(float(box[1])*size/640)))
        x2 = max(x1+1, min(size, math.ceil(float(box[2])*size/640)))
        y2 = max(y1+1, min(size, math.ceil(float(box[3])*size/640)))
        return [x1, y1, x2, y2]
    p160, q640 = bounds(160), bounds(640)
    mapped = [4*v for v in p160]
    return dict(prototype_grid160_crop=p160, probability_grid640_crop=q640,
                prototype_crop_in_input640=mapped,
                edge_delta_input_pixels=[a-b for a, b in zip(mapped, q640)],
                same_discrete_support=mapped == q640)


@torch.no_grad()
def add_direct(row, name, logits, image, candidate_index, coco):
    """Original-box crop and identical original-image inverse, unchanged."""
    if logits.shape != (640, 640) or not bool(torch.isfinite(logits).all()):
        raise AssertionError("Direct output must be finite continuous input640 logits")
    box = image["boxes"][candidate_index].to(logits.device)
    support = ops.crop_mask(torch.ones((1, 640, 640), device=logits.device), box[None])[0].bool()
    binary = ((logits > 0) & support).byte()[None]
    mask = em._scale_binary(binary, tuple(image["original_shape"]), image["ratio_pad"])[0]
    annotation = coco.anns[row["annotation_id"]]
    if int(annotation["image_id"]) != int(image["image_id"]):
        raise AssertionError("COCO evaluation identity changed")
    truth = torch.as_tensor(coco.annToMask(annotation).astype(bool), device=logits.device)
    padded_truth = em._padded_gt(truth, image["ratio_pad"], (640, 640))
    intersection = int((mask & truth).sum())
    row["iou_"+name] = intersection/max(1, int((mask | truth).sum()))
    row["coverage_"+name] = intersection/max(1, int(truth.sum()))
    row["mask75_"+name] = int(row["iou_"+name] >= .75)
    row["auc_"+name], row["fpr_"+name] = em._pixel_auc_fpr(logits, padded_truth, support)
    row["empty_mask_"+name] = not bool(mask.any())
    if not torch.equal(ops.crop_mask(logits[None].clone(), box[None]).gt(0).byte(), binary):
        raise AssertionError("Direct crop/threshold ordering differs")


def solver_check(result):
    error = float(result["diagnostics"]["normal_equation_relative_backward_error"].max())
    if not math.isfinite(error) or error > 1e-10:
        raise ArithmeticError(f"Fixed FP64 OGPS backward error {error} exceeds 1e-10")
    return error


def summarize(rows, out, audit, budget):
    predicates = dict(all=lambda r: True, box_good_mask_bad=lambda r: bool(r["box_good_mask_bad"]),
                      original_success=lambda r: bool(r["mask75_A"]), original_failure=lambda r: not bool(r["mask75_A"]))
    tables, image_rows = {}, []
    for name, predicate in predicates.items():
        budget(); rs = [r for r in rows if predicate(r)]; groups = em.image_groups(rs)
        tab = dict(images=len(groups), candidates=len(rs), candidate={}, image_macro={}, undefined={}, comparisons={})
        for metric in METRICS:
            tab["candidate"][metric] = {a: em.avg([r.get(metric+"_"+a) for r in rs]) for a in ARMS}
            tab["image_macro"][metric] = {a: em.avg([em.avg([r.get(metric+"_"+a) for r in g]) for g in groups]) for a in ARMS}
            tab["undefined"][metric] = {a: sum(not em.finite(r.get(metric+"_"+a)) for r in rs) for a in ARMS}
        for arm, ref in PAIRS:
            compared = {m: em.paired(groups, arm, ref, m, 20261004, 1000) for m in METRICS}
            repair = sum(not r["mask75_"+ref] and r["mask75_"+arm] for r in rs)
            damage = sum(r["mask75_"+ref] and not r["mask75_"+arm] for r in rs)
            compared["crossings"] = dict(repair=repair, damage=damage, net=repair-damage)
            tab["comparisons"][arm+"_minus_"+ref] = compared
        tables[name] = tab
        for g in groups:
            row = dict(split="dev", image_id=g[0]["image_id"], group=name, candidates=len(g))
            for metric in METRICS:
                for arm in ARMS:
                    row[metric+"_"+arm] = em.avg([r.get(metric+"_"+arm) for r in g])
                    row[metric+"_"+arm+"_defined"] = sum(em.finite(r.get(metric+"_"+arm)) for r in g)
                for arm, ref in PAIRS:
                    row["delta_"+metric+"_"+arm+"_minus_"+ref] = em.avg([
                        r[metric+"_"+arm]-r[metric+"_"+ref] for r in g
                        if em.finite(r.get(metric+"_"+arm)) and em.finite(r.get(metric+"_"+ref))])
            image_rows.append(row)
    budget(); em.append_rows(out/"PER_IMAGE.jsonl", image_rows)
    summary = dict(schema="external-spatial-projection-v1", arms=list(ARMS), tables=tables, audit=audit,
                   no_training=True, no_COCO_AP=True, no_new_blind_test=True, decision="STOP_after_fixed_replay",
                   automatic_followup=False, bootstrap=dict(seed=20261004, draws=1000, unit="paired whole image", multiplicity="descriptive unadjusted intervals"))
    dump(out/"SUMMARY.json", em.clean(summary))
    def pp(x): return "undefined" if x is None else f"{100*x:+.5f}"
    lines = ["# External frozen spatial evidence projected into original YOLO prototypes", "",
             "STOP after this fixed replay; no automatic training, prompts, threshold search or solver changes.", "",
             f"{'SMOKE ONLY. ' if audit['smoke'] else ''}{len(rows)} fixed candidates in {tables['all']['images']} effective images. External SAM ViT-B is an additional pretrained model and compute source, not a lightweight YOLO-only improvement.", "",
             "| Group | Arm | Macro IoU | Candidate IoU | Macro Mask75 | Coverage | AUC | FPR |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group, tab in tables.items():
        for arm in ARMS:
            vals = [tab["image_macro"]["iou"][arm], tab["candidate"]["iou"][arm]] + [tab["image_macro"][m][arm] for m in METRICS[1:]]
            lines.append("| "+" | ".join([group, arm, *map(pp, vals)])+" |")
    lines += ["", "All changes below are percentage points. Image macro point estimates and intervals use the same statistic; candidate-mean intervals also resample whole images.", "",
              "| Group | Comparison | Macro IoU delta [95% CI] | Candidate delta | Repair / damage | Coverage delta | AUC delta | FPR delta |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group, tab in tables.items():
        for pair, compared in tab["comparisons"].items():
            value = compared["iou"]["image_macro"]; lo, hi = value["ci95"]; cross = compared["crossings"]
            lines.append(f"| {group} | {pair} | {pp(value['delta'])} [{pp(lo)}, {pp(hi)}] | {pp(compared['iou']['candidate']['delta'])} | {cross['repair']} / {cross['damage']} | {pp(compared['coverage']['image_macro']['delta'])} | {pp(compared['auc']['image_macro']['delta'])} | {pp(compared['fpr']['image_macro']['delta'])} |")
    lines += ["", "Arm definitions and limits:", "",
              "- A is original c0 with complete original P and normal frozen-box decoding.",
              "- BASE_SOLVE uses original full640 logits -> sigmoid once -> identical floor/ceil predicted-box pooling -> 8x8 -> probability clip [.01,.99] -> logit -> existing FP64 ridge (.003). It is not the previous sigmoid(A8 c0) baseline.",
              "- SAM_FULL uses postprocessed continuous SAM logits at 640, the original prediction-box support, and the same original-image inverse. The input is exact cached RGB uint8; SAM internally performs its official 1024 preprocessing.",
              "- SAM_GRID and SAM_SOLVE share precisely the same q8 and clipped target logits. GRID bilinearly renders those logits into the floor/ceil box; SOLVE fits original-P coefficients centered on c0 and uses normal original-P decoding.",
              "- FULL -> GRID changes pooling, clipping and rendering together, not only resolution. P pooling operates on the original160 grid while probability pooling uses640; legacy integer-quantization differences remain and a negative result is not a prototype-capacity proof.",
              "- Quality predictions are recorded but never select masks, candidates, prompts or subsets. No point prompts, GT boxes/classes, multimask oracle selection or iterative mask inputs are used.",
              "- Finite degenerate/nonintersecting prediction boxes remain: SAM_FULL falls back to original logits; the same baseline q8 is used by GRID/SOLVE, explicitly marked unavailable. Nonfinite assets fail instead of being silently repaired.",
              "- Every candidate, including empty outputs and undefined AUC, remains. Original groups use A only. This is reused developer data on fixed official supervision candidates, not full inference AP or a new blind test.",
              "- External pretrained knowledge and encoder/prompt latency are extra costs; a positive replay does not establish distillability, YOLO-internal information sufficiency, or a deployable lightweight contribution.",
              "- Replay q8/coefficient tensors remain on the server under REPLAY_TENSORS for independent reconstruction. No historical OGPS effect size or oracle-recovery ratio is imported.", ""]
    (out/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    budget()


@torch.no_grad()
def run(args):
    started = time.monotonic()
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Authorized Linux GPU server only; never execute on desktop or laptop")
    cfg = resolve_runtime_config(load_json(args.config))
    if cfg["arms"] != list(ARMS) or cfg["sam_model_type"] != "vit_b" or cfg["prompt_batch"] != 8:
        raise AssertionError("Five fixed arms, ViT-B and prompt batch8 are required")
    if cfg["multimask_output"] is not False or cfg["return_logits"] is not True or cfg["sam_image_size"] != 640:
        raise AssertionError("Fixed cached640/single-mask/continuous-logit protocol required")
    if (RIDGE_LAMBDA, PROBABILITY_MIN, PROBABILITY_MAX) != (.003, .01, .99):
        raise AssertionError("Existing OGPS solver parameters changed")
    if ultralytics.__version__ != "8.4.100" or not Path(ultralytics.__file__).resolve().is_relative_to(Path(cfg["source_python"]).resolve()):
        raise AssertionError("Normal decoder must use the frozen Ultralytics8.4.100 source")
    torch.set_num_threads(int(cfg.get("cpu_threads", 6)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0); np.random.seed(0)
    torch.cuda.reset_peak_memory_stats()
    deadline = started+(300 if args.smoke else 3600)
    def budget():
        if time.monotonic() >= deadline:
            raise TimeoutError("Fixed replay budget exhausted; no prefix conclusions or automatic retry")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ("PER_CANDIDATE.jsonl", "COMPLETE.json", "SUMMARY.json")):
        raise RuntimeError("Use an independent Run; retain prior outputs")
    index = load_index(cfg); planned = index["dev"]
    if len(planned) != 256 or sum(int(e["n"]) for e in planned) != 1816:
        raise AssertionError("Frozen fast-screen dev population changed")
    if [int(e["image_id"]) for e in planned] != list(map(int, load_json(cfg["split"])["dev"])):
        raise AssertionError("Frozen dev order changed")
    items = [e for e in planned if int(e["n"])][:2] if args.smoke else planned
    expected = dict(planned_images=len(items), effective_images=sum(bool(e["n"]) for e in items),
                    candidates=sum(int(e["n"]) for e in items), no_positive_image_ids=[int(e["image_id"]) for e in items if not e["n"]])
    required_images = {int(e["image_id"]) for e in items}; prior = {}
    for line in Path(cfg["prior_results"]).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] == "dev" and int(row["image_id"]) in required_images:
            k = identity(row)
            if k in prior: raise AssertionError("Duplicate historical identity")
            prior[k] = row
    if len(prior) != expected["candidates"]:
        raise AssertionError("Historical baseline does not cover all required candidates")
    dump(out/"EVALUATION_INDEX.json", dict(items=items, expected=expected))
    source = FrozenSAM(cfg); budget(); coco = COCO(str(cfg["annotations_train"]))
    environment = dict(torch=torch.__version__, ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(),
                       source_assets=source.assets, config_sha256=sha256(args.config), script_sha256=sha256(__file__),
                       solver_sha256=sha256(sys.modules["ogps_solver"].__file__),
                       evaluation_sha256=sha256(em.__file__), cache_identity_sha256=sha256(Path(cfg["cache"])/"CACHE_IDENTITY.json"),
                       previous_results_sha256=sha256(cfg["prior_results"]), training_updates=0)
    dump(out/"ENVIRONMENT.json", environment)
    saved = out/"REPLAY_TENSORS"; saved.mkdir()
    rows, seen = [], set(); max_errors = {m: 0. for m in METRICS}
    max_solver_error = 0.; source_fallbacks = 0; solver_seconds = 0.; metric_seconds = 0.
    for position, entry in enumerate(items):
        budget()
        if not int(entry["n"]): continue
        x = load_asset(cfg, int(entry["image_id"]), verify=True)
        if x["split"] != "dev" or len(x["rows"]) != int(entry["n"]):
            raise AssertionError("Candidate population changed")
        keys = [identity(r) for r in x["rows"]]
        if len(keys) != len(set(keys)) or seen.intersection(keys):
            raise AssertionError("Duplicate candidate identities")
        input_hashes = {name: tensor_sha(x[name]) for name in ("proto", "c0", "boxes", "input_uint8", "raw_ids")}
        proto, c0, boxes = (x[name].to("cuda").float() for name in ("proto", "c0", "boxes"))
        if not all(bool(torch.isfinite(v).all()) for v in (proto, c0, boxes)):
            raise FloatingPointError("Nonfinite frozen scientific inputs")
        gpu = dict(x, proto=proto, c0=c0, boxes=boxes)
        torch.cuda.synchronize(); tick = time.monotonic()
        matrix = pooled_design_matrix(proto, boxes).double()
        own_pooled_logits = (matrix@c0.double().unsqueeze(2)).squeeze(2)
        exact = solve_ogps_logits(matrix, c0, own_pooled_logits)
        if bool((exact["delta"] != 0).any()) or not torch.equal(exact["coeff"].float(), c0):
            raise AssertionError("Unclipped self-logit identity must exactly return original c0")
        base_q = []
        for lo in range(0, len(c0), 4):
            z0 = em._logits(proto, c0[lo:lo+4], (640, 640))
            for offset in range(len(z0)):
                base_q.append(crop_pool_7o(z0[offset].sigmoid()[None], boxes[lo+offset]).reshape(8, 8))
        base_q = torch.stack(base_q)
        base = solve_ogps(matrix, c0, base_q)
        max_solver_error = max(max_solver_error, solver_check(base))
        torch.cuda.synchronize(); solver_seconds += time.monotonic()-tick
        direct, sam_q, sam_coeff = [], [], []
        payload = {name: x[name] for name in ("image_id", "input_uint8", "boxes")}
        for j, logits, metadata in source.infer(payload, budget):
            budget()
            if j != len(direct): raise AssertionError("SAM reordered candidates")
            if logits is None:
                lo = (j//4)*4
                logits = em._logits(proto, c0[lo:lo+4], (640, 640))[j-lo]
                q = base_q[j]
                source_fallbacks += 1
            else:
                q = crop_pool_7o(logits.sigmoid()[None], boxes[j]).reshape(8, 8)
            torch.cuda.synchronize(); tick = time.monotonic()
            solved = solve_ogps(matrix[j], c0[j], q)
            max_solver_error = max(max_solver_error, solver_check(solved))
            clipped = q.double().clamp(.01, .99)
            expected_target = (clipped.log()-torch.log1p(-clipped)).flatten()
            torch.testing.assert_close(solved["target_logits"], expected_target, atol=0, rtol=0)
            torch.cuda.synchronize(); solver_seconds += time.monotonic()-tick
            tick = time.monotonic()
            row = dict(annotation_id=x["rows"][j]["annotation_id"], sam_source=metadata,
                       pooling_support=crop_support_audit(boxes[j]),
                       solver={k: float(v) for k, v in solved["diagnostics"].items()})
            add_direct(row, "SAM_FULL", logits, gpu, j, coco)
            add_direct(row, "SAM_GRID", grid_to_canvas(solved["target_logits"], boxes[j], proto.device), gpu, j, coco)
            metric_seconds += time.monotonic()-tick
            direct.append(row); sam_q.append(q.detach().cpu()); sam_coeff.append(solved["coeff"].float())
        if len(direct) != len(c0): raise AssertionError("SAM dropped a candidate")
        coeffs = dict(A=c0, BASE_SOLVE=base["coeff"].float(), SAM_SOLVE=torch.stack(sam_coeff))
        tick = time.monotonic(); decoded = em.evaluate_image(gpu, coeffs, coco, chunk_size=4)
        if [identity(r) for r in decoded] != keys: raise AssertionError("Normal decoder changed candidate identities")
        for j, row in enumerate(decoded):
            if row["zero_bias_padded_pixel_differences"] or row["zero_bias_original_pixel_differences"]:
                raise AssertionError("Original zero-evidence decode did not reproduce")
            old = prior[identity(row)]
            for metric in METRICS:
                actual, expected_value = row[metric+"_A"], old[metric+"_A"]
                if em.finite(actual) != em.finite(expected_value):
                    raise AssertionError("Baseline metric definedness differs")
                if em.finite(actual):
                    error = abs(actual-expected_value); max_errors[metric] = max(max_errors[metric], error)
                    if error > 1e-12: raise AssertionError(f"A {metric} replay differs by {error}; tolerance1e-12")
            row.update(direct[j])
            row["unclipped_self_delta_max_abs"] = 0.
        metric_seconds += time.monotonic()-tick
        for name, digest in input_hashes.items():
            if tensor_sha(x[name]) != digest: raise AssertionError(f"Frozen {name} changed")
        torch.save(dict(image_id=x["image_id"], identities=keys, q8_base=base_q.cpu(), q8_sam=torch.stack(sam_q),
                        c0=c0.cpu(), base_coefficient=coeffs["BASE_SOLVE"].cpu(), sam_coefficient=coeffs["SAM_SOLVE"].cpu(),
                        matrix_sha256=tensor_sha(matrix), input_sha256=input_hashes), saved/f"{int(x['image_id']):012d}.pt")
        em.append_rows(out/"PER_CANDIDATE.jsonl", decoded)
        em.append_rows(out/"INPUT_AUDIT.jsonl", [dict(image_id=x["image_id"], hashes=input_hashes,
                       matrix_sha256=tensor_sha(matrix), asset=x["_asset_integrity"], candidates=len(c0))])
        rows.extend(decoded); seen.update(keys)
        if position % 10 == 0 or position+1 == len(items):
            progress = dict(images_processed=position+1, candidates=len(rows), planned_images=len(items), elapsed_s=time.monotonic()-started)
            dump(out/"PROGRESS.json", progress); print(json.dumps(progress), flush=True)
        del x, gpu, proto, c0, boxes, matrix, base, coeffs, decoded, direct, sam_q, sam_coeff, base_q
    budget()
    if seen != set(prior) or len(rows) != expected["candidates"]:
        raise AssertionError("Fixed evaluation population incomplete")
    source_audit = source.finish(); budget()
    if source_audit["embedding_count"] != expected["effective_images"] or source_audit["candidates"] != expected["candidates"]:
        raise AssertionError("SAM must embed each effective image once and retain every candidate")
    audit = dict(population=expected, smoke=args.smoke, source=source_audit, no_training=True,
                 original_inputs_unchanged=True, baseline_max_errors=max_errors, baseline_tolerance=1e-12,
                 maximum_solver_backward_error=max_solver_error, unclipped_identity_exact=True,
                 no_evidence_decoder_exact=True, fallback_candidates=source_fallbacks,
                 solver_seconds=solver_seconds, metric_seconds=metric_seconds,
                 peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                 maximum_seconds=300 if args.smoke else 3600,
                 probability_path="both BASE and SAM: full640 logits -> sigmoid once -> crop_pool_7o -> clip -> logit -> solve",
                 matrix_path="frozen P160 -> legacy crop_pool_7o -> [64,32]; no cached16x16 operator")
    dump(out/"AUDIT.json", em.clean(audit))
    summarize(rows, out, audit, budget); budget()
    dump(out/"COMPLETE.json", dict(completed=True, passed=True, smoke=args.smoke, candidates=len(rows),
         elapsed_s=time.monotonic()-started, automatic_followup=False, no_training=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        target = Path(args.out); target.mkdir(parents=True, exist_ok=True)
        dump(target/"FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        if not (target/"COMPLETE.json").exists():
            dump(target/"COMPLETE.json", dict(completed=False, passed=False, smoke=args.smoke, no_training=True))
        raise


if __name__ == "__main__":
    main()
