"""Server-only frozen epoch-three cross-task bridge evaluation.

Replay original official identities and the unchanged full-prototype decoder.
No checkpoint selection, assignment, threshold adjustment, or extra training.
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

import torch
from pycocotools.coco import COCO
from online_runtime import FrozenReplay, dump, load_asset, load_index, load_json, resolve_runtime_config, sha256, tensor_sha
from joint_head import JointCoefficientReadout
from bridge_model import BridgeReadout
from prepare_features import load_feature_cache, selection_for
from dense_runtime import official_values, setup
import evaluation_metrics as em

ARMS = ("A", "N", "T", "R", "M", "TW")
PAIRS = (("T", "A"), ("T", "N"), ("T", "M"), ("T", "TW"),
         ("R", "A"), ("R", "N"), ("T", "R"), ("M", "A"), ("N", "A"))
METRICS = ("iou", "mask75", "coverage", "auc", "fpr", "bce")
IDENTITY = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
PAIRED_REQUIRED = ("seed", "epochs", "microbatch_images", "effective_batch_images", "branch_lr",
                   "weight_decay", "warmup_epochs", "eta_ratio", "gradient_clip_norm")
BOOTSTRAP, BOOTSTRAP_SEED, HISTORY_ATOL = 1000, 20261004, 1e-12
em.ARMS = ARMS


def identity(row):
    return tuple(row[key] for key in IDENTITY)


def digest(model):
    return {key: tensor_sha(value) for key, value in model.state_dict().items()}


def buffers(model):
    return {key: tensor_sha(value) for key, value in model.named_buffers()}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def add_input(inputs, path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    inputs[str(path)] = sha256(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_models(cfg, replay, bank, inputs):
    keys = tuple(cfg["paired_keys"])
    if not set(PAIRED_REQUIRED).issubset(keys):
        raise AssertionError("Paired settings omit a prespecified training-budget key")
    if set(cfg["checkpoints"]) != {"T", "R", "M"}:
        raise AssertionError("Exactly T/R/M fixed checkpoints are required")
    paths = {"N": Path(cfg["control_checkpoint"]), **{k: Path(v) for k, v in cfg["checkpoints"].items()}}
    if paths["N"].parent != Path(cfg["control_run"]):
        raise AssertionError("N must be the registered historical control")
    models, records, histories, native_initial = {}, {}, {}, {}
    source = digest(replay.source)
    for arm in ("N", "T", "R", "M"):
        path = paths[arm]
        if path.name != "final.pt":
            raise AssertionError("Only fixed final.pt is eligible")
        for item in (path, path.parent / "MODEL.json", path.parent / "COMPLETE.json", path.parent / "RESOLVED_CONFIG.json"):
            add_input(inputs, item)
        receipt, meta = load_json(path.parent / "COMPLETE.json"), load_json(path.parent / "MODEL.json")
        if receipt.get("completed") is not True or receipt.get("epochs") != 3 or receipt.get("mode") != arm:
            raise AssertionError(f"{arm}: epoch-three training is not complete")
        if receipt.get("all_frozen_states_unchanged") is not True:
            raise AssertionError(f"{arm}: missing frozen-state confirmation")
        for key in ("checkpoint_sha256", "final_sha256"):
            if receipt.get(key) and receipt[key] != inputs[str(path)]:
                raise AssertionError(f"{arm}: checkpoint/receipt hash disagreement")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("mode") != arm or checkpoint.get("epoch") != 3:
            raise AssertionError(f"{arm}: wrong checkpoint mode/epoch")
        if list(checkpoint["feature_channels"]) != list(replay.feature_channels):
            raise AssertionError(f"{arm}: frozen feature dimensions changed")
        if arm != "N":
            feature_inputs = {path: inputs[path] for path in (cfg["feature_cache"], str(cfg["feature_cache"]) + ".meta.json")}
            if checkpoint.get("feature_identity", {}).get("files") != feature_inputs:
                raise AssertionError(f"{arm}: training/evaluation feature-cache hashes differ")
        saved = resolve_runtime_config(checkpoint["config"])
        disk = resolve_runtime_config(load_json(path.parent / "RESOLVED_CONFIG.json"))
        for config in (saved, disk):
            for key in keys:
                if key not in config or config[key] != cfg[key]:
                    raise AssertionError(f"{arm}: unequal paired setting {key}")
            for key in ("weights", "source_python", "cache", "annotations_train"):
                if config[key] != cfg[key]:
                    raise AssertionError(f"{arm}: incompatible frozen input {key}")
            if load_json(config["split"]) != load_json(cfg["split"]):
                raise AssertionError(f"{arm}: split content differs")
            if arm != "N" and config["feature_cache"] != cfg["feature_cache"]:
                raise AssertionError(f"{arm}: detection feature cache differs")
            if arm != "N" and config.get("bridge_lr") != cfg["branch_lr"]:
                raise AssertionError(f"{arm}: bridge learning rate differs from the fixed paired rate")
        setup(cfg["seed"])
        if arm == "N":
            model = JointCoefficientReadout(replay.native_cv4, replay.feature_channels, "N", cfg)
        else:
            model = BridgeReadout(replay.native_cv4, replay.feature_channels, arm, cfg, bank["stats"])
        model = model.to(replay.device).float().eval()
        initial = digest(model)
        if meta.get("initialization_state_sha256") != canonical_hash(initial):
            raise AssertionError(f"{arm}: checkpoint initialization differs from reconstructed model")
        if meta.get("source_initial") != source:
            raise AssertionError(f"{arm}: original frozen source differs")
        native_initial[arm] = {key: value for key, value in initial.items() if key.startswith("native_cv4.")}
        if not native_initial[arm]:
            raise AssertionError("No original native coefficient branch found")
        before_buffers = buffers(model)
        counts = model.parameter_counts()
        if meta.get("counts") != counts:
            raise AssertionError(f"{arm}: declared/actual model dimensions or parameter counts differ")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        if before_buffers != buffers(model):
            raise AssertionError(f"{arm}: BN/statistics buffers changed during training")
        models[arm] = model
        histories[arm] = checkpoint.get("history", [])
        records[arm] = dict(path=str(path), sha256=inputs[str(path)], epoch=3, mode=arm,
            initialization_state_sha256=canonical_hash(initial), counts=counts, frozen_buffers=before_buffers,
            checkpoint_rule=checkpoint.get("checkpoint_rule"), receipt=receipt)
        del checkpoint
    if any(value != native_initial["N"] for value in native_initial.values()):
        raise AssertionError("Original native initialization is not paired")
    if records["T"]["counts"] != records["M"]["counts"]:
        raise AssertionError("T and M capacities differ")
    if records["T"]["initialization_state_sha256"] != records["M"]["initialization_state_sha256"]:
        raise AssertionError("T and M zero-bridge initializations differ")
    orders = []
    if any(len(value) != 3 for value in histories.values()):
        raise AssertionError("Every arm must retain all three epoch records")
    for epoch in range(1, 4):
        current = {arm: history[epoch-1] for arm, history in histories.items()}
        order = current["N"].get("image_order_sha256")
        if not order or any(row.get("epoch") != epoch or row.get("image_order_sha256") != order for row in current.values()):
            raise AssertionError("Training image orders are not paired")
        orders.append(dict(epoch=epoch, image_order_sha256=order))
    return models, records, dict(settings={k: cfg[k] for k in keys}, image_orders=orders,
        native_initialization_identical=True, T_M_exact_capacity_and_initialization_match=True,
        T_R_capacity_match_claimed=False)


def table(rows):
    groups = em.image_groups(rows)
    result = dict(images=len(groups), candidates=len(rows), candidate={}, image_macro={}, undefined={}, comparisons={})
    for metric in METRICS:
        result["candidate"][metric] = {a: em.avg([r.get(f"{metric}_{a}") for r in rows]) for a in ARMS}
        result["image_macro"][metric] = {a: em.avg([em.avg([r.get(f"{metric}_{a}") for r in g]) for g in groups]) for a in ARMS}
        result["undefined"][metric] = {a: sum(not em.finite(r.get(f"{metric}_{a}")) for r in rows) for a in ARMS}
    for arm, ref in PAIRS:
        comparison = {metric: em.paired(groups, arm, ref, metric, BOOTSTRAP_SEED, BOOTSTRAP) for metric in METRICS}
        repair = sum(r[f"mask75_{ref}"] == 0 and r[f"mask75_{arm}"] == 1 for r in rows)
        damage = sum(r[f"mask75_{ref}"] == 1 and r[f"mask75_{arm}"] == 0 for r in rows)
        comparison["crossings"] = dict(repair=repair, damage=damage, net=repair-damage, reference=ref)
        result["comparisons"][f"{arm}_minus_{ref}"] = comparison
    return result


def assessment(tables):
    checks = {}
    full = tables["dev"]["all"]["comparisons"]
    target = tables["dev"]["box_good_mask_bad"]["comparisons"]
    for ref in ("A", "N"):
        d, t = full[f"T_minus_{ref}"]["iou"]["image_macro"], target[f"T_minus_{ref}"]["iou"]["image_macro"]
        net = full[f"T_minus_{ref}"]["crossings"]["net"]
        overall = em.finite(d["delta"]) and d["delta"] >= .002 and em.finite(d["ci95"][0]) and d["ci95"][0] > 0 and net > 0
        targeted = em.finite(t["delta"]) and t["delta"] >= .005 and em.finite(t["ci95"][0]) and t["ci95"][0] > 0 and em.finite(d["delta"]) and d["delta"] >= -.001
        checks[ref] = dict(overall_rule=bool(overall), target_rule=bool(targeted), full_iou=d, target_iou=t, net_mask75=net)
    match_all = full["T_minus_M"]["iou"]["image_macro"]
    match_target = target["T_minus_M"]["iou"]["image_macro"]
    def positive(interval):
        return em.finite(interval["delta"]) and interval["delta"] > 0 and em.finite(interval["ci95"][0]) and interval["ci95"][0] > 0
    overall_route = all(v["overall_rule"] for v in checks.values()) and positive(match_all)
    target_route = all(v["target_rule"] for v in checks.values()) and positive(match_target)
    passed = overall_route or target_route
    return dict(prespecified_signal=bool(passed), status="prespecified_signal" if passed else "prespecified_screen_not_passed",
        overall_route=bool(overall_route), target_route=bool(target_route), checks=checks,
        T_minus_M_same_population=dict(overall=match_all, target=match_target), automatic_followup=False,
        other_meaningful_metrics_require_value_and_tradeoff_review=True,
        interpretation="Exploratory reused-development screen. T/R capacity differs; only T/M is an exact architecture control. No isolated subgroup automatically establishes a useful method.")


def summarize(rows, out, expected, audit):
    predicates = {"all": lambda r: True, "box_good_mask_bad": lambda r: bool(r["box_good_mask_bad"]),
        "original_success": lambda r: r["mask75_A"] == 1, "original_failure": lambda r: r["mask75_A"] == 0,
        "same_predicted_class_wrong_source": lambda r: bool(r["wrong_same_predicted_class"]),
        "same_predicted_class_wrong_source_target": lambda r: bool(r["wrong_same_predicted_class"]) and bool(r["box_good_mask_bad"]),
        **{f"pyramid_L{level}": lambda r, level=level: r["pyramid_level"] == level for level in range(3)}}
    tables, image_rows = {}, []
    for split in ("fit", "dev"):
        selected = [r for r in rows if r["split"] == split]
        tables[split] = {}
        for name, predicate in predicates.items():
            members = [r for r in selected if predicate(r)]
            tables[split][name] = table(members)
            for group in em.image_groups(members):
                record = dict(split=split, image_id=int(group[0]["image_id"]), group=name, candidates=len(group))
                for metric in METRICS:
                    for arm in ARMS:
                        record[f"{metric}_{arm}"] = em.avg([r.get(f"{metric}_{arm}") for r in group])
                        record[f"{metric}_{arm}_defined"] = sum(em.finite(r.get(f"{metric}_{arm}")) for r in group)
                    for arm, ref in PAIRS:
                        record[f"delta_{metric}_{arm}_minus_{ref}"] = em.avg([r[f"{metric}_{arm}"]-r[f"{metric}_{ref}"]
                            for r in group if em.finite(r.get(f"{metric}_{arm}")) and em.finite(r.get(f"{metric}_{ref}"))])
                image_rows.append(record)
    em.append_rows(out / "PER_IMAGE.jsonl", image_rows)
    result = dict(schema="cross-task-bridge-evaluation-v1", arms=list(ARMS), population=expected, tables=tables,
        assessment=assessment(tables), audit=audit, bootstrap=dict(draws=BOOTSTRAP, seed=BOOTSTRAP_SEED, unit="paired whole images"),
        primary="dev all image-macro original-image Mask IoU; T-A, T-N, T-M",
        no_COCO_AP=True, no_new_blind_test=True, automatic_followup=False)
    dump(out / "SUMMARY.json", em.clean(result))
    def pp(v):
        return "undefined" if not em.finite(v) else f"{100*v:+.4f}"
    lines = ["# Cross-task coefficient bridge: fixed epoch-three screen", "",
        "A is original YOLO; N is historical ordinary native fine-tuning. T reads frozen classification hidden features; R reads frozen regression hidden features; M trains with the same architecture as T but a fixed wrong-image source. TW reuses frozen T parameters with M's wrong source, without training or changing the candidate.", "",
        "Primary evidence is normal original-image mask quality. This is reused-development evaluation of GT-conditioned fixed official candidates, not a new blind test or COCO AP result. T/R have different input dimensions and are not a matched-capacity mechanism contrast.", "",
        "Deltas below are percentage points. Each confidence interval resamples whole images 1,000 times. Fit is the first 128 planned images; dev uses all 256 planned images. No-positive images remain in the manifest.", "",
        "| Split / group | Images / candidates | Pair | Macro IoU delta [95% CI] | Candidate IoU delta | Repair / damage | Coverage delta | AUC delta | FPR delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for split, groups in tables.items():
        for group, values in groups.items():
            # Full machine-readable tables retain every comparison; report avoids repeating secondary rows.
            pairs = ("T_minus_TW",) if group.startswith("same_predicted_class") else ("T_minus_A", "T_minus_N", "T_minus_M", "R_minus_N", "T_minus_TW")
            for pair in pairs:
                comp = values["comparisons"][pair]
                d, cross = comp["iou"]["image_macro"], comp["crossings"]
                lines.append(f"| {split} / {group} | {values['images']} / {values['candidates']} | {pair} | {pp(d['delta'])} [{pp(d['ci95'][0])}, {pp(d['ci95'][1])}] | {pp(comp['iou']['candidate']['delta'])} | {cross['repair']} / {cross['damage']} | {pp(comp['coverage']['image_macro']['delta'])} | {pp(comp['auc']['image_macro']['delta'])} | {pp(comp['fpr']['image_macro']['delta'])} |")
    lines += ["", "## Official diagnostic loss", "", "The same complete sparse official mask BCE, gain and normalization are used for every arm. No teacher or output regularizer is added.", "",
        "| Split | Candidate BCE (A / N / T / R / M / TW) | Image-macro BCE (same order) |", "|---|---|---|"]
    for split in ("fit", "dev"):
        values = tables[split]["all"]
        def fmt(v):
            return "undefined" if not em.finite(v) else f"{v:.8f}"
        lines.append(f"| {split} | {' / '.join(fmt(values['candidate']['bce'][a]) for a in ARMS)} | {' / '.join(fmt(values['image_macro']['bce'][a]) for a in ARMS)} |")
    lines += ["", "## Decision and validity", "",
        f"Prespecified screen status: **{result['assessment']['status']}**. Other metric improvements require an explicit magnitude and damage review; this script does not authorize a new experiment.", "",
        "T-M compares equally trained identical architectures. T-TW is a frozen-parameter input intervention; it can expose reliance on source correspondence but alone cannot establish superiority over the baseline. Wrong-source fallback cases are excluded from the separately reported same-predicted-class subset; matching is based on original predicted class, never GT class.", "",
        "A/N dev replay is checked at every permanent identity and all five historical mask metrics within 1e-12; original coefficient replay uses atol=rtol=3e-5. Full prototypes, original predicted boxes, process_mask(upsample=True), and the original letterbox inverse are unchanged. AUC uses continuous input-grid logits and the fixed prediction-box support. Undefined AUC/FPR retain their IoU/Mask75 evaluation.", "",
        "Full input hashes, checkpoint/initialization/order checks, buffer checks, donor counts and numeric replay errors are in AUDIT.json; all pair/group metrics are in SUMMARY.json. Layer indices L0/L1/L2 denote P3/P4/P5, not object-size classes.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def run(args):
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Evaluation is permitted only on the authorized Linux CUDA server")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    cfg = resolve_runtime_config(load_json(args.config))
    deadline = started + float(cfg["evaluation_max_seconds"])
    def budget():
        if time.monotonic() > deadline:
            raise TimeoutError("Fixed evaluation budget exceeded; no automatic resume")
    setup(cfg["seed"])
    if cfg.get("epochs") != 3 or cfg.get("bootstrap") != BOOTSTRAP:
        raise AssertionError("Only fixed epoch-three checkpoints and 1,000 bootstrap draws are allowed")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("PER_CANDIDATE.jsonl", "SUMMARY.json", "COMPLETE.json")):
        raise RuntimeError("Use a new independent Run; existing results cannot be overwritten")
    inputs = {}
    for path in (args.config, cfg["weights"], cfg["annotations_train"], cfg["prior_results"], cfg["feature_cache"],
                 cfg["split"], Path(cfg["cache"]) / "INDEX.json", Path(cfg["cache"]) / "CACHE_IDENTITY.json"):
        add_input(inputs, path)
    feature_meta = Path(str(cfg["feature_cache"]) + ".meta.json")
    add_input(inputs, feature_meta)
    bank = load_feature_cache(cfg)
    index = load_index(cfg)
    if len(index["fit"]) != 1024 or len(index["dev"]) != 256 or index.get("val"):
        raise AssertionError("Original fit/dev image manifest changed")
    selected = {"fit": index["fit"][:128], "dev": index["dev"]}
    expected = {}
    for split, entries in selected.items():
        ids = [int(e["image_id"]) for e in entries]
        if len(ids) != len(set(ids)):
            raise AssertionError("Repeated image identity")
        expected[split] = dict(planned_images=len(entries), effective_images=sum(int(e["n"]) > 0 for e in entries),
            candidates=sum(int(e["n"]) for e in entries), no_positive_image_ids=[int(e["image_id"]) for e in entries if not int(e["n"])])
    dump(out / "EVALUATION_INDEX.json", selected)
    previous = read_rows(cfg["prior_results"])
    prior = {identity(row): row for row in previous}
    if len(prior) != len(previous):
        raise AssertionError("Historical candidate identities are duplicated")
    replay = FrozenReplay(cfg)
    source_before = digest(replay.source)
    models, checkpoint_audit, paired_audit = load_models(cfg, replay, bank, inputs)
    model_before = {a: digest(m) for a, m in models.items()}
    buffers_before = {a: buffers(m) for a, m in models.items()}
    coco = COCO(cfg["annotations_train"])
    rows, seen, history_keys, assets = [], set(), set(), []
    history_errors = {f"{metric}_{arm}": 0.0 for arm in ("A", "N") for metric in METRICS[:-1]}
    coefficient_error = 0.0
    donor_counts = {split: dict(same_predicted_class=0, fallback=0) for split in selected}
    for split, entries in selected.items():
        for position, entry in enumerate(entries):
            budget()
            if not int(entry["n"]):
                continue
            iid = int(entry["image_id"])
            image = load_asset(cfg, iid, verify=True)
            if image["split"] != split or len(image["rows"]) != int(entry["n"]):
                raise AssertionError("Frozen candidate count or split differs")
            keys = [identity(r) for r in image["rows"]]
            if len(keys) != len(set(keys)) or seen.intersection(keys) or any(k[3] != "one2one" for k in keys):
                raise AssertionError("Invalid or repeated original one-to-one candidate identities")
            asset = Path(cfg["cache"]) / "images" / f"{iid:012d}.pt.gz"
            inputs[str(asset)] = image["_asset_integrity"]["compressed_sha256"]
            add_input(inputs, asset.with_name(f"{iid:012d}.meta.json"))
            donor = bank["wrong"][iid]
            required_donor = ("source_image_ids", "source_row_indices", "source_raw_ids", "same_predicted_class", "fallback", "donor_pred_class")
            if any(key not in donor or len(donor[key]) != len(keys) for key in required_donor):
                raise AssertionError("Incomplete fixed wrong-source provenance")
            if any(int(x) == iid for x in donor["source_image_ids"]):
                raise AssertionError("Wrong source is not from another image")
            if any(bool(a) == bool(b) for a, b in zip(donor["same_predicted_class"], donor["fallback"])):
                raise AssertionError("Wrong-source class-match/fallback flags disagree")
            with torch.no_grad():
                features = replay.replay([image])
                original = replay.native_outputs(features, [image])[0]["c"]
                c0 = image["c0"].to(replay.device)
                error = float((original-c0).abs().max())
                coefficient_error = max(coefficient_error, error)
                torch.testing.assert_close(original, c0, atol=3e-5, rtol=3e-5)
                coefficients = {"A": c0}
                coefficients["N"] = models["N"](features, [{"raw_ids": image["raw_ids"].to(replay.device)}])[0]
                for arm in ("T", "R", "M"):
                    coefficients[arm] = models[arm](features, [selection_for(image, bank, arm, device=replay.device)])[0]
                coefficients["TW"] = models["T"](features, [selection_for(image, bank, "M", device=replay.device)])[0]
                gpu = {key: value.to(replay.device) if torch.is_tensor(value) else value for key, value in image.items()
                       if key not in ("F", "input_uint8", "operator", "_operator")}
                bce = {arm: official_values(gpu, value).cpu() for arm, value in coefficients.items()}
                if any(tuple(v.shape) != (len(keys),) or not bool(torch.isfinite(v).all()) for v in bce.values()):
                    raise AssertionError("Official BCE must retain every ordered candidate with a finite value")
                decoded = em.evaluate_image(gpu, coefficients, coco)
            if [identity(r) for r in decoded] != keys:
                raise AssertionError("Normal decoder changed candidate identities/order")
            for i, row in enumerate(decoded):
                key = identity(row)
                for arm in ARMS:
                    row[f"bce_{arm}"] = float(bce[arm][i])
                row.update(wrong_source_image_id=int(donor["source_image_ids"][i]),
                    wrong_source_row_index=int(donor["source_row_indices"][i]), wrong_source_raw_id=int(donor["source_raw_ids"][i]),
                    wrong_same_predicted_class=bool(donor["same_predicted_class"][i]),
                    wrong_fallback=bool(donor["fallback"][i]), wrong_donor_pred_class=int(donor["donor_pred_class"][i]),
                    original_pred_class=int(bank["images"][iid]["pred_class"][i]))
                donor_counts[split]["same_predicted_class" if row["wrong_same_predicted_class"] else "fallback"] += 1
                if split == "dev":
                    if key not in prior:
                        raise AssertionError(f"Missing historical candidate: {key}")
                    history_keys.add(key)
                    for field in history_errors:
                        old, new = prior[key].get(field), row.get(field)
                        if em.finite(old) != em.finite(new):
                            raise AssertionError(f"Historical metric definedness changed: {key} {field}")
                        if em.finite(old):
                            delta = abs(float(old)-float(new))
                            history_errors[field] = max(history_errors[field], delta)
                            if delta > HISTORY_ATOL:
                                raise AssertionError(f"Historical replay mismatch: {key} {field} {delta}")
            assets.append(dict(split=split, image_id=iid, candidates=len(keys), integrity=image["_asset_integrity"],
                prototype_sha256=tensor_sha(image["proto"])))
            em.append_rows(out / "PER_CANDIDATE.jsonl", decoded)
            em.append_rows(out / "REPLAY_AUDIT.jsonl", [dict(split=split, image_id=iid, candidates=len(keys),
                original_coefficient_max_abs_error=error, coefficient_sha256={a: tensor_sha(v) for a, v in coefficients.items()})])
            rows.extend(decoded)
            seen.update(keys)
            if position % 20 == 0 or position+1 == len(entries):
                progress = dict(stage="normal_image_evaluation", split=split, planned_position=position+1,
                    planned_images=len(entries), candidates=len(rows), elapsed_s=time.monotonic()-started)
                dump(out / "PROGRESS.json", progress)
                print(json.dumps(progress), flush=True)
            del image, features, original, c0, coefficients, gpu, bce, decoded
    for split in selected:
        actual = [row for row in rows if row["split"] == split]
        if len(actual) != expected[split]["candidates"] or len(em.image_groups(actual)) != expected[split]["effective_images"]:
            raise AssertionError("Incomplete fixed evaluation population")
    dev_ids = {int(e["image_id"]) for e in selected["dev"]}
    if history_keys != {key for key in prior if key[0] == "dev" and key[1] in dev_ids}:
        raise AssertionError("Historical and current dev candidate sets differ")
    replay.assert_unchanged()
    if source_before != digest(replay.source):
        raise AssertionError("Original source parameters or buffers changed")
    for arm, model in models.items():
        if digest(model) != model_before[arm] or buffers(model) != buffers_before[arm]:
            raise AssertionError(f"Frozen model changed during evaluation: {arm}")
        if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise AssertionError("Evaluation enabled train mode or created parameter gradients")
    for path, expected_sha in inputs.items():
        if sha256(path) != expected_sha:
            raise AssertionError(f"Frozen input changed: {path}")
    audit = dict(passed=True, source=replay.import_info, checkpoints=checkpoint_audit, paired_training=paired_audit,
        historical_metric_max_abs_errors=history_errors, historical_metric_atol=HISTORY_ATOL,
        historical_candidates_reproduced=len(history_keys), original_coefficient_max_abs_error=coefficient_error,
        original_coefficient_atol=3e-5, original_coefficient_rtol=3e-5, wrong_source_counts=donor_counts,
        same_class_subset_uses_predicted_class_only=True, fallback_excluded_from_class_control=True,
        all_source_parameters_and_buffers_unchanged=True, all_evaluation_models_unchanged=True, all_inputs_unchanged=True,
        assignment_recomputed=False, empty_masks_removed=False, input_sha256=inputs, assets=assets,
        decode="Unmodified evaluation_metrics.py: full P, frozen prediction boxes, process_mask(upsample=True), original letterbox inverse",
        auc_fpr="Continuous input-grid logits; fixed prediction-box support; padded original COCO labels")
    dump(out / "AUDIT.json", em.clean(audit))
    dump(out / "INPUT_MANIFEST.json", inputs)
    budget()
    result = summarize(rows, out, expected, audit)
    budget()
    torch.cuda.synchronize()
    dump(out / "COMPLETE.json", dict(completed=True, passed=True, kind="normal_image_evaluation", epochs=3,
        arms=list(ARMS), candidates=len(rows), population=expected, elapsed_s=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        input_sha256=inputs, scientific_status=result["assessment"]["status"], automatic_followup=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        dump(Path(args.out) / "FAILURE.json", dict(error_type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == "__main__":
    main()
