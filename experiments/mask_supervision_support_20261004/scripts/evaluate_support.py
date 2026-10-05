"""Frozen epoch-3 support-redistribution evaluation on the authorized server.

Only the original sparse official candidate identities are evaluated. This
script neither reassigns labels nor chooses an epoch, support, or threshold.
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
    _config = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text(encoding="utf-8-sig"))
    if _config.get("source_python"):
        sys.path.insert(0, _config["source_python"])

import torch
from pycocotools.coco import COCO

from online_runtime import FrozenReplay, dump, load_asset, load_index, load_json, resolve_runtime_config, sha256, tensor_sha
from joint_head import JointCoefficientReadout
from dense_runtime import official_values, setup
import evaluation_metrics as em

ARMS = ("A", "N", "D")
PAIRS = (("D", "A"), ("D", "N"), ("N", "A"))
METRICS = ("iou", "mask75", "coverage", "auc", "fpr", "bce")
IDENTITY = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
REQUIRED_PAIRED = ("seed", "epochs", "microbatch_images", "effective_batch_images", "branch_lr",
                   "weight_decay", "warmup_epochs", "eta_ratio", "gradient_clip_norm")
BOOTSTRAP = 1000
SEED = 20261004
HISTORY_TOLERANCE = 1e-12
em.ARMS = ARMS


def identity(row):
    return tuple(row[key] for key in IDENTITY)


def state_digest(model):
    return {name: tensor_sha(value) for name, value in model.state_dict().items()}


def buffer_digest(model):
    return {name: tensor_sha(value) for name, value in model.named_buffers()}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def gpu_payload(image, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in image.items() if key not in ("F", "input_uint8", "operator", "_operator")}


def add_input(inputs, path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    inputs[str(path)] = sha256(path)


def verify_inputs(inputs):
    for path, expected in inputs.items():
        if sha256(path) != expected:
            raise AssertionError(f"Frozen input changed: {path}")


def load_models(cfg, replay, inputs):
    keys = tuple(cfg["paired_keys"])
    if not set(REQUIRED_PAIRED).issubset(keys):
        raise ValueError("paired_keys must include every prespecified optimizer/budget key")
    paths = {"N": Path(cfg["control_checkpoint"]), "D": Path(cfg["method_checkpoint"])}
    if paths["N"].parent != Path(cfg["control_run"]):
        raise AssertionError("Control checkpoint does not belong to registered control Run")
    models, records, checkpoint_configs, histories = {}, {}, {}, {}
    for arm, path in paths.items():
        if path.name != "final.pt":
            raise ValueError("Only the fixed final checkpoint is eligible")
        for filename in ("COMPLETE.json", "MODEL.json", "RESOLVED_CONFIG.json"):
            add_input(inputs, path.parent / filename)
        add_input(inputs, path)
        receipt = load_json(path.parent / "COMPLETE.json")
        if receipt.get("completed") is not True or receipt.get("epochs") != 3:
            raise AssertionError(f"{arm}: final epoch-3 training has not completed")
        if receipt.get("all_frozen_states_unchanged") is not True:
            raise AssertionError(f"{arm}: training did not certify immutable source/buffers")
        for key in ("checkpoint_sha256", "final_sha256"):
            if receipt.get(key) and receipt[key] != inputs[str(path)]:
                raise AssertionError(f"{arm}: checkpoint hash differs from training receipt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("mode") != arm or checkpoint.get("epoch") != 3:
            raise AssertionError(f"{arm}: unexpected checkpoint mode or epoch")
        if list(checkpoint["feature_channels"]) != list(replay.feature_channels):
            raise AssertionError(f"{arm}: frozen feature channels differ")
        saved_config = resolve_runtime_config(checkpoint["config"])
        disk_config = resolve_runtime_config(load_json(path.parent / "RESOLVED_CONFIG.json"))
        for key in keys:
            if key not in saved_config or saved_config[key] != cfg[key] or disk_config.get(key) != cfg[key]:
                raise AssertionError(f"{arm}: unequal paired training setting {key}")
        for key in ("weights", "source_python", "cache", "annotations_train"):
            if saved_config[key] != cfg[key] or disk_config[key] != cfg[key]:
                raise AssertionError(f"{arm}: incompatible fixed asset {key}")
        model = JointCoefficientReadout(replay.native_cv4, replay.feature_channels, "N", cfg).to(replay.device).float().eval()
        if load_json(saved_config['split']) != load_json(cfg['split']) or load_json(disk_config['split']) != load_json(cfg['split']):
            raise AssertionError(f'{arm}: frozen split contents differ')
        initialization_hash = canonical_hash(state_digest(model))
        model_record = load_json(path.parent / "MODEL.json")
        if model_record.get("initialization_state_sha256") != initialization_hash:
            raise AssertionError(f"{arm}: initialization differs from the original native branch")
        if model_record.get("source_initial") != state_digest(replay.source):
            raise AssertionError(f"{arm}: training used a different immutable source model")
        original_buffers = buffer_digest(model)
        counts = model.parameter_counts()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        if buffer_digest(model) != original_buffers:
            raise AssertionError(f"{arm}: final checkpoint modified frozen normalization buffers")
        if model.evidence is not None or counts["new"] != 0:
            raise AssertionError("Both arms must be the same native-only readout")
        models[arm] = model
        checkpoint_configs[arm] = saved_config
        histories[arm] = checkpoint.get("history", [])
        records[arm] = dict(path=str(path), sha256=inputs[str(path)], mode=arm, internal_architecture_mode="N",
                            epoch=3, initialization_state_sha256=initialization_hash, counts=counts,
                            buffers=original_buffers, receipt=receipt)
        del checkpoint
    if records["N"]["initialization_state_sha256"] != records["D"]["initialization_state_sha256"]:
        raise AssertionError("Native initialization is unpaired")
    if records["N"]["counts"] != records["D"]["counts"]:
        raise AssertionError("Native model capacities differ")
    if len(histories["N"]) != 3 or len(histories["D"]) != 3:
        raise AssertionError("Need all three completed epoch histories")
    order_records = []
    for epoch, (control, method) in enumerate(zip(histories["N"], histories["D"]), 1):
        if control.get("epoch") != epoch or method.get("epoch") != epoch:
            raise AssertionError("Checkpoint history has a missing or reordered epoch")
        order = control.get("image_order_sha256")
        if not order or order != method.get("image_order_sha256"):
            raise AssertionError("Paired training image order differs")
        order_records.append(dict(epoch=epoch, image_order_sha256=order))
    return models, records, dict(paired_keys={key: cfg[key] for key in keys},
                                 paired_image_order=order_records, checkpoint_configs=checkpoint_configs)


def table(rows):
    groups = em.image_groups(rows)
    result = dict(images=len(groups), candidates=len(rows), candidate={}, image_macro={}, undefined={}, comparisons={})
    for metric in METRICS:
        result["candidate"][metric] = {arm: em.avg([row.get(f"{metric}_{arm}") for row in rows]) for arm in ARMS}
        result["image_macro"][metric] = {arm: em.avg([em.avg([row.get(f"{metric}_{arm}") for row in image]) for image in groups]) for arm in ARMS}
        result["undefined"][metric] = {arm: sum(not em.finite(row.get(f"{metric}_{arm}")) for row in rows) for arm in ARMS}
    for arm, ref in PAIRS:
        comparison = {metric: em.paired(groups, arm, ref, metric, SEED, BOOTSTRAP) for metric in METRICS}
        repair = sum(row[f"mask75_{ref}"] == 0 and row[f"mask75_{arm}"] == 1 for row in rows)
        damage = sum(row[f"mask75_{ref}"] == 1 and row[f"mask75_{arm}"] == 0 for row in rows)
        comparison["crossings"] = dict(repair=repair, damage=damage, net=repair-damage,
                                        reference=ref, candidates=len(rows))
        comparison["iou_changes"] = dict(improve=sum(row[f"iou_{arm}"] > row[f"iou_{ref}"] for row in rows),
                                          worsen=sum(row[f"iou_{arm}"] < row[f"iou_{ref}"] for row in rows),
                                          unchanged=sum(row[f"iou_{arm}"] == row[f"iou_{ref}"] for row in rows))
        result["comparisons"][f"{arm}_minus_{ref}"] = comparison
    return result


def signal(tables, smoke):
    checks = {}
    for ref in ("A", "N"):
        full = tables["dev"]["all"]["comparisons"][f"D_minus_{ref}"]
        target = tables["dev"]["box_good_mask_bad"]["comparisons"][f"D_minus_{ref}"]
        d, t = full["iou"]["image_macro"], target["iou"]["image_macro"]
        full_pass = (em.finite(d["delta"]) and d["delta"] >= .002 and em.finite(d["ci95"][0])
                     and d["ci95"][0] > 0 and full["crossings"]["net"] > 0)
        target_pass = (em.finite(t["delta"]) and t["delta"] >= .005 and em.finite(t["ci95"][0])
                       and t["ci95"][0] > 0 and em.finite(d["delta"]) and d["delta"] >= -.001)
        checks[ref] = dict(overall_rule=bool(full_pass), target_rule=bool(target_pass),
                           overall_macro_iou=d, target_macro_iou=t, all_crossings=full["crossings"])
    positive = all(x["overall_rule"] for x in checks.values()) or all(x["target_rule"] for x in checks.values())
    return dict(status="smoke_only_no_scientific_decision" if smoke else
                "prespecified_IoU_screen_signal" if positive else "prespecified_IoU_screen_not_passed",
                prespecified_signal=bool(positive) if not smoke else None, checks=checks,
                automatic_followup=False, additional_metrics_require_value_and_tradeoff_review=True,
                interpretation="An exploratory reused-development screen. All metrics need not improve; isolated gains cannot hide material damage. No new experiment is automatically authorized by this status.")


def summarize(rows, out, expected, audit, smoke):
    predicates = {"all": lambda r: True,
                  "box_good_mask_bad": lambda r: bool(r["box_good_mask_bad"]),
                  "original_success": lambda r: r["mask75_A"] == 1,
                  "original_failure": lambda r: r["mask75_A"] == 0,
                  **{f"pyramid_L{level}": lambda r, level=level: r["pyramid_level"] == level for level in range(3)}}
    tables, image_rows = {}, []
    for split in ("fit", "dev"):
        tables[split] = {}
        split_rows = [row for row in rows if row["split"] == split]
        for name, predicate in predicates.items():
            selected = [row for row in split_rows if predicate(row)]
            tables[split][name] = table(selected)
            for group in em.image_groups(selected):
                record = dict(split=split, image_id=int(group[0]["image_id"]), group=name, candidates=len(group))
                for metric in METRICS:
                    for arm in ARMS:
                        record[f"{metric}_{arm}"] = em.avg([r.get(f"{metric}_{arm}") for r in group])
                        record[f"{metric}_{arm}_defined"] = sum(em.finite(r.get(f"{metric}_{arm}")) for r in group)
                    for arm, ref in PAIRS:
                        record[f"delta_{metric}_{arm}_minus_{ref}"] = em.avg([
                            r[f"{metric}_{arm}"] - r[f"{metric}_{ref}"] for r in group
                            if em.finite(r.get(f"{metric}_{arm}")) and em.finite(r.get(f"{metric}_{ref}"))])
                image_rows.append(record)
    em.append_rows(out / "PER_IMAGE.jsonl", image_rows)
    result = dict(schema="mask-supervision-support-evaluation-v1", smoke=smoke, arms=list(ARMS),
                  population=expected, tables=tables, assessment=signal(tables, smoke), audit=audit,
                  bootstrap=dict(draws=BOOTSTRAP, seed=SEED, unit="paired whole images"),
                  no_COCO_AP=True, no_new_blind_test=True, automatic_followup=False)
    dump(out / "SUMMARY.json", em.clean(result))
    def pp(value):
        return "undefined" if not em.finite(value) else f"{value*100:+.4f}"
    lines = ["# Mask supervision support redistribution: frozen epoch-3 evaluation", "",
             "This is a reused-development diagnostic, not an independent confirmation or COCO AP result. Only original sparse official one-to-one candidate identities are evaluated; no assignment, threshold, gate or checkpoint is selected here.", "",
             "A is original YOLO; N is the completed ordinary native-branch control; D changes the fixed training supervision support while preserving the native architecture and training budget. The support change is not a claim of pure sample-count causality.", "",
             "The main metric is dev image-macro original-image Mask IoU. Candidate means also resample whole images. All deltas below are percentage points; CIs are descriptive 1,000-image-bootstrap intervals.", "",
             "| Split / group | Images / candidates | Comparison | Macro IoU delta [95% CI] | Candidate IoU delta | Mask75 repair / damage | Coverage delta | AUC delta | FPR delta |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for split, groups in tables.items():
        for group, item in groups.items():
            for pair, comparison in item["comparisons"].items():
                val = comparison["iou"]["image_macro"]
                cross = comparison["crossings"]
                lines.append(f"| {split} / {group} | {item['images']} / {item['candidates']} | {pair} | "
                             f"{pp(val['delta'])} [{pp(val['ci95'][0])}, {pp(val['ci95'][1])}] | "
                             f"{pp(comparison['iou']['candidate']['delta'])} | {cross['repair']} / {cross['damage']} | "
                             f"{pp(comparison['coverage']['image_macro']['delta'])} | {pp(comparison['auc']['image_macro']['delta'])} | {pp(comparison['fpr']['image_macro']['delta'])} |")
    lines += ["", "## Original sparse diagnostic loss", "",
              "BCE uses the original sparse official GT-mask support, original gain and normalization for all arms, even though D trained with a different support set. It is not the dense training trajectory loss.", "",
              "| Split | Candidate BCE A / N / D | Image-macro BCE A / N / D |", "|---|---|---|"]
    for split in ("fit", "dev"):
        item = tables[split]["all"]
        def fmt(value):
            return "undefined" if not em.finite(value) else f"{value:.8f}"
        lines.append(f"| {split} | {' / '.join(fmt(item['candidate']['bce'][a]) for a in ARMS)} | "
                     f"{' / '.join(fmt(item['image_macro']['bce'][a]) for a in ARMS)} |")
    lines += ["", "## Frozen-state and interpretation checks", "",
              "- A and N replay the historical dev metrics at the exact permanent candidate identities; maximum allowable absolute error is 1e-12. Original coefficient replay uses atol=rtol=3e-5.",
              "- Full P and the frozen predicted box feed the existing process_mask(upsample=True) decoder and true letterbox inverse. Original COCO instance masks evaluate IoU/coverage; continuous input-grid logits and the fixed prediction-box support evaluate AUC/FPR.",
              "- AUC/FPR undefined cases remain in IoU and Mask75 evaluation. Original success/failure and the box-good-mask-bad subgroup are defined by A. A pyramid level is not an object-size class.",
              "- The fit evaluation is the first 128 planned fit images, not the full 1,024-image training set. Smoke uses two effective images per split and makes no scientific decision.",
              f"- Prespecified screen status: **{result['assessment']['status']}**. Other meaningful metrics require explicit magnitude, uncertainty and trade-off review. No automatic new run, expanded data, or extra training follows.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def run(args):
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Model evaluation is permitted only on the authorized Linux CUDA server")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    cfg = resolve_runtime_config(load_json(args.config))
    deadline = started + float(cfg['evaluation_max_seconds'])
    def budget():
        if time.monotonic() > deadline:
            raise TimeoutError('Fixed evaluation time budget exceeded; no automatic resume')
    setup(cfg['seed'])
    if cfg.get("epochs") != 3 or cfg.get("bootstrap") != BOOTSTRAP:
        raise AssertionError("Only fixed epoch3 and 1000 bootstrap are permitted")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("PER_CANDIDATE.jsonl", "SUMMARY.json", "COMPLETE.json")):
        raise RuntimeError("Use a new independent Run; prior outputs cannot be overwritten")
    inputs = {}
    for path in (args.config, cfg["weights"], cfg["annotations_train"], cfg["prior_results"],
                 Path(cfg["cache"]) / "INDEX.json", Path(cfg["cache"]) / "CACHE_IDENTITY.json"):
        add_input(inputs, path)
    index = load_index(cfg)
    if len(index["fit"]) != 1024 or len(index["dev"]) != 256 or index.get("val"):
        raise AssertionError("Original fit/dev manifest changed")
    selected = {"fit": index["fit"][:128], "dev": index["dev"]}
    if args.smoke:
        selected = {split: [entry for entry in index[split] if int(entry["n"]) > 0][:2]
                    for split in ("fit", "dev")}
        if any(len(entries) != 2 for entries in selected.values()):
            raise AssertionError("Smoke needs two effective images per split")
    expected = {}
    for split, entries in selected.items():
        ids = [int(x["image_id"]) for x in entries]
        if len(ids) != len(set(ids)):
            raise AssertionError("Duplicate image identity")
        expected[split] = dict(planned_images=len(entries), effective_images=sum(int(x["n"]) > 0 for x in entries),
                               candidates=sum(int(x["n"]) for x in entries),
                               no_positive_image_ids=[int(x["image_id"]) for x in entries if not int(x["n"])])
    dump(out / "EVALUATION_INDEX.json", selected)
    prior_rows = read_jsonl(cfg["prior_results"])
    prior = {identity(row): row for row in prior_rows}
    if len(prior) != len(prior_rows):
        raise AssertionError("Historical evaluation contains duplicate permanent candidate keys")
    replay = FrozenReplay(cfg)
    source_before = state_digest(replay.source)
    models, checkpoints, paired_config = load_models(cfg, replay, inputs)
    model_before = {arm: state_digest(model) for arm, model in models.items()}
    model_buffers = {arm: buffer_digest(model) for arm, model in models.items()}
    coco = COCO(cfg["annotations_train"])
    rows, seen, assets, replay_rows = [], set(), [], []
    max_history_error = {f"{metric}_{arm}": 0.0 for arm in ("A", "N") for metric in METRICS[:-1]}
    historical_keys = set()
    max_original_coefficient_error = 0.0
    for split, entries in selected.items():
        for position, entry in enumerate(entries):
            budget()
            if not int(entry["n"]):
                continue
            image = load_asset(cfg, int(entry["image_id"]), verify=True)
            if image["split"] != split or len(image["rows"]) != int(entry["n"]):
                raise AssertionError("Frozen split or candidate count changed")
            keys = [identity(row) for row in image["rows"]]
            if len(set(keys)) != len(keys) or seen.intersection(keys):
                raise AssertionError("Duplicate candidate identity")
            for key in keys:
                if key[3] != "one2one":
                    raise AssertionError("Only original one-to-one identities are evaluated")
            asset_path = Path(cfg["cache"]) / "images" / f"{int(entry['image_id']):012d}.pt.gz"
            inputs[str(asset_path)] = image["_asset_integrity"]["compressed_sha256"]
            add_input(inputs, asset_path.with_name(f"{int(entry['image_id']):012d}.meta.json"))
            with torch.no_grad():
                features = replay.replay([image])
                original = replay.native_outputs(features, [image])[0]["c"]
                c0 = image["c0"].to(replay.device)
                error = float((original - c0).abs().max())
                max_original_coefficient_error = max(max_original_coefficient_error, error)
                torch.testing.assert_close(original, c0, atol=3e-5, rtol=3e-5)
                coefficients = {"A": c0}
                selection = [{"raw_ids": image["raw_ids"].to(replay.device)}]
                for arm, model in models.items():
                    coefficients[arm] = model(features, selection)[0]
                gpu = gpu_payload(image, replay.device)
                bce = {arm: official_values(gpu, value).detach().cpu() for arm, value in coefficients.items()}
                if any(tuple(value.shape) != (len(keys),) or not bool(torch.isfinite(value).all()) for value in bce.values()):
                    raise AssertionError("Sparse official BCE must be finite and ordered one value per original candidate")
                decoded = em.evaluate_image(gpu, coefficients, coco)
            if [identity(row) for row in decoded] != keys:
                raise AssertionError("Decoder changed fixed candidate identities/order")
            for index_in_image, row in enumerate(decoded):
                key = identity(row)
                for arm in ARMS:
                    row[f"bce_{arm}"] = float(bce[arm][index_in_image])
                if split == "dev":
                    if key not in prior:
                        raise AssertionError(f"Historical A/N candidate missing: {key}")
                    historical_keys.add(key)
                    for field in max_history_error:
                        old, new = prior[key].get(field), row.get(field)
                        if em.finite(old) != em.finite(new):
                            raise AssertionError(f"Historical metric definedness differs for {key}: {field}")
                        if em.finite(old):
                            difference = abs(float(old) - float(new))
                            max_history_error[field] = max(max_history_error[field], difference)
                            if difference > HISTORY_TOLERANCE:
                                raise AssertionError(f"Historical replay exceeds fixed tolerance: {key} {field} {difference}")
            assets.append(dict(split=split, image_id=int(entry["image_id"]), candidates=len(keys),
                               integrity=image["_asset_integrity"], prototype_sha256=tensor_sha(image["proto"])))
            replay_rows.append(dict(split=split, image_id=int(entry["image_id"]), candidates=len(keys),
                                    original_coefficient_max_abs_error=error,
                                    coefficients_sha256={arm: tensor_sha(value) for arm, value in coefficients.items()}))
            em.append_rows(out / "PER_CANDIDATE.jsonl", decoded)
            em.append_rows(out / "REPLAY_AUDIT.jsonl", [replay_rows[-1]])
            rows.extend(decoded)
            seen.update(keys)
            if position % 20 == 0 or position + 1 == len(entries):
                progress = dict(stage="normal_image_evaluation", split=split, planned_position=position+1,
                                planned_images=len(entries), cumulative_candidates=len(rows), elapsed_s=time.monotonic()-started)
                dump(out / "PROGRESS.json", progress)
                print(json.dumps(progress), flush=True)
            del image, features, original, c0, coefficients, selection, gpu, bce, decoded
    for split in selected:
        actual = [row for row in rows if row["split"] == split]
        if len(actual) != expected[split]["candidates"] or len(em.image_groups(actual)) != expected[split]["effective_images"]:
            raise AssertionError("Incomplete fixed evaluation population")
    selected_dev_images = {int(entry["image_id"]) for entry in selected["dev"]}
    expected_history_keys = {key for key in prior if key[0] == "dev" and key[1] in selected_dev_images}
    if historical_keys != expected_history_keys:
        raise AssertionError("Historical/new dev candidate key sets differ")
    replay.assert_unchanged()
    if source_before != state_digest(replay.source):
        raise AssertionError("Original replay model changed")
    for arm, model in models.items():
        if model_before[arm] != state_digest(model) or model_buffers[arm] != buffer_digest(model):
            raise AssertionError(f"Frozen evaluation model changed: {arm}")
        if model.training or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError("Evaluation created gradients or enabled train mode")
    verify_inputs(inputs)
    budget()
    audit = dict(passed=True, source=replay.import_info, checkpoints=checkpoints, paired_training=paired_config,
                 historical_metric_max_abs_errors=max_history_error, historical_metric_atol=HISTORY_TOLERANCE,
                 historical_candidates_reproduced=len(historical_keys), original_coefficient_atol=3e-5,
                 original_coefficient_rtol=3e-5, original_coefficient_max_abs_error=max_original_coefficient_error,
                 all_source_parameters_and_buffers_unchanged=True, all_evaluation_models_unchanged=True,
                 all_inputs_unchanged=True, assignment_recomputed=False, empty_masks_removed=False,
                 no_positive_images_retained_in_manifest=True, input_sha256=inputs, assets=assets,
                 decode="Full original P; process_mask(upsample=True); true letterbox inverse; original COCO GT",
                 auc_fpr="Continuous input-grid logits within the frozen predicted box; padded original COCO mask labels")
    dump(out / "AUDIT.json", em.clean(audit))
    dump(out / "INPUT_MANIFEST.json", inputs)
    result = summarize(rows, out, expected, audit, args.smoke)
    torch.cuda.synchronize()
    dump(out / "COMPLETE.json", dict(completed=True, passed=True, kind="smoke_evaluation" if args.smoke else "normal_image_evaluation",
                                     smoke=args.smoke, epochs=3, arms=list(ARMS), candidates=len(rows), population=expected,
                                     elapsed_s=time.monotonic()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                     peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), input_sha256=inputs,
                                     scientific_status=result["assessment"]["status"], automatic_followup=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        dump(Path(args.out) / "FAILURE.json", dict(error_type=type(exc).__name__, error=str(exc),
                                                traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == "__main__":
    main()
