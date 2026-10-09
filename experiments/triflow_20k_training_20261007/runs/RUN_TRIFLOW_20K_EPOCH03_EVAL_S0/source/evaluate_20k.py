"""Frozen-official versus TriFlow native coefficient readout on full COCO val.

The learned module sees only frozen prediction inputs. GT is opened exclusively
for image identities, standard COCO AP, and a separate paired damage diagnostic.
An earlier official baseline is reused only after fresh per-image native forward
and native mask decode reproduce every saved identity and RLE exactly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from frozen_io import FrozenYOLO, OFFICIAL_SHA256, dump_json, predicted_neighbors, sha256, state_digest
from readout_support import coco_ap, gather_images, paired_metrics
from triflow_model import TriFlowConfig, TriFlowModel

EVALUATOR_VERSION = "triflow_20k_eval_v1"
SOURCE_FILES = ("evaluate_20k.py", "readout_support.py", "frozen_io.py", "triflow_model.py")
SOURCE_TRAIN_FILES = ("train_20k.py", "stream_data.py", "frozen_io.py", "triflow_model.py")
DECODER = "SegmentationValidator.save_json=True: process_mask_native; scale_preds.byte()"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def config_digest(configuration):
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_declared_subset(receipt, expected_original_ids=None):
    """Recompute the locked GT-independent selection from the declared universe."""
    universe = receipt.get("original_image_ids")
    selected = receipt.get("selected_image_ids")
    selector = receipt.get("subset_selector")
    if not isinstance(universe, list) or len(universe) != 118287 or universe != sorted(set(universe)) or not all(type(iid) is int for iid in universe):
        raise ValueError("20k selector requires the unique original118287 image-ID universe")
    if expected_original_ids is not None and universe != expected_original_ids:
        raise ValueError("20k selector universe differs from authenticated original COCO annotation IDs")
    expected = sorted(sorted(universe, key=lambda iid: (hashlib.sha256(("triflow_train20k_seed0:"+str(iid)).encode()).hexdigest(), iid))[:20000])
    identity = config_digest(expected)
    if selected != expected or receipt.get("image_identity_sha256") != identity or receipt.get("declared_subset_identity_sha256") != identity:
        raise ValueError("Actual20k IDs do not equal the locked hash-rank selection")
    if not isinstance(selector, dict) or selector.get("selector_name") != "sha256_triflow_train20k_seed0_v1" or selector.get("selection_seed") != 0 or selector.get("rank_prefix") != "triflow_train20k_seed0:" or selector.get("selection_size") != 20000:
        raise ValueError("20k selector algorithm/seed/prefix differ")
    if selector.get("uses_gt_categories_masks_or_image_contents") is not False or selector.get("universe_images") != 118287 or selector.get("selected_images") != 20000 or selector.get("universe_image_identity_sha256") != config_digest(universe) or selector.get("selected_image_identity_sha256") != identity:
        raise ValueError("20k selector source identity or GT-independence declaration differs")
    if receipt.get("subset_selector_sha256") != config_digest(selector):
        raise ValueError("20k selector metadata SHA differs")
    return identity


def load_triflow_checkpoint(head_path, device="cuda"):
    """Validate actual full-scale online coverage, never fabricate a cache proof."""
    import torch
    head_path = Path(head_path).resolve()
    payload = torch.load(head_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != "triflow_20k_module_final":
        raise ValueError("Expected triflow_20k_module_final checkpoint, not a pilot/cache checkpoint")
    if type(payload.get("epoch")) is not int or payload["epoch"] != 8 or payload.get("smoke_only") is not False:
        raise ValueError("Only fixed formal epoch8, smoke_only=false may enter final evaluation")
    if payload.get("audit_passed") is not True or payload.get("fixed_final_epoch") is not True:
        raise ValueError("Formal checkpoint requires passed actual training audit and fixed final epoch")
    if payload.get("complete_declared_train_subset") is not True or payload.get("complete_original_train_split") is not False:
        raise ValueError("Formal checkpoint must authenticate the declared20k subset, not claim original118287 coverage")
    if payload.get("base_weights_sha256") != OFFICIAL_SHA256:
        raise ValueError("TriFlow initialization source must match locked official checkpoint")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict) or config_digest(configuration) != payload.get("configuration_sha256"):
        raise ValueError("TriFlow configuration SHA mismatch")
    expected_config_keys = {f.name for f in fields(TriFlowConfig)} | {"feature_channels", "instance_hidden_channels"}
    if set(configuration) != expected_config_keys:
        raise ValueError("TriFlow configuration must declare exactly the method fields and real input channels")
    config = TriFlowConfig(**{f.name: configuration[f.name] for f in fields(TriFlowConfig)})
    if config != TriFlowConfig():
        raise ValueError("Changed configuration is outside this fixed feasibility screen")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_TRAIN_FILES):
        raise ValueError("Checkpoint must declare all training source hashes")
    for filename in SOURCE_TRAIN_FILES:
        archived = head_path.parent/"source"/filename
        if not archived.is_file() or sha256(archived) != sources[filename]:
            raise ValueError(f"Training source archive differs from checkpoint: {filename}")
    for filename in ("frozen_io.py", "triflow_model.py"):
        if sha256(Path(__file__).with_name(filename)) != sources[filename]:
            raise ValueError(f"Runtime method/extraction source differs from training: {filename}")
    audit_path = head_path.parent/"TRAINING_AUDIT.json"
    if not audit_path.is_file() or sha256(audit_path) != payload.get("training_audit_sha256"):
        raise ValueError("Checkpoint actual training audit bytes are missing or changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit.get("epochs") != 8:
        raise ValueError("Formal training audit has not passed epoch8")
    for name in ("all_applied_gradients_finite", "all_parameters_finite", "all_groups_really_changed",
                 "compiler_task_gradient_observed_nonzero", "compiler_task_phi_and_attention_same_probe_observed_nonzero",
                 "complete_declared_train_subset", "every_epoch_declared_images_complete", "every_epoch_actual_unique_images_complete"):
        if audit.get(name) is not True:
            raise ValueError(f"Training audit lacks actual required evidence: {name}")
    if audit.get("base_weights_sha256") != OFFICIAL_SHA256 or audit.get("sources") != sources:
        raise ValueError("Training audit source/checkpoint identities differ")
    if audit.get("configuration_sha256") != payload.get("configuration_sha256") or audit.get("smoke_only") is not False:
        raise ValueError("Training audit configuration/formal status differs from checkpoint")
    training_inputs = json.loads((head_path.parent/"TRAINING_INPUTS.json").read_text(encoding="utf-8"))
    if training_inputs.get("epochs") != 8 or training_inputs.get("sources") != sources or training_inputs.get("smoke_only") is not False:
        raise ValueError("Formal training input source identities differ")
    contract = payload.get("contract")
    if not isinstance(contract, dict) or config_digest(contract) != payload.get("contract_sha256") or contract != training_inputs.get("contract"):
        raise ValueError("Actual training contract bytes/contents differ from checkpoint")
    if contract.get("epochs") != 8 or contract.get("effective_images") != 20000 or contract.get("max_images") is not None or contract.get("smoke_only") is not False or contract.get("complete_declared_train_subset") is not True or contract.get("complete_original_train_split") is not False:
        raise ValueError("Formal20k contract requires eight complete declared20000-image epochs without an engineering max-images limit")
    locked_recipe = {"seed": 0, "instance_chunk": 4, "augmentation": False, "precision": "FP32, TF32 disabled, no AMP",
                     "optimizer": {"class": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "clip_grad_norm": 10}}
    if any(contract.get(name) != expected for name, expected in locked_recipe.items()) or contract.get("base_weights_sha256") != OFFICIAL_SHA256 or contract.get("configuration") != configuration:
        raise ValueError("Changed optimizer/precision/source configuration is outside the locked 20k recipe")
    if contract.get("sources") != sources or training_inputs.get("configuration_sha256") != payload.get("configuration_sha256") or audit.get("contract_sha256") != payload.get("contract_sha256"):
        raise ValueError("Training source/configuration/contract audit identities differ")
    stream_receipt_path = head_path.parent/"STREAM_RECEIPT.json"
    stream_inputs_path = head_path.parent/"STREAM_INPUTS.json"
    epochs_path = head_path.parent/"EPOCHS.json"
    for name, path in (("stream_receipt_sha256", stream_receipt_path), ("stream_inputs_sha256", stream_inputs_path), ("epochs_sha256", epochs_path)):
        if not path.is_file() or sha256(path) != payload.get(name) or audit.get(name) != payload.get(name):
            raise ValueError(f"Actual online training evidence bytes changed: {name}")
    stream_receipt = json.loads(stream_receipt_path.read_text(encoding="utf-8"))
    stream_inputs = json.loads(stream_inputs_path.read_text(encoding="utf-8"))
    epoch_rows = json.loads(epochs_path.read_text(encoding="utf-8")).get("epochs")
    expected_data = {name: stream_receipt.get(name) for name in ("annotation_sha256", "images_list_sha256", "image_identity_sha256", "declared_subset_identity_sha256", "subset_selector_sha256")}
    if expected_data != payload.get("data_hashes") or expected_data != training_inputs.get("data_hashes") or expected_data != contract.get("data_hashes"):
        raise ValueError("Checkpoint, online provider, training inputs and contract data identities differ")
    if expected_data["annotation_sha256"] != "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d":
        raise ValueError("Original train2017 annotation SHA differs")
    original_path = Path(stream_receipt["annotations"])
    if not original_path.is_file() or sha256(original_path) != expected_data["annotation_sha256"]:
        raise ValueError("Actual original COCO train annotation bytes are required for20k universe verification")
    original_json = json.loads(original_path.read_text(encoding="utf-8"))
    original_ids = sorted(int(image["id"]) for image in original_json["images"])
    del original_json
    for receipt in (stream_inputs, stream_receipt):
        if receipt.get("complete_declared_train_subset") is not True or receipt.get("complete_original_train_split") is not False or receipt.get("listed_images") != 20000 or receipt.get("original_split_images") != 118287 or receipt.get("engineering_subset") is not False:
            raise ValueError("Online provider did not authenticate the locked20k declared subset of original118287 train2017")
        if receipt.get("base_weights_sha256") != OFFICIAL_SHA256 or any(receipt.get(name) != digest for name, digest in expected_data.items()):
            raise ValueError("Online provider's official checkpoint/data identities differ")
        if receipt.get("sources") != {name: sources[name] for name in ("stream_data.py", "frozen_io.py", "triflow_model.py")}:
            raise ValueError("Actual online provider sources differ from trained checkpoint")
        validate_declared_subset(receipt, original_ids)
    if stream_receipt.get("observed_unique_images") != 20000 or stream_receipt.get("all_listed_images_observed") is not True or stream_receipt.get("all_native_replay_exact") is not True:
        raise ValueError("Provider did not actually observe every declared training image")
    dimensions = stream_receipt.get("dimensions", {})
    if configuration.get("feature_channels") != dimensions.get("feature_channels") or configuration.get("instance_hidden_channels") != dimensions.get("instance_hidden_channels") or training_inputs.get("configuration") != configuration:
        raise ValueError("Actual online feature/hidden dimensions or input configuration differ from trained module")
    frozen = stream_receipt.get("frozen_integrity", {})
    if frozen.get("passed") is not True or frozen.get("all_state_exact") is not True or frozen.get("initial_state_sha256") != frozen.get("final_state_sha256"):
        raise ValueError("Actual online frozen model integrity has not passed")
    if not isinstance(epoch_rows, list) or len(epoch_rows) != 8 or [row.get("epoch") for row in epoch_rows] != list(range(1, 9)):
        raise ValueError("Actual epoch history does not contain exactly epochs one through eight")
    for row in epoch_rows:
        if row.get("images") != 20000 or row.get("visited_unique_images") != 20000 or row.get("visited_image_identity_sha256") != expected_data["image_identity_sha256"] or row.get("actual_completed_image_ids_match_declared") is not True:
            raise ValueError("Actual per-epoch completed-image identities do not cover the complete original split")
    if sum(row.get("applied_steps", -1) for row in epoch_rows) != audit.get("optimizer_applied") or audit.get("completed_epochs") != 8 or audit.get("images_per_epoch") != 20000:
        raise ValueError("Actual epoch history and applied-update audit disagree")
    completion = json.loads((head_path.parent/"TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    if completion.get("head_sha256") != sha256(head_path) or completion.get("head_bytes") != head_path.stat().st_size or completion.get("audit_passed") is not True or completion.get("smoke_only") is not False:
        raise ValueError("Actual final-checkpoint completion bytes/status differ")
    task_proof_path = head_path.parent/"TASK_GRADIENT_EVIDENCE.json"
    if sha256(task_proof_path) != payload.get("task_gradient_evidence_sha256") or audit.get("task_gradient_evidence_sha256") != payload.get("task_gradient_evidence_sha256"):
        raise ValueError("Actual task-gradient evidence bytes changed")
    task_proof = json.loads(task_proof_path.read_text(encoding="utf-8"))
    def joint_phi_attention_evidence(row):
        phi = row.get("phi_task_gradient_norm")
        attention = row.get("cross_attention_task_gradient_norm")
        return (isinstance(phi, (int, float)) and not isinstance(phi, bool) and math.isfinite(phi) and phi > 0
                and isinstance(attention, (int, float)) and not isinstance(attention, bool) and math.isfinite(attention) and attention > 0
                and row.get("task_gradient_phi_attention_same_probe_nonzero") is True
                and row.get("all_task_gradients_finite") is True)
    actual_joint_probes = sum(joint_phi_attention_evidence(row) for row in task_proof.get("evidence", []))
    probe_rows = task_proof.get("evidence")
    if not isinstance(probe_rows, list) or len(probe_rows) != 8 or [row.get("epoch") for row in probe_rows] != list(range(1, 9)) or task_proof.get("no_hidden_warmup") is not True:
        raise ValueError("Actual task-gradient probe history must cover eight epochs without hidden warmup")
    if task_proof.get("any_observed_phi_attention_same_probe_nonzero") is not True or actual_joint_probes < 1:
        raise ValueError("Actual same-probe finite nonzero phi and cross-attention task gradients are required")
    if audit.get("compiler_task_phi_attention_same_probe_count") != actual_joint_probes:
        raise ValueError("Training audit joint phi/attention probe count differs from actual evidence")
    if audit.get("optimizer_attempts") != audit.get("optimizer_applied") or not isinstance(audit.get("optimizer_applied"), int) or audit["optimizer_applied"] < 1 or audit.get("all_optimizer_states_finite") is not True:
        raise ValueError("Actual optimizer updates/states have not passed the 20k audit")
    module = TriFlowModel(configuration["feature_channels"], configuration["instance_hidden_channels"], config).float()
    state = payload.get("state_dict")
    expected = module.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("TriFlow learned module state has missing/extra keys")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"Invalid TriFlow state tensor: {name}")
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"TriFlow state requires finite FP32: {name}")
    module.load_state_dict(state, strict=True)
    if state_digest(module) != audit.get("final_head_state_sha256"):
        raise ValueError("Loaded final module state differs from audited trained bytes")
    module = module.to(device).eval()
    module.requires_grad_(False)
    record = {"kind": payload["kind"], "epoch": payload["epoch"], "smoke_only": False,
              "audit_passed": True, "base_weights_sha256": OFFICIAL_SHA256,
              "configuration": configuration, "configuration_sha256": payload["configuration_sha256"],
              "head_path": str(head_path), "head_sha256": sha256(head_path), "head_bytes": head_path.stat().st_size,
              "training_audit_sha256": payload["training_audit_sha256"], "training_sources": sources,
              "task_gradient_evidence_sha256": sha256(task_proof_path),
              "phi_attention_same_probe_count": actual_joint_probes,
              "stream_receipt_sha256": payload["stream_receipt_sha256"], "stream_inputs_sha256": payload["stream_inputs_sha256"], "epochs_sha256": payload["epochs_sha256"],
              "stream_receipt_path": str(stream_receipt_path), "stream_inputs_path": str(stream_inputs_path), "epochs_path": str(epochs_path),
              "data_hashes": payload["data_hashes"], "contract": contract, "contract_sha256": payload["contract_sha256"],
              "complete_original_train_split": False, "complete_declared_train_subset": True, "declared_train_images": 20000, "all_epochs_declared_subset_coverage_verified": True,
              "subset_selector": stream_receipt["subset_selector"], "subset_selector_sha256": stream_receipt["subset_selector_sha256"], "declared_subset_identity_sha256": stream_receipt["declared_subset_identity_sha256"], "engineering_subset": False,
              "training_run_id": head_path.parent.name, "parent": payload.get("parent"),
              "loaded_state_sha256": audit["final_head_state_sha256"], "loaded_state_keys": sorted(state),
              "all_state_finite_fp32": True, "no_gt_forward": True}
    return module, record


def decode(extracted, coefficients, validator, device):
    """The exact official native validator route, including its byte scaling."""
    import torch
    native = torch.cat((extracted["boxes"], extracted["scores"][:, None],
                        extracted["classes"].float()[:, None], coefficients), 1)[None].to(device)
    proto = extracted["P"][None].to(device)
    predictions = validator.postprocess(((native, proto), None))[0]
    pbatch = {"imgsz": (640, 640), "ori_shape": tuple(extracted["original_shape"]),
              "ratio_pad": (extracted["letterbox"]["ratio"],
                            (extracted["letterbox"]["left"], extracted["letterbox"]["top"])),
              "im_file": extracted["image_path"]}
    raw_boxes = predictions["bboxes"].detach().cpu().numpy().copy()
    raw_conf = predictions["conf"].detach().cpu().numpy().copy()
    classes = predictions["cls"].detach().cpu().numpy().astype(int)
    scaled = validator.scale_preds(predictions, pbatch)
    return raw_boxes, raw_conf, classes, scaled["bboxes"].cpu().numpy(), scaled["masks"].cpu().numpy()


def detection_records(extracted, iid, coefficients, validator, device, categories, mask_utils):
    import numpy as np
    raw_boxes, scores, classes, boxes, masks = decode(extracted, coefficients, validator, device)
    rows = []
    for index, (raw_box, score, cls, box, mask) in enumerate(zip(raw_boxes, scores, classes, boxes, masks)):
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        xywh = [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])]
        rows.append({"image_id": int(iid), "category_id": int(categories[cls]),
                     "bbox": [round(x, 3) for x in xywh], "score": round(float(score), 5),
                     "segmentation": rle, "detection_index": index,
                     "raw_input_box_xyxy": raw_box.tolist(), "raw_confidence": float(score),
                     "box_xyxy": box.tolist(), "model_class": int(cls)})
    return rows


def refine_coefficients(extracted, module, device):
    """All deployment choices use only native predictions, never GT matches."""
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    began = time.monotonic()
    graph_rows, graph_valid = predicted_neighbors(extracted["boxes"].numpy(), extracted["raw_indices"].numpy(),
                                                extracted["scores"].numpy(), k=2)
    eligible = (extracted["scores"] > .001).nonzero().flatten().tolist()
    selected_rows = eligible[:64]
    height, width = extracted["P"].shape[-2:]
    executable_rows, unsupported = [], []
    for row in selected_rows:
        box = extracted["boxes_proto"][row]
        x_inside = (torch.arange(width) >= box[0]) & (torch.arange(width) < box[2])
        y_inside = (torch.arange(height) >= box[1]) & (torch.arange(height) < box[3])
        if bool(x_inside.any() and y_inside.any()):
            executable_rows.append(row)
        else:
            unsupported.append({"native_output_row": row, "post_conf_detection_index": eligible.index(row),
                                "raw_index": int(extracted["raw_indices"][row]),
                                "reason": "empty_native_prototype_roi", "kept_c0_exact": True,
                                "box_proto": box.tolist()})
    refined = extracted["c0"].clone()
    p, f = extracted["P"].to(device), extracted["F"].to(device)
    all_diagnostics, mappings = [], []
    for start in range(0, len(executable_rows), 4):
        rows = executable_rows[start:start+4]
        indices = torch.tensor(rows, dtype=torch.long)
        neighbor_indices = torch.from_numpy(graph_rows[rows])
        valid = torch.from_numpy(graph_valid[rows]).to(device)
        neighbor_c = extracted["c0"][neighbor_indices.clamp_min(0)].to(device)
        neighbor_logits = torch.einsum("chw,nkc->nkhw", p, neighbor_c).masked_fill(~valid[:, :, None, None], 0)
        outputs = module(p, extracted["c0"][indices].to(device), f, extracted["boxes_proto"][indices].to(device),
                         neighbor_logits, valid, extracted["h"][indices].to(device))
        if not torch.isfinite(outputs["coefficients_refined"]).all():
            raise FloatingPointError("Nonfinite deployment coefficient readout")
        refined[indices] = outputs["coefficients_refined"].cpu()
        diagnostics = {key: value.detach().cpu().tolist() for key, value in outputs["diagnostics"].items()}
        all_diagnostics.append({"native_output_rows": rows, "diagnostics": diagnostics})
        for local, row in enumerate(rows):
            mappings.append({"native_output_row": row, "post_conf_detection_index": eligible.index(row),
                             "raw_index": int(extracted["raw_indices"][row]),
                             "predicted_neighbor_output_rows": graph_rows[row].tolist(),
                             "predicted_neighbor_raw_indices": [int(extracted["raw_indices"][j]) if j >= 0 else None for j in graph_rows[row]],
                             "neighbor_valid": graph_valid[row].tolist(),
                             "coefficient_max_abs_change": float((refined[row]-extracted["c0"][row]).abs().max())})
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    unchanged = set(range(len(refined))) - set(selected_rows)
    if unchanged and not torch.equal(refined[sorted(unchanged)], extracted["c0"][sorted(unchanged)]):
        raise RuntimeError("Unselected native coefficients changed")
    unsupported_rows = [item["native_output_row"] for item in unsupported]
    if unsupported_rows and not torch.equal(refined[unsupported_rows], extracted["c0"][unsupported_rows]):
        raise RuntimeError("Unsupported-ROI native coefficients must retain c0 exactly")
    return refined, {"selected_rows": selected_rows, "eligible_output_rows": eligible, "selected_count": len(selected_rows),
                     "executable_rows": executable_rows, "refined_count": len(executable_rows),
                     "unsupported_roi_count": len(unsupported), "unsupported_roi": unsupported,
                     "unselected_coefficients_exact": True, "mappings": mappings, "chunks": all_diagnostics,
                     "module_seconds": time.monotonic()-began, "gt_used_in_forward": False}


def validate_baseline_cache(cache, configuration, ids):
    receipt_file = cache/"COMPLETE.json"
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if receipt.get("status") != "prediction_complete" or receipt.get("image_count") != 5000:
        raise ValueError("Original official baseline must be a complete full-val cache")
    original = receipt.get("fingerprint", {})
    weights = original.get("weights", {}).get("checkpoint", {})
    if weights.get("sha256") != OFFICIAL_SHA256:
        raise ValueError("Cached official reference checkpoint differs")
    if original.get("image_ids") != ids or original.get("annotations", {}).get("sha256") != configuration["annotations"]["sha256"]:
        raise ValueError("Cached official reference annotation/image identities differ")
    if original.get("images_list", {}).get("sha256") != configuration["images_list"]["sha256"]:
        raise ValueError("Cached baseline exact image list bytes differ")
    if original.get("vendor_source_sha256") != configuration["vendor_source_sha256"]:
        raise ValueError("Cached baseline official decoder/source bytes differ")
    original_config = original.get("config", {})
    for key, value in configuration["config"].items():
        if key != "device" and original_config.get(key) != value:
            raise ValueError(f"Cached baseline inference setting differs: {key}")
    if sha256(cache/"predictions.json") != receipt.get("predictions_sha256"):
        raise ValueError("Cached original official predictions changed")
    for name in ("COCO_METRICS.json", "COCO_segm_ACCUMULATED.npz", "COCO_bbox_ACCUMULATED.npz", "COCO_segm.txt", "COCO_bbox.txt"):
        if not (cache/name).is_file():
            raise FileNotFoundError(cache/name)
    metrics = json.loads((cache/"COCO_METRICS.json").read_text(encoding="utf-8"))
    if metrics.get("predictions_sha256") != receipt["predictions_sha256"] or metrics.get("image_ids") != sorted(ids):
        raise ValueError("Original baseline COCO metrics identity differs")
    return receipt


def assemble_predictions(arm, ids):
    temporary = arm/"predictions.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first = True
        for iid in ids:
            records = json.loads((arm/"images"/f"{iid:012d}.json").read_text(encoding="utf-8"))["detections"]
            for detection in records:
                standard = {k: detection[k] for k in ("image_id", "category_id", "bbox", "score", "segmentation")}
                if not first:
                    handle.write(",\n")
                handle.write(json.dumps(standard, separators=(",", ":"), allow_nan=False))
                first = False
        handle.write("\n]\n")
    temporary.replace(arm/"predictions.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "head", "images-list", "annotations", "weights", "vendor", "baseline-cache"):
        parser.add_argument("--"+name, required=name == "head")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-checkpoint-only", action="store_true", help="Strict engineering checkpoint contract; no image inference or AP")
    parser.add_argument("--verify-output", type=Path, help="Optional NEW checkpoint-contract JSON; existing output is refused")
    args = parser.parse_args()
    if args.verify_checkpoint_only:
        _, metadata = load_triflow_checkpoint(args.head, args.device)
        receipt = {"status": "passed", "passed": True, "checkpoint_contract_only": True, "inference_executed": False, "ap_measured": False, "head": metadata}
        if args.verify_output is not None:
            with args.verify_output.resolve().open("x", encoding="utf-8") as handle:
                json.dump(receipt, handle, indent=2, allow_nan=False)
        print(json.dumps(receipt, allow_nan=False))
        return 0
    for name in ("root", "run_id", "images_list", "annotations", "weights", "vendor", "baseline_cache"):
        if getattr(args, name) is None:
            parser.error("normal evaluation requires --"+name.replace("_", "-"))
    root = Path(args.root).resolve()
    run = root/"runs"/args.run_id
    if (run/"SUMMARY.json").exists() or (run/"BASELINE_REPLAY_IMAGES.jsonl").exists():
        raise FileExistsError("Evaluation history is immutable; retry needs another Run ID")
    run.mkdir(parents=True, exist_ok=True)
    source = run/"source"
    source.mkdir(exist_ok=True)
    for name in SOURCE_FILES:
        current = Path(__file__).with_name(name)
        target = source/name
        if target.exists() and sha256(target) != sha256(current):
            raise ValueError("Evaluation source snapshot differs from executing source")
        if not target.exists():
            shutil.copy2(current, target)
    started, began = now(), time.monotonic()
    extractor = None
    try:
        import numpy as np
        import torch
        extractor = FrozenYOLO(args.weights, args.vendor, args.device)
        import cv2
        import ultralytics
        from pycocotools import mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from ultralytics.data import converter
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        from ultralytics.utils import ops
        cv2.setNumThreads(1)
        module, head_metadata = load_triflow_checkpoint(args.head, args.device)
        initial_module_digest = state_digest(module)
        coco = COCO(str(Path(args.annotations).resolve()))
        paths, ids = gather_images(Path(args.images_list).resolve(), coco)
        if len(ids) != 5000 or sorted(ids) != sorted(coco.imgs):
            raise ValueError("Formal evaluation requires all 5000 original COCO val2017 images")
        vendor = Path(args.vendor).resolve()
        source_paths = {"segment_validator": "ultralytics/models/yolo/segment/val.py", "ops": "ultralytics/utils/ops.py",
                        "nms": "ultralytics/utils/nms.py", "letterbox": "ultralytics/data/augment.py"}
        configuration = {"evaluator_version": EVALUATOR_VERSION, "evaluator_sha256": sha256(Path(__file__)),
                         "ultralytics": ultralytics.__version__, "torch": torch.__version__,
                         "vendor_source_sha256": {k: sha256(vendor/v) for k, v in source_paths.items()},
                         "method_source_sha256": {k: sha256(source/k) for k in SOURCE_FILES},
                         "annotations": {"path": str(Path(args.annotations).resolve()), "sha256": sha256(args.annotations)},
                         "images_list": {"path": str(Path(args.images_list).resolve()), "sha256": sha256(args.images_list)},
                         "image_ids": ids, "config": {"imgsz": 640, "shape": [640, 640], "batch": 1, "half": False,
                         "conf": .001, "max_det": 300, "coco_maxDets": [1, 10, 100], "rect": False, "scaleup": False,
                         "augment": False, "native_one2one": True, "decode": DECODER, "device": args.device, "tf32": False},
                         "method": {"input": "actual frozen P3 visual/prototypes/genuine final-conv hidden/native c0/boxes",
                                    "head_configuration_sha256": head_metadata["configuration_sha256"],
                                    "eligible_confidence": .001, "first_post_conf_native_rows": 64, "instance_chunk": 4,
                                    "predicted_neighbors": "top2 positive predicted-box IoU; stable output row tie; same raw id excluded",
                                    "gt_used_in_forward": False, "learned_coefficient_residual": False,
                                    "unsupported_native_roi": "first64 budget retained; unsupported rows keep c0 exactly, no replacement row",
                                    "compiler": "continuous ownership field -> outward boundary flow and active hinges -> differentiable weighted SPD solve"}}
        baseline_cache = Path(args.baseline_cache).resolve()
        original_baseline = validate_baseline_cache(baseline_cache, configuration, ids)
        dump_json(run/"EVALUATION_INPUTS.json", {"configuration": configuration, "head": head_metadata,
                                               "baseline_cache": str(baseline_cache), "original_baseline_receipt_sha256": sha256(baseline_cache/"COMPLETE.json"),
                                               "started_at": started, "interpreter": sys.executable})
        provenance = run/"head_provenance"
        provenance.mkdir(exist_ok=True)
        for filename in SOURCE_TRAIN_FILES:
            shutil.copy2(Path(args.head).resolve().parent/"source"/filename, provenance/filename)
        for filename in ("TRAINING_AUDIT.json", "TRAINING_INPUTS.json", "TASK_GRADIENT_EVIDENCE.json", "TRAINING_COMPLETE.json", "STREAM_INPUTS.json", "STREAM_RECEIPT.json", "EPOCHS.json"):
            shutil.copy2(Path(args.head).resolve().parent/filename, provenance/filename)
        dump_json(run/"HEAD_PROVENANCE.json", head_metadata)
        shutil.copy2(baseline_cache/"COMPLETE.json", run/"ORIGINAL_BASELINE_RECEIPT.json")
        validator = SegmentationValidator(args={"conf": .001, "max_det": 300, "save_json": True,
                                               "plots": False, "imgsz": 640, "half": False, "rect": False})
        validator.nc, validator.end2end, validator.process = 80, True, ops.process_mask_native
        categories = converter.coco80_to_coco91_class()
        for name in ("baseline", "triflow"):
            (run/name/"images").mkdir(parents=True, exist_ok=True)
        replay_count, replay_detections, modified_instances, refined_instances = 0, 0, 0, 0
        total_module_seconds, total_forward_seconds = 0., 0.
        compiler_counts = {"boundary_valid_count": 0, "root_count": 0, "neighbor_transition_count": 0,
                           "active_anchor_count": 0, "trust_saturated_instances": 0, "invalid_boundary_root_count": 0}
        unsupported_roi_instances = 0
        baseline_copy_hashes = []
        with torch.inference_mode():
            for position, (path, iid) in enumerate(zip(paths, ids), 1):
                forward_start = time.monotonic()
                extracted = extractor.extract(path)
                total_forward_seconds += time.monotonic()-forward_start
                if extracted["F"].shape[0] != head_metadata["configuration"]["feature_channels"] or extracted["h"].shape[-1] != head_metadata["configuration"]["instance_hidden_channels"]:
                    raise ValueError("Real native feature/hidden dimensions differ from trained module")
                original_image = baseline_cache/"images"/f"{iid:012d}.json"
                cached_image = json.loads(original_image.read_text(encoding="utf-8"))
                fresh_baseline = detection_records(extracted, iid, extracted["c0"], validator, args.device, categories, mask_utils)
                if cached_image.get("image_id") != iid or cached_image.get("detections") != fresh_baseline:
                    raise RuntimeError(f"Original baseline replay mismatch at image {iid}; refuse mixed evaluation definitions")
                shutil.copy2(original_image, run/"baseline"/"images"/original_image.name)
                image_sha = sha256(original_image)
                baseline_copy_hashes.append(image_sha)
                replay_count += 1
                replay_detections += len(fresh_baseline)
                with (run/"BASELINE_REPLAY_IMAGES.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"image_id": iid, "native_replay_exact": True, "baseline_rows_rle_exact": True,
                                             "detection_count": len(fresh_baseline), "source_image_cache_sha256": image_sha,
                                             "current_input_sha256": extracted["input_sha256"], "image_sha256": extracted["image_sha256"]}, allow_nan=False)+"\n")
                refined, diagnostic = refine_coefficients(extracted, module, args.device)
                method_rows = detection_records(extracted, iid, refined, validator, args.device, categories, mask_utils)
                identity_keys = ("image_id", "category_id", "bbox", "score", "detection_index", "raw_input_box_xyxy",
                                 "raw_confidence", "box_xyxy", "model_class")
                if len(method_rows) != len(fresh_baseline) or any(any(a[k] != b[k] for k in identity_keys) for a, b in zip(fresh_baseline, method_rows)):
                    raise RuntimeError(f"TriFlow changed native box/class/confidence/order at image {iid}")
                dump_json(run/"triflow"/"images"/f"{iid:012d}.json", {"image_id": iid, "image_path": str(path),
                          "original_shape": extracted["original_shape"], "letterbox": extracted["letterbox"], "detections": method_rows})
                diagnostic.update(image_id=iid, native_rows=len(extracted["c0"]), frozen_native_replay_exact=True,
                                  head_sha256=head_metadata["head_sha256"], box_class_confidence_order_exact=True)
                with (run/"COMPILER_IMAGES.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(diagnostic, allow_nan=False)+"\n")
                modified_instances += diagnostic["selected_count"]
                refined_instances += diagnostic["refined_count"]
                unsupported_roi_instances += diagnostic["unsupported_roi_count"]
                total_module_seconds += diagnostic["module_seconds"]
                for chunk in diagnostic["chunks"]:
                    d = chunk["diagnostics"]
                    for key in ("boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count"):
                        compiler_counts[key] += int(sum(d[key]))
                    compiler_counts["trust_saturated_instances"] += sum(bool(x) for x in d["trust_saturated"])
                if position == 1 or position % 25 == 0 or position == len(ids):
                    print(f"PREDICT triflow {position}/{len(ids)} baseline_replay_exact={replay_count} selected={modified_instances} module_seconds={total_module_seconds:.1f} elapsed_s={time.monotonic()-began:.1f}", flush=True)
        integrity = extractor.verify_frozen()
        final_module_digest = state_digest(module)
        if final_module_digest != initial_module_digest:
            raise RuntimeError("Learned module state changed during frozen evaluation")
        compiler_counts["invalid_boundary_root_count"] = compiler_counts["boundary_valid_count"]-compiler_counts["root_count"]
        replay = {"status": "passed", "all_5000_images_replayed": replay_count == 5000,
                  "images": replay_count, "detections": replay_detections,
                  "all_native_forward_exact": True, "all_baseline_identity_and_rle_exact": True,
                  "original_baseline_cache": str(baseline_cache), "original_baseline_receipt_sha256": sha256(baseline_cache/"COMPLETE.json"),
                  "original_frozen_state_sha256": original_baseline["frozen_state_sha256"],
                  "original_digest_scope": "legacy digest excluded coefficient/prototype-tail non-BN parameters; not equal to new full-state digest",
                  "current_frozen_model_integrity": integrity,
                  "baseline_image_cache_sha256_chain": hashlib.sha256("".join(baseline_copy_hashes).encode()).hexdigest(),
                  "reuse_basis": "fresh actual native forward and native decode, per-image detection identity and canonical RLE exact",
                  "no_original_experiment_mutation": True}
        dump_json(run/"BASELINE_REPLAY_VERIFICATION.json", replay)
        for filename in ("predictions.json", "COCO_METRICS.json", "COCO_segm_ACCUMULATED.npz", "COCO_bbox_ACCUMULATED.npz", "COCO_segm.txt", "COCO_bbox.txt"):
            shutil.copy2(baseline_cache/filename, run/"baseline"/filename)
        assemble_predictions(run/"triflow", ids)
        receipts = {}
        for name in ("baseline", "triflow"):
            receipt = {"status": "prediction_complete", "image_count": 5000,
                       "fingerprint": configuration | {"weights": {"checkpoint": {"path": str(Path(args.weights).resolve()),
                       "sha256": OFFICIAL_SHA256, "bytes": Path(args.weights).stat().st_size}}},
                       "frozen_state_sha256": integrity["final_state_sha256"], "frozen_digest_scope": "all actual frozen official model state_dict tensors",
                       "frozen_state_provenance": "one shared actual frozen official model; fresh both-arm mask decode and full-state before/after hash",
                       "frozen_integrity": integrity, "readout_view": "postprocessed native one2one detections after top-k/conf; not full raw candidates",
                       "predictions_sha256": sha256(run/name/"predictions.json"),
                       "model": {"source_checkpoint": str(Path(args.weights).resolve()), "official_base_sha256": OFFICIAL_SHA256,
                                 "learned_module_loaded": name == "triflow", "official_weights_unchanged": True,
                                 "no_gt_forward": True, "head": head_metadata if name == "triflow" else None},
                       "baseline_replay_receipt_sha256": sha256(run/"BASELINE_REPLAY_VERIFICATION.json")}
            if name == "baseline":
                receipt["original_cache_reuse"] = replay
                receipt["source_original_predictions_sha256"] = original_baseline["predictions_sha256"]
            receipts[name] = receipt
            dump_json(run/name/"COMPLETE.json", receipt)
        metrics = {name: coco_ap(name, coco, ids, run, COCO, COCOeval) for name in ("baseline", "triflow")}
        pairs = paired_metrics("baseline", ["baseline", "triflow"], coco, ids, run, receipts, np, mask_utils,
                               SimpleNamespace(bootstrap=5000, bootstrap_seed=0))
        deltas = {"triflow": {kind: {key+"_points": 100*(metrics["triflow"][kind][key]-metrics["baseline"][kind][key])
                                               if metrics["triflow"][kind][key] is not None and metrics["baseline"][kind][key] is not None else None
                                        for key in ("AP", "AP75", "APsmall")} for kind in ("segm", "bbox")}}
        summary = {"status": "complete", "baseline": "baseline", "image_count": 5000, "covers_all_annotation_images": True,
                   "evaluation_scope": "all images in supplied original COCO val2017 annotations", "configuration": configuration,
                   "prediction_receipts": receipts, "metrics": metrics, "delta_vs_baseline": deltas, "paired": pairs,
                   "baseline_replay": replay, "head_provenance": head_metadata, "compiler_counts": compiler_counts,
                   "timing": {"actual_frozen_forward_seconds": total_forward_seconds, "extra_module_seconds": total_module_seconds,
                              "extra_module_ms_per_image": 1000*total_module_seconds/5000,
                              "total_elapsed_seconds": time.monotonic()-began, "includes_python_and_adapter_transfer": True},
                   "selected_instances": modified_instances, "started_at": started, "completed_at": now(),
                   "refined_instances": refined_instances, "unsupported_roi_instances": unsupported_roi_instances,
                   "learned_module_state_exact": initial_module_digest == final_module_digest,
                   "limitations": ["Single seed, fixed8 epochs on declared hash-ranked20000 train2017 images; not original118287 full training, not a universal sufficiency guarantee; val2017 was previously observed",
                                   "Only first64 post-conf native rows are refined; full raw-candidate five-state metrics are unmeasured",
                                   "Explanatory same-class GT damage pairing permits duplicate detection-to-GT matches and differs from COCO AP matching",
                                   "Timing includes adapter transfers and Python; it is not deployment throughput profiling"]}
        dump_json(run/"SUMMARY.json", summary)
        dump_json(run/"EVALUATION_COMPLETE.json", {"status": "completed", "summary_sha256": sha256(run/"SUMMARY.json"),
                                                   "images": 5000, "completed_at": now()})
        print(f"EVALUATION_COMPLETE {run/'SUMMARY.json'}", flush=True)
    except Exception as exc:
        dump_json(run/"EVALUATION_FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(), "time": now()})
        raise
    finally:
        if extractor is not None:
            extractor.close()


if __name__ == "__main__":
    main()
