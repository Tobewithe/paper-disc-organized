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
from diagnostics_contract_r3 import archive_contract, verify_contract, import_runtime, REQUIRED_DIAGNOSTICS

EVALUATOR_VERSION = "triflow_20k_epoch_eval_r3_v1"
SOURCE_FILES = ("evaluate_epoch_r3.py", "readout_support.py", "frozen_io.py", "triflow_model.py", "diagnostics_runtime.py", "diagnostics_contract_r3.py")
SOURCE_TRAIN_FILES = ("train_20k.py", "stream_data.py", "frozen_io.py", "triflow_model.py")
ORIGINAL_TRAIN_SOURCES = {
    "train_20k.py": "944dcb27d75f1d1d9ce8cc950c206c607b73ba6023a7a5f19517134bbd2a86bb",
    "stream_data.py": "6460a9277b47a8ca790e03213afbaa5d628592b9319608e0be70dd4799091fb6",
    "frozen_io.py": "cff75eec547d9ec12eda5409bd273231ab99c16cdc6119d75f6d413098363293",
    "triflow_model.py": "1b96e0767ab1985875cdfec33cdd6e98a666ecd3ee5caefc7dbca9e80c35691e",
}
GRADIENT_GROUPS = {"token_projection", "ownership_embeddings", "ownership_projection", "cross_attention", "interaction_norm", "field_head"}
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


def original_image_ids_only(path):
    """Decode only the authenticated JSON images array; do not retain GT annotations.

    The entire file SHA is checked by the caller. The standard COCO top-level
    images array is decoded incrementally, stopping before annotation payloads.
    """
    import re
    decoder = json.JSONDecoder()
    buffer, start = "", None
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                raise ValueError("Authenticated COCO JSON has no complete images array")
            buffer += chunk
            if start is None:
                match = re.search(r'"images"\s*:\s*\[', buffer)
                if match is None:
                    continue
                start = match.end() - 1
            try:
                images, _ = decoder.raw_decode(buffer, start)
            except json.JSONDecodeError:
                continue
            if not isinstance(images, list):
                raise ValueError("COCO images field is not an array")
            ids = sorted(int(image["id"]) for image in images)
            del images, buffer
            return ids


def load_epoch_snapshot(head_path, device="cuda", expected_epoch=None, training_run=None, diagnostics_execution=None):
    """Load an actual formal epoch boundary, without pretending final audit."""
    import re
    import torch
    head_path = Path(head_path).resolve()
    snapshot = torch.load(head_path, map_location="cpu", weights_only=False)
    if not isinstance(snapshot, dict) or snapshot.get("kind") != "triflow_20k_training_snapshot" or snapshot.get("trainer_version") != "triflow_20k_online_v1" or snapshot.get("reason") != "epoch_boundary" or snapshot.get("resumable") is not True:
        raise ValueError("Only an actual resumable20k epoch_boundary training snapshot may enter this evaluator")
    contract, state, provider = snapshot.get("contract"), snapshot.get("state"), snapshot.get("provider_state")
    if not all(isinstance(value, dict) for value in (contract, state, provider)):
        raise ValueError("Snapshot lacks actual contract/state/provider state")
    if config_digest(contract) != snapshot.get("contract_sha256"):
        raise ValueError("Snapshot contract content SHA differs")
    if contract.get("trainer_version") != "triflow_20k_online_v1" or contract.get("epochs") != 8 or contract.get("declared_train_images") != 20000 or contract.get("smoke_only") is not False or contract.get("max_images") is not None or contract.get("effective_images") != 20000 or contract.get("complete_declared_train_subset") is not True or contract.get("complete_original_train_split") is not False or contract.get("engineering_subset") is not False:
        raise ValueError("Epoch evaluation requires the locked formal20000/8epoch contract; engineering truncation is not formal evidence")
    locked = {"seed": 0, "instance_chunk": 4, "augmentation": False, "precision": "FP32, TF32 disabled, no AMP",
              "optimizer": {"class": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "clip_grad_norm": 10}, "base_weights_sha256": OFFICIAL_SHA256}
    if any(contract.get(key) != value for key, value in locked.items()):
        raise ValueError("Snapshot changed locked precision/optimizer/frozen-official recipe")
    epochs = state.get("epochs")
    if not isinstance(epochs, list) or not 1 <= len(epochs) <= 8:
        raise ValueError("Snapshot has no valid actual completed epoch prefix")
    epoch = len(epochs)
    if expected_epoch is not None and expected_epoch != epoch:
        raise ValueError("Actual completed snapshot epoch differs from --expected-epoch")
    cursor = {"epoch": epoch + 1, "image_position": 0, "chunk_start": 0}
    if state.get("cursor") != cursor or [row.get("epoch") for row in epochs] != list(range(1, epoch + 1)):
        raise ValueError("Snapshot is not a complete consecutive epoch boundary with next-epoch cursor")
    run_id = snapshot.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"RUN_TRIFLOW_20K_[A-Z0-9_]+", run_id) is None:
        raise ValueError("Snapshot source training Run ID is invalid")
    if training_run is None:
        experiment = next((ancestor for ancestor in head_path.parents if ancestor.name == "triflow_20k_training_20261007"), None)
        if experiment is None:
            raise ValueError("Use --training-run to locate actual snapshot source evidence")
        training_run = experiment / "runs" / run_id
    origin = Path(training_run).resolve()
    if origin.name != run_id:
        raise ValueError("Actual training evidence directory differs from snapshot Run ID")
    stream_path, inputs_path = origin / "STREAM_INPUTS.json", origin / "TRAINING_INPUTS.json"
    stream = json.loads(stream_path.read_text(encoding="utf-8"))
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    if inputs.get("contract") != contract or inputs.get("contract_sha256") != snapshot["contract_sha256"]:
        raise ValueError("Actual training inputs differ from captured snapshot contract")
    sources = contract.get("sources")
    if sources != ORIGINAL_TRAIN_SOURCES or inputs.get("sources") != sources:
        raise ValueError("Snapshot lacks the exact four original training source hashes")
    for filename, digest in sources.items():
        if sha256(origin / "source" / filename) != digest:
            raise ValueError("Actual archived training source differs: " + filename)
    for filename in ("frozen_io.py", "triflow_model.py"):
        if sha256(Path(__file__).with_name(filename)) != sources[filename]:
            raise ValueError("Epoch runtime changed original method/extraction source: " + filename)
    data = contract.get("data_hashes")
    names = ("annotation_sha256", "images_list_sha256", "image_identity_sha256", "declared_subset_identity_sha256", "subset_selector_sha256")
    if not isinstance(data, dict) or set(data) != set(names) or data != provider.get("data_hashes") or data != inputs.get("data_hashes") or data != {key: stream.get(key) for key in names}:
        raise ValueError("Snapshot/provider/actual input data identities differ")
    if stream.get("complete_declared_train_subset") is not True or stream.get("complete_original_train_split") is not False or stream.get("engineering_subset") is not False or stream.get("listed_images") != 20000 or stream.get("original_split_images") != 118287:
        raise ValueError("Actual initial provider did not authenticate declared20k scope")
    if stream.get("stream_version") != "triflow_online_declared_train20k_v1" or provider.get("stream_version") != stream["stream_version"] or stream.get("base_weights_sha256") != OFFICIAL_SHA256 or provider.get("base_weights_sha256") != OFFICIAL_SHA256:
        raise ValueError("Actual online provider version or official source weights differ")
    if sha256(Path(stream["images_list"])) != data["images_list_sha256"]:
        raise ValueError("Actual initial20k image-list bytes differ")
    annotation_path = Path(stream["annotations"])
    if data["annotation_sha256"] != "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d" or sha256(annotation_path) != data["annotation_sha256"]:
        raise ValueError("Actual original COCO training JSON bytes differ")
    original_ids = original_image_ids_only(annotation_path)
    validate_declared_subset(stream, original_ids)
    del original_ids
    selected = stream["selected_image_ids"]
    identity = config_digest(selected)
    if contract.get("effective_image_identity_sha256") != identity or contract.get("effective_image_order_sha256") != identity or contract.get("declared_subset_identity_sha256") != identity or contract.get("subset_selector_sha256") != data["subset_selector_sha256"] or data["declared_subset_identity_sha256"] != identity:
        raise ValueError("Actual formal active image identity/order differs from declared sorted20k subset")
    observed = provider.get("observed_inputs")
    if not isinstance(observed, dict) or sorted(int(iid) for iid in observed) != selected:
        raise ValueError("Actual provider snapshot has not observed all declared20k image identities")
    if provider.get("sources") != {name: sources[name] for name in ("stream_data.py", "frozen_io.py", "triflow_model.py")}:
        raise ValueError("Actual provider snapshot source hashes differ")
    for row in epochs:
        if row.get("images") != 20000 or row.get("visited_unique_images") != 20000 or row.get("visited_image_identity_sha256") != data["image_identity_sha256"] or row.get("actual_completed_image_ids_match_declared") is not True:
            raise ValueError("Actual completed epoch prefix does not cover every declared20k identity")
    applied, attempts = state.get("applied"), state.get("attempts")
    if type(applied) is not int or applied < 1 or applied != attempts or sum(row.get("applied_steps", -1) for row in epochs) != applied:
        raise ValueError("Actual applied optimizer counts disagree with completed epoch prefix")
    def finite_tree(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tree(item) for item in value)
        return not isinstance(value, float) or math.isfinite(value)
    optimizer = snapshot.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or not optimizer.get("state") or not finite_tree(optimizer) or not isinstance(state.get("finite_groups"), dict) or set(state["finite_groups"]) != GRADIENT_GROUPS or any(value is not True for value in state["finite_groups"].values()):
        raise ValueError("Actual snapshot optimizer/applied gradients have not remained finite")
    frozen = snapshot.get("frozen_integrity", {})
    for key in ("passed", "all_state_exact", "all_params_requires_grad_false", "all_bn_eval", "all_gradients_none"):
        if frozen.get(key) is not True:
            raise ValueError("Snapshot actual frozen model integrity missing: " + key)
    if frozen.get("initial_state_sha256") != frozen.get("final_state_sha256") or frozen.get("initial_state_sha256") != provider.get("frozen_initial_state_sha256"):
        raise ValueError("Snapshot/provider original full frozen model state differs")
    probes = state.get("task_evidence")
    if not isinstance(probes, list) or len(probes) != epoch or [row.get("epoch") for row in probes] != list(range(1, epoch + 1)):
        raise ValueError("Actual task-probe prefix differs from completed epochs")
    joint = 0
    for row in probes:
        phi, attention = row.get("phi_task_gradient_norm"), row.get("cross_attention_task_gradient_norm")
        if row.get("all_task_gradients_finite") is not True or any(not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in (phi, attention)):
            raise ValueError("Actual epoch task-only gradients are not finite")
        actual_joint = phi > 0 and attention > 0
        if row.get("task_gradient_phi_attention_same_probe_nonzero") is not actual_joint:
            raise ValueError("Actual epoch same-probe phi/attention flag differs from measured norms")
        joint += int(actual_joint)
    if joint < 1:
        raise ValueError("No actual same-probe nonzero phi and attention task gradient at this boundary")
    configuration = contract.get("configuration")
    if not isinstance(configuration, dict) or config_digest(configuration) != contract.get("configuration_sha256"):
        raise ValueError("Snapshot module configuration SHA differs")
    if set(configuration) != {field.name for field in fields(TriFlowConfig)} | {"feature_channels", "instance_hidden_channels"}:
        raise ValueError("Snapshot module configuration keys differ")
    config = TriFlowConfig(**{field.name: configuration[field.name] for field in fields(TriFlowConfig)})
    dimensions = provider.get("dimensions", {})
    if config != TriFlowConfig() or configuration["feature_channels"] != dimensions.get("feature_channels") or configuration["instance_hidden_channels"] != dimensions.get("instance_hidden_channels"):
        raise ValueError("Snapshot changed method defaults or actual feature dimensions")
    module = TriFlowModel(configuration["feature_channels"], configuration["instance_hidden_channels"], config).float()
    tensors = snapshot.get("head_state_dict")
    expected = module.state_dict()
    if not isinstance(tensors, dict) or set(tensors) != set(expected):
        raise ValueError("Snapshot module state keys differ")
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tensor.shape != expected[name].shape or not finite_tree(tensor):
            raise ValueError("Snapshot module state is not finite original-shape FP32: " + name)
    module.load_state_dict(tensors, strict=True)
    if state_digest(module) != snapshot.get("head_state_sha256"):
        raise ValueError("Actually loaded epoch module differs from captured state digest")
    capture_path = head_path.parent / "BOUNDARY_CAPTURE.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    snapshot_sha = sha256(head_path)
    if capture.get("capture_passed") is not True or capture.get("epoch") != epoch or capture.get("training_run_id") != run_id or capture.get("snapshot_sha256", capture.get("captured_snapshot_sha256")) != snapshot_sha or capture.get("head_state_sha256") != snapshot["head_state_sha256"]:
        raise ValueError("Actual observer boundary capture does not bind this immutable snapshot")
    capture_bindings = {"snapshot_run_id": run_id, "reason": "epoch_boundary", "completed_epoch_count": epoch, "cursor": cursor,
                        "optimizer_applied": applied, "contract_sha256": snapshot["contract_sha256"], "data_hashes": data,
                        "training_source_sha256": sources, "engineering_only": False, "final_head_kind_claimed": False}
    if any(capture.get(key) != value for key, value in capture_bindings.items()):
        raise ValueError("Actual observer capture source/budget/cursor/state identities differ")
    observer_file = capture.get("observer_source_file")
    if observer_file != "epoch_provenance/train_epoch_observer_r3.py" or sha256(head_path.parent / observer_file) != capture.get("observer_source_sha256"):
        raise ValueError("Actual observer source archive differs from capture")
    addendum_file = capture.get("addendum_source_file")
    if addendum_file != "epoch_provenance/EPOCH_ADDENDUM.md" or sha256(head_path.parent / addendum_file) != capture.get("addendum_sha256"):
        raise ValueError("Actual epoch observation addendum bytes differ")
    if capture.get("callback_script_sha256", {}).get("epoch_evaluator") != sha256(Path(__file__)):
        raise ValueError("Actual epoch callback evaluator source differs from capture")
    captured_execution = capture.get("diagnostics_execution", {})
    verify_contract(head_path.parent, captured_execution)
    if diagnostics_execution is None or captured_execution != diagnostics_execution:
        raise ValueError("Actual R3 evaluator diagnostics runtime differs from paused training execution")
    if snapshot.get("execution_diagnostics_metadata") != captured_execution:
        raise ValueError("Actual R3 snapshot does not bind its declared execution runtime")
    provider_evidence = {"sources": provider["sources"], "data_hashes": data, "dimensions": dimensions, "observed_unique_images": len(observed),
                         "observed_image_identity_sha256": config_digest(sorted(int(iid) for iid in observed)), "visits": provider.get("visits"), "unique_images": provider.get("unique_images"),
                         "frozen_initial_state_sha256": provider["frozen_initial_state_sha256"]}
    derived = {"EPOCHS_AT_BOUNDARY.json": {"derived_from_snapshot_sha256": snapshot_sha, "epochs": epochs},
               "TASK_GRADIENT_EVIDENCE_AT_BOUNDARY.json": {"derived_from_snapshot_sha256": snapshot_sha, "evidence": probes, "same_probe_joint_count": joint},
               "SNAPSHOT_STATE_EVIDENCE.json": {"derived_from_snapshot_sha256": snapshot_sha, "kind": snapshot["kind"], "run_id": run_id, "reason": snapshot["reason"], "cursor": cursor,
                    "completed_epochs": epoch, "applied": applied, "attempts": attempts, "instances": state.get("instances"), "finite_groups": state["finite_groups"],
                    "optimizer_all_finite": True, "head_state_all_finite_fp32": True, "head_state_sha256": snapshot["head_state_sha256"], "frozen_integrity": frozen,
                    "execution_diagnostics_metadata": snapshot["execution_diagnostics_metadata"],
                    "provider_state_evidence": provider_evidence, "contract_sha256": snapshot["contract_sha256"], "is_final_training_audit": False}}
    record = {"kind": snapshot["kind"], "evaluation_epoch": epoch, "epoch": epoch, "intermediate_snapshot": True, "final_training_audit_claimed": False,
              "snapshot_path": str(head_path), "head_path": str(head_path), "snapshot_sha256": snapshot_sha, "head_sha256": snapshot_sha, "snapshot_bytes": head_path.stat().st_size,
              "head_state_sha256": snapshot["head_state_sha256"], "loaded_state_sha256": snapshot["head_state_sha256"], "training_run_id": run_id, "training_run_path": str(origin),
              "reason": "epoch_boundary", "cursor": cursor, "completed_epochs": epoch, "training_budget_epochs": 8, "smoke_only": False,
              "configuration": configuration, "configuration_sha256": contract["configuration_sha256"], "contract": contract, "contract_sha256": snapshot["contract_sha256"],
              "training_sources": sources, "data_hashes": data, "base_weights_sha256": OFFICIAL_SHA256, "declared_train_images": 20000,
              "complete_declared_train_subset": True, "complete_original_train_split": False, "engineering_subset": False, "all_completed_epochs_declared_subset_coverage_verified": True,
              "subset_selector": stream["subset_selector"], "subset_selector_sha256": stream["subset_selector_sha256"], "declared_subset_identity_sha256": stream["declared_subset_identity_sha256"],
              "stream_inputs_sha256": sha256(stream_path), "training_inputs_sha256": sha256(inputs_path), "capture_receipt_sha256": sha256(capture_path),
              "observer_source_sha256": capture["observer_source_sha256"], "addendum_sha256": capture["addendum_sha256"], "phi_attention_same_probe_count": joint,
              "diagnostics_execution": diagnostics_execution,
              "runtime_loaded_source_file": diagnostics_execution["runtime_source_file"],
              "evaluation_diagnostic_scope": getattr(module, "diagnostic_scope", None),
              "all_state_finite_fp32": True, "optimizer_all_finite": True, "no_gt_forward": True, "frozen_integrity": frozen, "_derived_evidence": derived}
    del snapshot, tensors, optimizer, state, provider, stream, inputs, observed
    module = module.to(device).eval()
    module.requires_grad_(False)
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
        scope = outputs.get("diagnostic_scope")
        unmeasured = outputs.get("diagnostics_unmeasured")
        if scope not in ("minimal", "full") or not isinstance(unmeasured, list) or not set(REQUIRED_DIAGNOSTICS).issubset(diagnostics):
            raise ValueError("Actual R3 module did not explicitly report required diagnostics and scope")
        if scope == "minimal" and set(diagnostics) != set(REQUIRED_DIAGNOSTICS):
            raise ValueError("Minimal diagnostic scope must contain exactly five actual diagnostic tensors")
        if scope == "full" and unmeasured:
            raise ValueError("Full diagnostic scope cannot label computed diagnostics as unmeasured")
        all_diagnostics.append({"native_output_rows": rows, "diagnostics": diagnostics,
                                "diagnostic_scope": scope, "diagnostics_unmeasured": unmeasured})
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
    global TriFlowModel
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "head", "images-list", "annotations", "weights", "vendor", "baseline-cache"):
        if name == "head":
            parser.add_argument("--snapshot", "--head", dest="head", required=True)
        else:
            parser.add_argument("--"+name)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-epoch", "--epoch", type=int, choices=range(1, 9), help="Expected actual completed snapshot epoch; otherwise derived from its prefix")
    parser.add_argument("--training-run", type=Path, help="Actual originating training Run directory, if not automatically found")
    parser.add_argument("--ap-only", action="store_true", help="Full5000 Mask/Box AP and fresh identity replay; no paired IoU/bootstrap diagnostic")
    parser.add_argument("--verify-checkpoint-only", action="store_true", help="Strict engineering checkpoint contract; no image inference or AP")
    parser.add_argument("--verify-output", type=Path, help="Optional NEW checkpoint-contract JSON; existing output is refused")
    parser.add_argument("--diagnostics-runtime")
    parser.add_argument("--diagnostics-parity-receipt")
    parser.add_argument("--diagnostics-addendum")
    args = parser.parse_args()
    diagnostic_arguments = (args.diagnostics_runtime, args.diagnostics_parity_receipt, args.diagnostics_addendum)
    if any(diagnostic_arguments) and not all(diagnostic_arguments):
        parser.error("R3 diagnostics runtime, parity receipt and addendum must all be supplied")
    diagnostics_execution = None
    if args.verify_checkpoint_only:
        if all(diagnostic_arguments):
            capture = json.loads((Path(args.head).resolve().parent/"BOUNDARY_CAPTURE.json").read_text(encoding="utf-8"))
            diagnostics_execution = capture["diagnostics_execution"]
            verify_contract(Path(args.head).resolve().parent, diagnostics_execution)
            if (sha256(args.diagnostics_runtime) != diagnostics_execution["runtime_sha256"]
                    or sha256(args.diagnostics_parity_receipt) != diagnostics_execution["parity_sha256"]
                    or sha256(args.diagnostics_addendum) != diagnostics_execution["addendum_sha256"]):
                raise ValueError("Checkpoint-only actual diagnostic inputs differ from capture")
            runtime = import_runtime(Path(args.head).resolve().parent/diagnostics_execution["runtime_source_file"])
            TriFlowModel = runtime.install(TriFlowModel, mode="minimal" if args.ap_only else "full")
        _, metadata = load_epoch_snapshot(args.head, args.device, args.expected_epoch, args.training_run, diagnostics_execution)
        metadata.pop("_derived_evidence")
        receipt = {"status": "passed", "passed": True, "checkpoint_contract_only": True, "inference_executed": False, "ap_measured": False, "head": metadata}
        if args.verify_output is not None:
            with args.verify_output.resolve().open("x", encoding="utf-8") as handle:
                json.dump(receipt, handle, indent=2, allow_nan=False)
        print(json.dumps(receipt, allow_nan=False))
        return 0
    for name in ("root", "run_id", "images_list", "annotations", "weights", "vendor", "baseline_cache"):
        if getattr(args, name) is None:
            parser.error("normal evaluation requires --"+name.replace("_", "-"))
    if not all(diagnostic_arguments):
        parser.error("normal R3 evaluation requires all three explicit diagnostics execution inputs")
    root = Path(args.root).resolve()
    run = root/"runs"/args.run_id
    if (run/"SUMMARY.json").exists() or (run/"BASELINE_REPLAY_IMAGES.jsonl").exists():
        raise FileExistsError("Evaluation history is immutable; retry needs another Run ID")
    run.mkdir(parents=True, exist_ok=True)
    diagnostics_execution = archive_contract(args.diagnostics_runtime, args.diagnostics_parity_receipt, args.diagnostics_addendum, run)
    if sha256(Path(__file__).with_name("diagnostics_runtime.py")) != diagnostics_execution["runtime_sha256"]:
        raise ValueError("Actual evaluation runtime differs from runtime archived by run_stage")
    runtime = import_runtime(run/diagnostics_execution["runtime_source_file"])
    TriFlowModel = runtime.install(TriFlowModel, mode="minimal" if args.ap_only else "full")
    if Path(runtime.__file__).resolve() != (run/diagnostics_execution["runtime_source_file"]).resolve():
        raise ValueError("Runtime module must be loaded from the immutable actual evaluation archive")
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
        module, head_metadata = load_epoch_snapshot(args.head, args.device, args.expected_epoch, args.training_run, diagnostics_execution)
        derived_evidence = head_metadata.pop("_derived_evidence")
        initial_module_digest = state_digest(module)
        coco = COCO(str(Path(args.annotations).resolve()))
        paths, ids = gather_images(Path(args.images_list).resolve(), coco)
        if len(ids) != 5000 or sorted(ids) != sorted(coco.imgs):
            raise ValueError("Formal evaluation requires all 5000 original COCO val2017 images")
        vendor = Path(args.vendor).resolve()
        source_paths = {"segment_validator": "ultralytics/models/yolo/segment/val.py", "ops": "ultralytics/utils/ops.py",
                        "nms": "ultralytics/utils/nms.py", "letterbox": "ultralytics/data/augment.py"}
        configuration = {"evaluator_version": EVALUATOR_VERSION, "evaluator_sha256": sha256(Path(__file__)),
                         "diagnostics_execution": diagnostics_execution,
                         "diagnostic_scope": "minimal" if args.ap_only else "full",
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
        provenance = run/"epoch_provenance"
        provenance.mkdir(exist_ok=True)
        training_origin = Path(head_metadata["training_run_path"])
        for filename in SOURCE_TRAIN_FILES:
            shutil.copy2(training_origin/"source"/filename, provenance/filename)
        for filename in ("TRAINING_INPUTS.json", "STREAM_INPUTS.json"):
            shutil.copy2(training_origin/filename, provenance/filename)
        for filename, actual_payload in derived_evidence.items():
            dump_json(provenance/filename, actual_payload)
        head_metadata["derived_evidence_sha256"] = {filename: sha256(provenance/filename) for filename in derived_evidence}
        dump_json(run/"EVALUATION_INPUTS.json", {"configuration": configuration, "head": head_metadata,
                                               "baseline_cache": str(baseline_cache), "original_baseline_receipt_sha256": sha256(baseline_cache/"COMPLETE.json"),
                                               "started_at": started, "interpreter": sys.executable})
        dump_json(run/"HEAD_PROVENANCE.json", head_metadata)
        dump_json(run/"EPOCH_PROVENANCE.json", head_metadata)
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
        pairs = {} if args.ap_only else paired_metrics("baseline", ["baseline", "triflow"], coco, ids, run, receipts, np, mask_utils,
                                                       SimpleNamespace(bootstrap=5000, bootstrap_seed=0))
        deltas = {"triflow": {kind: {key+"_points": 100*(metrics["triflow"][kind][key]-metrics["baseline"][kind][key])
                                               if metrics["triflow"][kind][key] is not None and metrics["baseline"][kind][key] is not None else None
                                        for key in ("AP", "AP75", "APsmall")} for kind in ("segm", "bbox")}}
        summary = {"status": "complete", "baseline": "baseline", "image_count": 5000, "covers_all_annotation_images": True,
                   "evaluation_epoch": head_metadata["evaluation_epoch"], "training_budget_epochs": 8, "intermediate_snapshot": True,
                   "paired_evaluated": not args.ap_only, "evaluation_metrics_scope": "coco_ap_only" if args.ap_only else "coco_ap_and_paired_diagnostic",
                   "evaluation_scope": "all images in supplied original COCO val2017 annotations", "configuration": configuration,
                   "prediction_receipts": receipts, "metrics": metrics, "delta_vs_baseline": deltas, "paired": pairs,
                   "baseline_replay": replay, "head_provenance": head_metadata, "compiler_counts": compiler_counts,
                   "timing": {"actual_frozen_forward_seconds": total_forward_seconds, "extra_module_seconds": total_module_seconds,
                              "extra_module_ms_per_image": 1000*total_module_seconds/5000,
                              "total_elapsed_seconds": time.monotonic()-began, "includes_python_and_adapter_transfer": True},
                   "selected_instances": modified_instances, "started_at": started, "completed_at": now(),
                   "refined_instances": refined_instances, "unsupported_roi_instances": unsupported_roi_instances,
                   "learned_module_state_exact": initial_module_digest == final_module_digest,
                   "limitations": [f"Actual intermediate completed-epoch prefix1..{head_metadata['evaluation_epoch']} under fixed8epoch budget; not a final training audit; declared20000 train2017 images, single seed, val2017 previously observed",
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
