"""Real-model gradient and overlay diagnostic for the ACD prototype-tail screen.

The official checkpoint runs an actual training-mode forward at small resolution.
GT and assignments are diagnostic fixtures made from its detached masks so ACD
has an eligible action. This measures graph reachability, not accuracy or fit.
Production thresholds, objectives, detach, optimizer construction and EMA timing
are unchanged. The one-step diagnostic uses FP32 without AMP.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--image", type=Path, help="Optional local image instead of seeded random pixels")
    args = parser.parse_args()
    if args.imgsz < 64 or args.imgsz % 32:
        parser.error("Diagnostic input must be a multiple of 32 and at least 64")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Use a new diagnostic Run or output file")
    sys.path.insert(0, str(args.vendor.resolve()))

    import torch
    import train_acd as acd
    from ultralytics.utils.torch_utils import ModelEMA

    torch.set_num_threads(4)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    original_root, original_weights = acd.RUN_ROOT, acd.BASE_WEIGHTS
    receipt = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "device": str(device), "imgsz": args.imgsz, "torch": torch.__version__,
               "ultralytics": acd.ultralytics.__version__, "weight": str(args.weight.resolve()),
               "weight_sha256": acd._file_hash(args.weight),
               "training_script_sha256": acd._file_hash(Path(acd.__file__)),
               "verification_script_sha256": acd._file_hash(Path(__file__)),
               "scope": "Actual official model forward and one diagnostic optimizer step, fixture GT/assignment; no AP claim",
               "fixtures": "GT=detached tau-action mask for a selected real anchor; full-image support; no detector assignment claim",
               "production_acd_constants_changed": False, "one2one_detach_changed": False,
               "diagnostic_precision": "FP32, AMP disabled"}
    try:
        model = acd.YOLO(str(args.weight)).model.float().to(device).train()
        trainable = acd._freeze_for_mask_tail(model)
        frozen = set(model.state_dict()) - trainable
        initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        initial_frozen = acd._state_digest(model, frozen)
        assert all(not module.training for module in model.modules()
                   if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
        if args.image:
            from PIL import Image
            import numpy as np
            image = Image.open(args.image).convert("RGB").resize((args.imgsz, args.imgsz))
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
            sample = pixels[None].to(device)
            receipt["image"] = str(args.image.resolve())
            receipt["image_sha256"] = acd._file_hash(args.image)
        else:
            sample = torch.rand((1, 3, args.imgsz, args.imgsz), device=device)
            receipt["input"] = "seed-0 uniform random image"
        preds = model(sample)
        assert set(("one2many", "one2one")) <= preds.keys()
        p_many = preds["one2many"]["proto"]
        p_one = preds["one2one"]["proto"]
        p_many = p_many[0] if isinstance(p_many, tuple) else p_many
        p_one = p_one[0] if isinstance(p_one, tuple) else p_one
        assert p_many.requires_grad and not p_one.requires_grad
        assert torch.equal(p_many.detach(), p_one), "Official detached one2one prototype differs"

        group_parameters = {
            group: [(name, param) for name, param in model.named_parameters()
                    if name in trainable and name.startswith(prefixes)]
            for group, prefixes in acd.PARAMETER_GROUPS.items()}
        all_parameters = [(name, param) for values in group_parameters.values() for name, param in values]

        def reachability(loss, retain_graph=True):
            gradients = torch.autograd.grad(loss, [param for _, param in all_parameters],
                                            allow_unused=True, retain_graph=retain_graph)
            by_name = {name: gradient for (name, _), gradient in zip(all_parameters, gradients)}
            result = {}
            for group, values in group_parameters.items():
                connected = [name for name, _ in values if by_name[name] is not None]
                nonzero = [name for name in connected if bool(by_name[name].abs().sum() > 0)]
                assert all(bool(torch.isfinite(by_name[name]).all()) for name in connected)
                result[group] = {"connected_parameter_names": connected, "nonzero_parameter_names": nonzero,
                                 "gradient_l1": sum(float(by_name[name].abs().sum()) for name in connected),
                                 "all_finite": True}
            return result

        auxiliary_losses, checks = [], {}
        for branch in ("one2many", "one2one"):
            prediction = preds[branch]
            proto = prediction["proto"]
            proto = proto[0] if isinstance(proto, tuple) else proto
            coefficients = prediction["mask_coefficient"]
            with torch.no_grad():
                logits = torch.einsum("ca,chw->ahw", coefficients[0].detach(), proto[0].detach())
                base = logits > 0
                acted = logits > acd.ACD_TAU
                base_count = base.flatten(1).sum(1)
                acted_count = acted.flatten(1).sum(1)
                # Fixture GT=acted. Coverage is 1; the action's IoU is 1.
                gains = 1.0 - acted_count.float() / base_count.clamp_min(1)
                eligible = (acted_count > 0) & (base_count > acted_count) & (gains >= acd.ACD_MIN_GAIN)
                candidates = eligible.nonzero(as_tuple=False).view(-1)
                if not candidates.numel():
                    raise RuntimeError(f"No eligible real {branch} anchor in this diagnostic input; use --image with a real COCO image")
                anchor = int(candidates[0])
                gt = acted[anchor].float()[None]
            count = coefficients.shape[-1]
            fg = torch.zeros((1, count), dtype=torch.bool, device=device)
            fg[0, anchor] = True
            boxes = torch.zeros((1, count, 4), device=device)
            boxes[0, anchor] = torch.tensor([0, 0, args.imgsz, args.imgsz], device=device)
            assigned = (fg, torch.zeros((1, count), dtype=torch.long, device=device), boxes, None, None)
            batch = {"masks": gt, "batch_idx": torch.zeros(1, device=device)}
            criterion = SimpleNamespace(device=device, stride=model.model[-1].stride, overlap=True)
            before = acd.STATS[branch]["action_positive"]
            aux = acd._acd_aux(criterion, prediction, batch, assigned, branch)
            assert bool(torch.isfinite(aux)) and float(aux.detach()) > 0
            assert acd.STATS[branch]["action_positive"] == before + 1
            graph = reachability(aux)
            assert graph[branch]["nonzero_parameter_names"], f"Aux did not reach {branch} coefficients"
            other = "one2one" if branch == "one2many" else "one2many"
            assert not graph[other]["connected_parameter_names"]
            if branch == "one2many":
                for prefix in acd.PROTO_TAIL_PREFIXES:
                    assert any(name.startswith(prefix) for name in graph["proto_tail"]["nonzero_parameter_names"]), \
                        f"Aux did not reach prototype block {prefix}"
            else:
                assert not graph["proto_tail"]["connected_parameter_names"], "One2one prototype detach was lost"
            checks[branch] = {"selected_anchor": anchor, "fixture_gt_pixels": int(gt.sum()),
                              "fixture_false_positive_pixels": int(base_count[anchor] - acted_count[anchor]),
                              "fixture_action_iou_gain": float(gains[anchor]),
                              "auxiliary_loss": float(aux.detach()), "gradient_reachability": graph}
            auxiliary_losses.append(aux)

        def new_stats():
            stats = defaultdict(int)
            for name in ("all_finite", "observed_all_gradients_finite", "observed_all_scaled_gradients_finite",
                         "all_observed_nonfinite_steps_skipped", "all_applied_gradients_finite"):
                stats[name] = True
            stats["maximum_absolute_gradient"] = 0.0
            return stats

        trainer = object.__new__(acd.ACDTrainer)
        trainer.model = model
        config = acd.official_weight_config(args.weight)
        # Use unchanged official optimizer grouping and supported saved config.
        trainer.optimizer = trainer.build_optimizer(model, name=config["optimizer"], lr=config["lr0"],
                                                    momentum=config["momentum"], decay=config["weight_decay"])
        trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
        trainer.ema = ModelEMA(model)
        trainer.amp = False
        trainer.accumulate = 1
        trainer.epoch = 0
        trainer._batch_counter = 1
        trainer._trainable_names = trainable
        trainer._frozen_names = frozen
        trainer._frozen_initial = initial_frozen
        trainer._initial_coeff = {name: initial[name].float() for name in trainable}
        trainer._gradient_stats = {group: new_stats() for group in acd.PARAMETER_GROUPS}
        trainer._optimizer_step_stats = new_stats()
        trainer._finite_loss = True
        trainer._input_stream_digest = hashlib.sha256(b"diagnostic one-batch fixture")
        diagnostic_root = args.output.parent / "diagnostic_artifacts"
        diagnostic_root.mkdir(parents=True, exist_ok=False)
        trainer.last = diagnostic_root / "not_a_training_checkpoint.pt"
        acd.RUN_ROOT, acd.BASE_WEIGHTS = diagnostic_root, args.weight.resolve()
        reference = SimpleNamespace(model=copy.deepcopy(model), ema=copy.deepcopy(trainer.ema),
                                    scaler=torch.amp.GradScaler("cuda", enabled=False))
        reference.optimizer = acd.SegmentationTrainer.build_optimizer(
            reference, acd._OptimizerView(reference.model), name=config["optimizer"], lr=config["lr0"],
            momentum=config["momentum"], decay=config["weight_decay"])
        loss = acd.ACD_WEIGHT * sum(auxiliary_losses)
        loss.backward()
        reference_params = dict(reference.model.named_parameters())
        for name, param in model.named_parameters():
            reference_params[name].grad = None if param.grad is None else param.grad.detach().clone()
        trainer.optimizer_step()
        acd.SegmentationTrainer.optimizer_step(reference)
        for name, value in model.state_dict().items():
            assert torch.equal(value, reference.model.state_dict()[name]), f"Audit altered optimizer: {name}"
            if name in frozen:
                assert torch.equal(value.cpu(), initial[name]), f"Frozen live state changed: {name}"
                assert torch.equal(trainer.ema.ema.state_dict()[name].cpu(), initial[name]), f"Frozen EMA changed: {name}"
            else:
                assert torch.equal(trainer.ema.ema.state_dict()[name], reference.ema.ema.state_dict()[name]), \
                    f"Audit altered trainable EMA: {name}"
        assert trainer.ema.updates == reference.ema.updates == 1
        trainer._finish_audit()
        assert trainer._audit["audit_passed"]
        for group, update in trainer._audit["branch_updates"].items():
            assert update["changed_parameter_count"] > 0
            assert update["gradients"]["applied_steps"] == update["gradients"]["nonzero_steps"] == 1
        overlay = torch.load(diagnostic_root / "mask_final_ema.pt", map_location="cpu", weights_only=False)
        expected_keys = {name for name in model.state_dict() if acd._is_mask_scope(name)}
        assert set(overlay["state_dict"]) == expected_keys
        assert overlay["scope"] == list(acd.MASK_PREFIXES)
        assert overlay["kind"] == "mask_coefficient_proto_tail_ema_final"
        assert overlay["epoch"] == 1 and overlay["audit_passed"]
        for name, value in overlay["state_dict"].items():
            assert torch.equal(value, trainer.ema.ema.state_dict()[name].cpu()), f"Wrong EMA overlay: {name}"
            assert value.dtype == trainer.ema.ema.state_dict()[name].dtype
        reconstructed = dict(initial)
        reconstructed.update(overlay["state_dict"])
        # This exact strict full-state reconstruction is also used in evaluation.
        reference.model.load_state_dict(reconstructed, strict=True)
        assert acd._state_digest(reference.model, frozen)["sha256"] == initial_frozen["sha256"]
        for name in expected_keys:
            assert torch.equal(reference.model.state_dict()[name].cpu(), overlay["state_dict"][name])
        receipt.update(status="PASS", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                       trainable_parameters=sorted(trainable),
                       trainable_parameter_count=sum(param.numel() for param in model.parameters() if param.requires_grad),
                       scope_prefixes=list(acd.MASK_PREFIXES), checks=checks,
                       official_one2one_proto_detached=True, all_bn_eval_and_affine_frozen=True,
                       frozen_live_and_ema_exact=True, audited_optimizer_matches_official=True,
                       trainable_ema_matches_official=True, overlay_full_scope_exact=True,
                       overlay_includes_frozen_bn_original_values=True, overlay_strict_reconstruction=True,
                       overlay_diagnostic_epoch=1, overlay_training_claim=False,
                       optimizer_audit=trainer._audit["optimizer_audit"],
                       parameter_updates=trainer._audit["branch_updates"],
                       diagnostic_artifact_directory=str(diagnostic_root))
    except BaseException as error:
        receipt.update(status="FAIL", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                       exception_type=type(error).__name__, exception=str(error), traceback=traceback.format_exc())
        args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
        print("TRAINING_VERIFICATION", json.dumps(receipt, allow_nan=False), flush=True)
        raise
    finally:
        acd.RUN_ROOT, acd.BASE_WEIGHTS = original_root, original_weights
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    print("TRAINING_VERIFICATION", json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
