"""Bounded strict455, normalization, geometry/ordinal and author-core parity."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch

from segrefiner_runtime import OfficialSegRefiner, normalize_rgb, sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    output = Path(args.root) / "runs" / args.run_id
    if (output / "VERIFICATION.json").exists():
        raise FileExistsError("Verification exists; new retry Run required")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir(exist_ok=True)
    for name in ("verify_segrefiner_runtime.py", "segrefiner_runtime.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    if sha256(args.normalization_reference) != "708cf0b337f5ec6b66d85a45ef58ef59dfb0e80612243459475afb18d381b5ce":
        raise ValueError("Actual primary MMCV1.7.1 normalization source differs")
    shutil.copy2(args.normalization_reference, source / "mmcv_photometric.py")
    tree = ast.parse(Path(args.normalization_reference).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("imnormalize", "imnormalize_")]
    if len(tree.body) != 2:
        raise ValueError("Primary normalization functions unavailable")
    scope = {"np": np, "cv2": cv2}
    exec(compile(ast.fix_missing_locations(tree), str(args.normalization_reference), "exec"), scope)
    rgb = (np.arange(64 * 80 * 3).reshape(64, 80, 3) % 256).astype(np.uint8)
    mean = np.asarray([123.675, 116.28, 103.53], np.float32)
    std = np.asarray([58.395, 57.12, 57.375], np.float32)
    author_normal = scope["imnormalize"](rgb[..., ::-1].copy(), mean, std, to_rgb=True).transpose(2, 0, 1)
    adapted_normal = normalize_rgb(rgb)
    if not np.array_equal(author_normal, adapted_normal):
        raise RuntimeError("Actual author MMCV preprocessing and adapted RGB tensor differ")
    begin = time.perf_counter()
    runtime = OfficialSegRefiner(args.upstream, args.checkpoint, args.device)
    construction_seconds = time.perf_counter() - begin
    write(output / "MODEL_PROVENANCE.json", runtime.provenance)
    checks = {"normalization_primary_function_tensor_bytes_exact": True,
              "strict_complete455_keys": runtime.provenance["strict_loaded_state_keys"] == 455,
              "all_checkpoint_tensor_values_exact": runtime.provenance["all_loaded_tensor_bytes_exact"]}
    masks = np.zeros((4, 64, 80), np.uint8)
    masks[0, 3:27, 4:28] = 1
    masks[1, 20:24, 30:34] = 1
    masks[3, 15:47, 42:62] = 1
    identities = np.asarray([[0, 0, 60, 50, .75, 3], [0, 0, 60, 50, .75, 3],
                              [0, 0, 60, 50, .75, 3], [10, 10, 70, 60, .85, 1]], np.float32)
    # CPU geometry directly compares original crop output with the shim input;
    # author crop/paste implementations remain the sole producer of tensors.
    image, bitmap, boxes, metadata = runtime.author_inputs(rgb, masks, identities)
    actual_image = image.cpu().numpy()[0]
    checks["author_input_normalized_photo_exact"] = np.array_equal(actual_image, author_normal)
    calls = runtime.forward_calls
    tiny_indices = [1, 2]
    tiny_masks, tiny_identity = masks[tiny_indices], identities[tiny_indices]
    tiny_image, tiny_bitmap, tiny_boxes, tiny_meta = runtime.author_inputs(rgb, tiny_masks, tiny_identity)
    with torch.inference_mode():
        tiny_original = runtime.model.simple_test_instance(tiny_meta, img=tiny_image, coarse_masks=tiny_bitmap, dt_bboxes=tiny_boxes)
    # Independent interpretation of this fixed fixture's two identical class3
    # identities; no runtime scatter routine is used as its own golden oracle.
    tiny_restored = np.stack(tiny_original[0][1][3])
    adapted_tiny, tiny_diagnostics = runtime.refine(rgb, masks, tiny_indices, 139, identities)
    checks["author_alltiny_empty_identity_fallback_exact"] = np.array_equal(tiny_restored, tiny_masks) and np.array_equal(adapted_tiny, masks)
    checks["alltiny_denoiser_and_sampling_skipped"] = runtime.forward_calls == calls and tiny_diagnostics["actual_denoiser_forward_calls"] == 0
    zero, zero_diagnostics = runtime.refine(rgb, np.zeros((0, 64, 80), np.uint8), [], 139, np.zeros((0, 6), np.float32))
    checks["zero_native_candidates_preserved"] = zero.shape == (0, 64, 80) and zero_diagnostics["actual_denoiser_forward_calls"] == 0
    unselected, _ = runtime.refine(rgb, masks, [], 139, identities)
    checks["no_supported_scope_preserves_all_identity"] = np.array_equal(unselected, masks)
    golden = None
    if args.device.startswith("cuda"):
        # A genuine forward of official weights through the untouched author
        # core, then the scope wrapper, starting from exactly the same seed.
        runtime.seed(139)
        with torch.inference_mode():
            original = runtime.model.simple_test_instance(metadata, img=image, coarse_masks=bitmap, dt_bboxes=boxes)
        # Fixture's author valid/tiny order=[0,3,1,2], class1=[3], class3=[0,1,2].
        # Use those independently specified ordinals as the golden interpretation.
        original_masks = np.empty_like(masks)
        original_masks[3] = original[0][1][1][0]
        original_masks[0] = original[0][1][3][0]
        original_masks[1] = original[0][1][3][1]
        original_masks[2] = original[0][1][3][2]
        original_rng = torch.cuda.get_rng_state().clone()
        adapted, diagnostics = runtime.refine(rgb, masks, [0, 1, 2, 3], 139, identities)
        checks["official_weight_author_core_vs_wrapper_binary_exact"] = np.array_equal(original_masks, adapted)
        checks["same_state_author_core_vs_wrapper_rng_exact"] = diagnostics["torch_cuda_rng_after_sha256"] == hashlib.sha256(original_rng.cpu().numpy().tobytes()).hexdigest()
        checks["external_cpu_cuda_rng_restored"] = diagnostics["external_torch_cpu_rng_restored"] and diagnostics["external_torch_cuda_rng_restored"] and torch.equal(original_rng, torch.cuda.get_rng_state())
        checks["actual_six_step_forward_count"] = diagnostics["actual_denoiser_forward_calls"] == 6
        checks["original_mask_ordinal_scatter_exact"] = all(np.array_equal(original_masks[index], adapted[index]) for index in range(4))
        np.save(output / "AUTHOR_CORE_MASKS.npy", original_masks, allow_pickle=False)
        np.save(output / "WRAPPER_MASKS.npy", adapted, allow_pickle=False)
        golden = diagnostics
    integrity = runtime.verify_unchanged()
    passed = all(checks.values())
    report = {"status": "passed" if passed else "failed", "passed": passed, "checks": checks,
              "runtime_source_sha256": sha256(Path(__file__).with_name("segrefiner_runtime.py")),
              "verifier_source_sha256": sha256(__file__), "model_provenance_sha256": sha256(output / "MODEL_PROVENANCE.json"),
              "device": args.device, "actual_torch": torch.__version__,
              "actual_gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
              "construction_and_strict_load_seconds": construction_seconds, "frozen_integrity": integrity,
              "official_weight_gpu_forward_exercised": args.device.startswith("cuda"),
              "synthetic_geometry_fixture_only": True, "gt_used": False, "ap_measured": False,
              "same_state_fixture_diagnostics": golden,
              "normalization_tensor_sha256": hashlib.sha256(adapted_normal.tobytes()).hexdigest()}
    write(output / "VERIFICATION.json", report)
    if not passed:
        raise RuntimeError("Faithfulness verification failed; preserve the Run")
    print(json.dumps({"passed": True, "device": args.device, "checks": len(checks)}), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "upstream", "checkpoint", "normalization-reference"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    raise SystemExit(run(parser.parse_args()))
