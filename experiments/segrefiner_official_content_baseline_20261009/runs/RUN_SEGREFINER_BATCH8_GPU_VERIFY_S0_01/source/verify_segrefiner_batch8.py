"""Same-batch8 author core vs wrapper, fixed multi-batch/tiny/empty fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from segrefiner_runtime import normalize_rgb, sha256
from segrefiner_batch8_runtime import OfficialSegRefinerBatch8, BASE_RUNTIME_SHA


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main(args):
    output = Path(args.root) / "runs" / args.run_id
    if (output / "VERIFICATION.json").exists():
        raise FileExistsError("Existing verification; use a new Run")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir(exist_ok=True)
    names = ("verify_segrefiner_batch8.py", "segrefiner_batch8_runtime.py", "segrefiner_runtime.py")
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    runtime = OfficialSegRefinerBatch8(args.upstream, args.checkpoint, "cuda", 20261009)
    rgb = (np.arange(64 * 80 * 3).reshape(64, 80, 3) % 256).astype(np.uint8)
    masks = np.zeros((13, 64, 80), np.uint8)
    valid = [0, 2, 4, 5, 6, 8, 9, 10, 11, 12]
    for index in valid:
        masks[index, 5:29, 6:30] = 1
    masks[4] = 0
    masks[4, 16:32, 35:67] = 1  # Exactly512, must enter model.
    masks[1, 16:32, 35:67] = 1
    masks[1, 16, 35] = 0         # Exactly511, must keep tiny fallback.
    masks[7, 34:38, 50:54] = 1
    labels = [3, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 3]
    identities = np.asarray([[0, 0, 70, 60, .75, label] for label in labels], np.float32)
    image, bitmaps, boxes, metadata = runtime.author_inputs(rgb, masks, identities)
    runtime.seed(139)
    with torch.inference_mode():
        author_result = runtime.model.simple_test_instance(metadata, img=image, coarse_masks=bitmaps, dt_bboxes=boxes)
    author_rng = torch.cuda.get_rng_state().clone()
    golden = np.empty_like(masks)
    # Fixed independent fixture interpretation of original valid/tiny/class
    # order. Duplicate boxes/confidences cannot be used to recover identity.
    for label, ordinals in {1: [2, 5, 9, 11, 7], 3: [0, 4, 6, 8, 10, 12, 1, 3]}.items():
        if len(author_result[0][1][label]) != len(ordinals):
            raise RuntimeError("Author fixed fixture class count differs")
        for slot, ordinal in enumerate(ordinals):
            golden[ordinal] = author_result[0][1][label][slot]
    adapted, diagnostics = runtime.refine(rgb, masks, list(range(13)), 139, identities)
    zero, zero_diag = runtime.refine(rgb, np.zeros((0, 64, 80), np.uint8), [], 139, np.zeros((0, 6), np.float32))
    tiny, tiny_diag = runtime.refine(rgb, masks, [1, 3, 7], 139, identities)
    checks = {"strict455_all_values_exact": runtime.provenance["strict_loaded_state_keys"] == 455 and runtime.provenance["all_loaded_tensor_bytes_exact"],
              "author_method_bodies_unchanged": runtime.provenance["adaptation"]["original_method_bodies_changed"] is False,
              "same_batch8_author_core_vs_wrapper_all_binary_exact": np.array_equal(golden, adapted),
              "same_batch8_actual_after_rng_exact": diagnostics["torch_cuda_rng_after_sha256"] == hashlib.sha256(author_rng.numpy().tobytes()).hexdigest(),
              "external_rng_restore_exact": diagnostics["external_torch_cpu_rng_restored"] and diagnostics["external_torch_cuda_rng_restored"],
              "multi_batch8_plus2_true_forward_count": diagnostics["model_valid_area512"] == 10 and diagnostics["actual_denoiser_forward_calls"] == 12,
              "area512_enters_model_area511_tiny": masks[4].sum() == 512 and masks[1].sum() == 511 and diagnostics["model_valid_area512"] == 10,
              "all_tiny_511_empty_and16_fallback_exact": np.array_equal(adapted[[1, 3, 7]], masks[[1, 3, 7]]) and np.array_equal(tiny, masks) and tiny_diag["actual_denoiser_forward_calls"] == 0,
              "zero_candidate_outer_short_circuit": zero.shape == (0, 64, 80) and zero_diag["actual_denoiser_forward_calls"] == 0,
              "fixed_original_mask_ordinal_restore_exact": all(np.array_equal(golden[index], adapted[index]) for index in range(13)),
              "unchanged_6steps_fp32_crop_pad512_settings": runtime.model.step == 6 and runtime.model.test_cfg == {"pad_width": 20, "model_size": 256, "batch_max": 8, "area_thr": 512},
              "same_seed20261009_plus139": diagnostics["seed"] == 20261148,
              "normalization_tensor_matches_previously_proven_base": np.array_equal(image.cpu().numpy()[0], normalize_rgb(rgb))}
    np.save(output / "AUTHOR_CORE_BATCH8_MASKS.npy", golden, allow_pickle=False)
    np.save(output / "WRAPPER_BATCH8_MASKS.npy", adapted, allow_pickle=False)
    write(output / "MODEL_PROVENANCE.json", runtime.provenance)
    integrity = runtime.verify_unchanged()
    passed = all(checks.values())
    write(output / "VERIFICATION.json", {"status": "passed" if passed else "failed", "passed": passed, "checks": checks,
                  "runtime_source_sha256": sha256(Path(__file__).with_name("segrefiner_batch8_runtime.py")),
                  "base_runtime_source_sha256": BASE_RUNTIME_SHA, "verifier_source_sha256": sha256(__file__),
                  "official_weight_gpu_forward_exercised": True, "hardware_batch_max": 8,
                  "actual_torch": torch.__version__, "actual_gpu": torch.cuda.get_device_name(),
                  "frozen_integrity": integrity, "same_state_fixture_diagnostics": diagnostics,
                  "synthetic_fixture_only": True, "valid_fixture_instances": 10,
                  "batch32_output_bitwise_equivalence_claimed": False, "gt_used": False, "ap_measured": False})
    if not passed:
        raise RuntimeError("Faithful batch8 verification failed")
    print(json.dumps({"passed": True, "checks": len(checks), "actual_multi_batch_forward_calls": 12}), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "upstream", "checkpoint"):
        parser.add_argument("--" + name, required=True)
    raise SystemExit(main(parser.parse_args()))
