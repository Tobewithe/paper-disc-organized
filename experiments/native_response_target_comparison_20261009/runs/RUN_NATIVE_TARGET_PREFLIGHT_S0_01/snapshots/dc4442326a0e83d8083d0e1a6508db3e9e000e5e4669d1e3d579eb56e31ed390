"""Resolve existing laptop provenance and freeze the finite producer config."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

from target_common import (FROZEN_SOURCE, OFFICIAL_SHA, PRODUCER_FILES, TRAIN_ANN_SHA, TRAIN_IDS_SHA,
                           TRAIN_LIST_SHA, VAL_LIST_SHA, dump, now, paths_from_list, runtime, sha, synthetic_contract)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "protocol-sha256", "materialize-run", "native-reference", "frozen-scripts", "weights", "vendor", "runner"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    out, root = Path(args.output).resolve(), Path(args.root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "EXECUTION_CONFIG.json").exists():
        raise FileExistsError("Preflight retry needs a fresh Run")
    began = time.perf_counter()
    protocol = root / "PROTOCOL.md"
    if sha(protocol) != args.protocol_sha256 or sha(args.weights) != OFFICIAL_SHA:
        raise ValueError("Declared protocol/checkpoint bytes differ")
    materialize, reference = Path(args.materialize_run).resolve(), Path(args.native_reference).resolve()
    mr = json.loads((materialize / "MATERIALIZATION_RECEIPT.json").read_text())
    nr = json.loads((reference / "EVALUATION_INPUTS.json").read_text())
    cfg = nr["configuration"]
    if not mr["passed"] or not mr["complete_declared_train_subset"] or mr["selected_images"] != 20000:
        raise ValueError("Original materialization receipt incomplete")
    train_list = Path(mr["images_list"])
    train_paths, train_ids = paths_from_list(train_list, TRAIN_LIST_SHA, 20000)
    selected = materialize / "SELECTED_IMAGE_IDS.json"
    manifest = materialize / "SELECTED_IMAGES.sha256"
    if sha(selected) != TRAIN_IDS_SHA or json.loads(selected.read_text()) != train_ids or sha(mr["annotations"]) != TRAIN_ANN_SHA:
        raise ValueError("Original selected20k source/annotation differs")
    if sha(manifest) != mr["selected_manifest_sha256"]:
        raise ValueError("Materialized original JPEG manifest differs")
    expected = {}
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            expected[int(Path(name.strip().lstrip("*")).stem)] = digest
    if set(expected) != set(train_ids):
        raise ValueError("Materialized JPEG manifest identities differ")
    for position, (iid, path) in enumerate(zip(train_ids, train_paths), 1):
        if sha(path) != expected[iid]:
            raise ValueError("Actual original20k JPEG differs: " + str(iid))
        if position % 2000 == 0:
            print(f"PREFLIGHT_JPEG_SHA {position}/20000", flush=True)
    # Native reference records the proven original5000 source identity/order.
    val_list = Path(cfg.get("images_list", ""))
    if not val_list.is_file():
        # Older configuration omitted path but original baseline receipt includes it.
        original = json.loads((reference / "ORIGINAL_BASELINE_RECEIPT.json").read_text())
        val_list = Path(original["fingerprint"]["images_list"]["path"])
    val_paths, val_ids = paths_from_list(val_list, VAL_LIST_SHA, 5000)
    if cfg["source_image_ids"] != val_ids:
        raise ValueError("Original validation list order differs")
    for name, expected_sha in FROZEN_SOURCE.items():
        if sha(Path(args.frozen_scripts) / name) != expected_sha:
            raise ValueError("Reviewed frozen source changed: " + name)
    c = dict(schema="native_target_comparison_v1", root=str(root),
             study_id="STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009",
             protocol=str(protocol), protocol_sha256=args.protocol_sha256,
             frozen_scripts=str(Path(args.frozen_scripts).resolve()), weights=str(Path(args.weights).resolve()),
             vendor=str(Path(args.vendor).resolve()), runner=str(Path(args.runner).resolve()), runner_sha256=sha(args.runner),
             producer_source_sha256={name: sha(Path(__file__).with_name(name)) for name in PRODUCER_FILES},
             train=dict(images_list=str(train_list), selected_ids=str(selected), images_manifest=str(manifest),
                        annotations=mr["annotations"], materialization_run=str(materialize), materialization_receipt_sha256=sha(materialize / "MATERIALIZATION_RECEIPT.json")),
             val=dict(images_list=str(val_list), baseline_cache=str(reference),
                      native5k_inputs_sha256=sha(reference / "EVALUATION_INPUTS.json"), annotations_sha256=cfg["annotations_sha256"]))
    dump(out / "EXECUTION_CONFIG.json", c)
    dump(out / "SUMMARY.json", dict(passed=True, status="preflight_complete", train_images=20000, val_images=5000,
                                    original20k_all_jpeg_sha_exact=True, runtime=runtime(), synthetic_contract=synthetic_contract(),
                                    config_sha256=sha(out / "EXECUTION_CONFIG.json"), protocol_sha256=args.protocol_sha256,
                                    train_first4=train_ids[:4], val_first4=val_ids[:4],
                                    elapsed_seconds=time.perf_counter()-began, completed_at=now(),
                                    gt_parsed=False, fresh_forward=False, model_selection=False))
    dump(out / "COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json"), config_sha256=sha(out / "EXECUTION_CONFIG.json")))


if __name__ == "__main__":
    main()
