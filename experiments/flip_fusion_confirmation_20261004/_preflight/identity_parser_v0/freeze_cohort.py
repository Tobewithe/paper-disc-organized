"""Freeze a method-development holdout using IDs only (Python standard library).

This is a local metadata audit, never a model experiment. Prior full-COCO model
training and unindexed historical exposure remain explicit limitations. Run once
before inference; an existing freeze is never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
from pathlib import Path
import re


SEED = 20261004
COUNT = 1536
SKIP_PARTS = {"snapshots", "_transfer", "_collection_history", "__pycache__", "pkg", "vendor", ".git",
              "archive", "archives", "_archive", "_archives"}
PATTERNS = (
    "*split*.json", "*selection*.json", "fit_metadata.json", "*manifest*.json",
    "*ids.json", "*index.json", "*order.json", "*per_image*.jsonl",
    "*per_candidate*.jsonl",
)
ID_LIST_KEYS = {"fit", "dev", "val", "train", "test", "eval", "evaluation", "calibration",
                "selection", "explore", "confirm", "reserved", "heldout", "holdout", "images",
                "ids", "excluded", "validation", "training", "image_ids"}
NON_IMAGE_KEYS = {"annotation_ids", "all_annotation_ids", "raw_ids", "target_gt_idx", "owners",
                  "levels", "pyramid_levels", "category_ids", "classes", "shape", "image_shape"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(data)


def is_image_list(path):
    if not path:
        return True  # An explicit *_ids.json can be a root integer list.
    last = path[-1].lower()
    if last in NON_IMAGE_KEYS or "annotation" in last or last.startswith("raw_"):
        return False
    tokens = set(re.split(r"[^a-z]+", last))
    return bool(tokens & ID_LIST_KEYS) or (last.isdigit() and "folds" in path)


def extract(obj, output, scopes, path=(), *, identity_only=False):
    if isinstance(obj, dict):
        value = obj.get("image_id")
        if type(value) is int:
            output.add(value)
            scopes.add(".".join(path + ("image_id",)))
        elif isinstance(value, str) and value.isdigit():
            output.add(int(value))
            scopes.add(".".join(path + ("image_id",)))
        for key, value in obj.items():
            key = str(key)
            if not identity_only and path and path[-1].lower() in {"images", "by_image", "image_records"} and key.isdigit():
                output.add(int(key))
                scopes.add(".".join(path) + ".<image_id_key>")
            if key == "identities" and isinstance(value, list):
                valid = [row[0] for row in value if isinstance(row, list) and row and type(row[0]) is int]
                output.update(valid)
                if valid:
                    scopes.add(".".join(path + (key, "first_column")))
            elif isinstance(value, (dict, list)):
                extract(value, output, scopes, path + (key,), identity_only=identity_only)
    elif isinstance(obj, list):
        if not identity_only and obj and all(type(x) is int for x in obj) and is_image_list(path):
            output.update(obj)
            scopes.add(".".join(path) or "<root_integer_list>")
        else:
            for value in obj:
                if isinstance(value, (dict, list)):
                    extract(value, output, scopes, path + ("[]",), identity_only=identity_only)


def source_ids(path):
    ids, scopes = set(), set()
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        extract(json.loads(line), ids, scopes, identity_only=True)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"line {number}: {exc}") from exc
    else:
        extract(json.loads(path.read_text(encoding="utf-8-sig")), ids, scopes)
    return ids, sorted(scopes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = [out / name for name in ("CONFIRM_IDS.json", "EXCLUSION_IDS.json", "EXPOSURE_AUDIT.json")]
    if any(path.exists() for path in outputs):
        raise RuntimeError("Cohort already frozen or partial output exists; do not silently overwrite")
    annotation = (args.annotations or root / "assets/datasets/coco/annotations/instances_train2017.json").resolve()
    # Only the images[].id field contributes to the universe. Annotation geometry,
    # labels, classes and model metrics are never used for cohort selection.
    coco = json.loads(annotation.read_text(encoding="utf-8"))
    universe = {int(image["id"]) for image in coco["images"]}
    del coco
    if len(universe) != 118287:
        raise RuntimeError(f"Unexpected COCO train2017 universe: {len(universe)}")

    files = []
    for path in (root / "experiments").rglob("*"):
        if not path.is_file() or out == path or out in path.parents:
            continue
        if any(part.lower() in SKIP_PARTS for part in path.relative_to(root).parts):
            continue
        if any(fnmatch.fnmatch(path.name.lower(), pattern) for pattern in PATTERNS):
            files.append(path)
    files.sort(key=lambda path: path.relative_to(root).as_posix().lower())
    union, source_records, failures, parsed_hashes = set(), [], [], {}
    for path in files:
        relative, digest = path.relative_to(root).as_posix(), sha(path)
        try:
            if digest in parsed_hashes:
                ids, scopes = parsed_hashes[digest]
            else:
                ids, scopes = source_ids(path)
                parsed_hashes[digest] = ids, scopes
            train_ids = ids & universe
            new = train_ids - union
            union.update(train_ids)
            source_records.append(dict(path=relative, sha256=digest, bytes=path.stat().st_size,
                extracted_distinct_ids=len(ids), train2017_ids=len(train_ids), newly_excluded=len(new),
                id_fields=scopes, interpretation="conservatively exclude explicitly listed or analyzed identity; planned lists are excluded even if execution is not established"))
        except (ValueError, TypeError, UnicodeError) as exc:
            failures.append(dict(path=relative, sha256=digest, bytes=path.stat().st_size, error=str(exc)))

    # Fail loudly if major, already-reviewed method-development records vanished.
    mandatory = (
        "experiments/box_guided_coefficient_training_20261002/SPLIT.json",
        "experiments/coefficient_predictability_20260924/SPLIT.json",
        "experiments/coefficient_pixel_direction_20260927/DATA_MANIFEST.json",
        "experiments/flip_response_projection_20261004/SPLIT.json",
    )
    represented = {row["path"] for row in source_records if row["extracted_distinct_ids"]}
    if set(mandatory) - represented:
        raise RuntimeError(f"Missing required exposure sources: {set(mandatory) - represented}")
    recent = json.loads((root / mandatory[0]).read_text(encoding="utf-8-sig"))
    recent_train = set(recent["fit"]) | set(recent["dev"])
    if len(recent_train) != 11000 or not recent_train <= union:
        raise AssertionError("Recent 10000 fit + 1000 dev pool not entirely excluded")
    old = json.loads((root / mandatory[1]).read_text(encoding="utf-8-sig"))
    if not (set(old["fit"]) | set(old["dev"])) <= union:
        raise AssertionError("Original coefficient 800 fit + 200 dev not entirely excluded")
    available = universe - union
    if len(available) < COUNT:
        raise RuntimeError("Insufficient method-development holdout universe")
    rank = lambda image_id: (hashlib.sha256(f"{SEED}:{image_id}".encode("ascii")).hexdigest(), image_id)
    selected = sorted(available, key=rank)[:COUNT]
    assert len(set(selected)) == COUNT and not set(selected) & union
    timestamp = datetime.now(timezone.utc).isoformat()

    exclusion = dict(schema="explicit-image-exposure-exclusions-v1", recorded_at=timestamp,
        excluded_train2017_ids=sorted(union), count=len(union),
        scope="Conservative union of machine-readable IDs in the audited source patterns; not a complete historical exposure reconstruction")
    write(outputs[1], exclusion)
    confirmation = dict(schema="flip-mix-method-development-holdout-v1", recorded_at=timestamp,
        source_split="train2017", seed=SEED, planned_images=COUNT, confirm=selected,
        selection_rule="Sort remaining image IDs by (SHA256(ASCII('20261004:' + decimal image_id)), image_id); take first 1536",
        selection_fields_used=["COCO images[].id", "explicit historical image IDs"],
        exclusion_ids_sha256=sha(outputs[1]), source_annotation_sha256=sha(annotation),
        claims="Method-development holdout conditional on indexed exposure sources; not model-unseen, not whole-project-unused, not a new blind test")
    write(outputs[0], confirmation)
    reserved_path = root / "experiments/coco_supervision_mechanisms_20260920/confirm_reserved_ids.json"
    reserved, _ = source_ids(reserved_path)
    historical_report = root / "experiments/counterfactual_p3_fullcoco_20260914/RESULTS_20260915.md"
    audit = dict(schema="flip-mix-exposure-audit-v1", completed=True, recorded_at=timestamp,
        seed=SEED, planned_images=COUNT, source_train_images=len(universe),
        excluded_explicit_ids=len(union), eligible_after_explicit_exclusion=len(available),
        selected_images=len(selected), selected_exclusion_overlap=0,
        recent_11000_pool=dict(images=len(recent_train), complement=len(universe-recent_train), selected_overlap=len(set(selected)&recent_train)),
        reserved_500=dict(path=reserved_path.relative_to(root).as_posix(), images=len(reserved),
            excluded_conservatively=len(reserved & union), note="Reservation from an old study is not evidence of continued non-use; entire listed reserve is conservatively excluded here"),
        source_universe=dict(path=str(annotation), sha256=sha(annotation), used_field="images[].id only"),
        code_sha256=sha(__file__), confirm_ids_sha256=sha(outputs[0]), exclusion_ids_sha256=sha(outputs[1]),
        extraction_patterns=list(PATTERNS), skipped_directory_components=sorted(SKIP_PARTS),
        discovery_scope="Local canonical experiments directory at freezing time; JSON/JSONL identity-bearing source patterns only; conservative planned/excluded lists included",
        evidence_sources=source_records, unreadable_sources=failures,
        non_exclusion_exposure=[
            dict(type="official pretraining", scope="Official COCO pretrained model has seen train2017; selected cohort is not unseen by its pretraining"),
            dict(type="historical unrelated project model training", scope="Earlier baseline and IGP3 models were each actually trained on full 118287 train2017 images for one epoch; this fact is disclosed, not treated as selective MIX-method development exposure", path=historical_report.relative_to(root).as_posix(), sha256=sha(historical_report)),
        ],
        limitations=[
            "The conditional independence claim concerns current frozen MIX configuration development, not all prior project or official-model data exposure.",
            "Historical image viewing or training stored only in unindexed formats, absent artifacts, scripts with implicit selection or unrecovered runs is not comprehensively known.",
            "Unreadable sources remain unknown; no claim that selected identities were never present in them.",
            "Files containing per-image/per-candidate results contribute identity fields only, never their scores, labels, failure rates or model effects to selection.",
            "All original planned 1536 identities remain fixed; no-positive images must be reported, never replaced based on candidate count or effect.",
            "No existing evaluated cache for this new cohort is claimed; the authorized server must create official original-weight candidates with the frozen loader.",
        ],
        machine_assets=dict(coco_train_zip="/autodl-pub/data/COCO2017/train2017.zip",
            coco_val_zip="/autodl-pub/data/COCO2017/val2017.zip",
            provenance="Paths read from completed fast-screen prepare_online.py and RUN_CONFIG.json; no new remote existence check in this metadata audit",
            original_cache="/root/prototype_readout_fast_screen_20261003/cache", original_cache_scope="old 1024 fit + 256 dev; not the new confirmation cohort"))
    write(outputs[2], audit)
    print(json.dumps({"completed":True,"excluded":len(union),"eligible":len(available),"selected":COUNT,
        "unreadable_sources":len(failures),"sources":len(source_records),"confirm_sha256":sha(outputs[0]),
        "exclusion_sha256":sha(outputs[1]),"audit_sha256":sha(outputs[2])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
