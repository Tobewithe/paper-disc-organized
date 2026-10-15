"""Read-only input verification; writes only new S3 preparation receipts."""
from common import *
from datetime import datetime, timezone
import json
import os
import platform
import time


def main():
    started = time.monotonic()
    trace = read_json(BASE / 'trace/TRACE_MANIFEST.json')
    require(len(trace['images']) == 5000 and len(set(trace['image_ids'])) == 5000, 'not full cache')
    require(sha256(BASE / 'trace/TRACE_MANIFEST.json') == 'c4c2b11eb0f6c9628c7ab3494b74b08d6e0835b46426413fbdafb2b2119abfab', 'trace identity changed')
    files = [GT, WEIGHT, NATIVE, PROFILE, BASE / 'trace/TRACE_MANIFEST.json',
             BASE / 'image_ids.json', BASE / 'ledger/PER_GT_FAILURE.jsonl', BASE / 'ledger/SUMMARY.json',
             BASE / 'SUMMARY.json', BASE / 'EVAL_CONTRACT.json', TRAIN / 'last.pt',
             TRAIN / 'EVAL_CONTRACT.json', TRAIN / 'CONFIG_AUDIT.json', TRAIN / 'SUMMARY.json',
             PARENT / 'EVAL_CONTRACT.json', PARENT / 'CONFIG_AUDIT.json',
             GATE / 'MIXED_predictions.json', GATE / 'ROW_MAPPING.json', GATE / 'EVALUATION_IMAGES.json',
             GATE / 'EVAL_CONTRACT.json', GATE / 'CONFIG_AUDIT.json', GATE / 'SUMMARY.json', GATE / 'DECISION.json']
    for parent in (BASE, PARENT, TRAIN, GATE):
        files.append(parent / 'run.json')
    digest = {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
    profile = read_json(PROFILE)
    require(digest[GT.relative_to(ROOT).as_posix()] == profile['annotations_sha256'], 'GT changed')
    require(digest[WEIGHT.relative_to(ROOT).as_posix()] == profile['model_sha256'], 'weight changed')
    require(sha256(TRAIN / 'last.pt') == '15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce', 'checkpoint changed')
    require(sha256(NATIVE) == trace['predictions_sha256'], 'native predictions changed')
    nbytes = 0
    for i, record in enumerate(trace['images']):
        cache = BASE / 'trace' / record['cache']
        require(sha256(cache) == record['cache_sha256'], 'cache changed: ' + str(cache))
        nbytes += cache.stat().st_size
        image = ROOT / f"assets/datasets/coco_native_s0_20261013/images/val2017/{record['image_id']:012d}.jpg"
        require(sha256(image) == record['image_sha256'], 'image identity changed: ' + str(image))
        if (i + 1) % 1000 == 0:
            print('VERIFIED_CACHE_AND_IMAGE', i + 1, flush=True)
    ids = sorted(trace['image_ids'])
    old_ids = set(read_json(GATE / 'EVALUATION_IMAGES.json'))
    smoke_ids = [r['image_id'] for r in trace['images'] if r['image_id'] in old_ids][:8]
    native_records = {r['image_id']: r for r in trace['images']}
    # Include an empty-output image if one exists, without choosing by method outcome.
    empty = sorted(i for i, r in native_records.items() if not r['prediction_indices'])
    if empty:
        smoke_ids[-1] = empty[0]
    write_new(STUDY / 'INPUT_EXPECTATIONS.json', dict(files=digest, image_ids=ids, smoke_ids=smoke_ids,
              cache_file_count=5000, cache_bytes=nbytes, predictions=539128,
              caches_bound_by_trace_manifest=True, source_images_rehashed=5000))
    import psutil
    write_new(STUDY / 'PREPARATION.json', dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
              host=platform.node(), pid=os.getpid(), input_files=len(files), cache_files=5000,
              prediction_rows=sum(len(r['prediction_indices']) for r in trace['images']),
              images_rehashed=5000, cache_bytes=nbytes, empty_output_image_ids=empty,
              free_memory_bytes=psutil.virtual_memory().available, disk_free_bytes=shutil.disk_usage(ROOT).free,
              elapsed_seconds=time.monotonic()-started, base_model_forward_calls=0, new_training_steps=0))


if __name__ == '__main__':
    main()
