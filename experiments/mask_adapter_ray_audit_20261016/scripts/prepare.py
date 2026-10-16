"""Freeze source bytes and original panel/cache identity, without new scientific scores."""
from ray_common import *
import argparse,psutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--test-report',required=True);args=p.parse_args()
    ownership_check();cfg,expected,profile,environment=inspect_inputs()
    tests=read_json(args.test_report);require(tests['status']=='passed','tests failed')
    for rel,digest in tests['tested_sources'].items():require(sha256(ROOT/rel)==digest,'tested source changed')
    previous=read_json(PREVIOUS/'PREPARATION.json')
    panel=read_json(S1/'TARGET_PANEL.json')['targets'];smoke=previous['smoke_targets']
    require(panel==previous['full_targets'],'original panel changed')
    require(len(panel)==256 and len({r['image_id'] for r in panel})==243 and len(smoke)==4,'panel denominator')
    require(not ({r['image_id'] for r in panel}&{r['image_id'] for r in smoke}),'smoke overlaps panel')
    records=previous['verified_cache_and_image_records']
    for row in records:
        require(sha256(BASE/'trace'/row['cache'])==row['cache_sha256'],'cache hash')
        image=ROOT/'assets/datasets/coco_native_s0_20261013/images/val2017'/f"{row['image_id']:012d}.jpg"
        require(sha256(image)==row['image_sha256'],'image hash')
    available,free=psutil.virtual_memory().available,shutil.disk_usage(ROOT).free
    require(available>=3*1024**3 and free>=4*1024**3,'insufficient resources')
    write_new(STUDY/'PREPARATION.json',dict(status='passed',at=now(),environment=environment,full_targets=panel,smoke_targets=smoke,
        verified_cache_and_image_records=records,available_ram_bytes=available,free_disk_bytes=free,engineering_test_report=args.test_report,
        engineering_test_sha256=sha256(args.test_report),new_training_steps=0,base_model_forward_calls=0,new_ray_outcomes_computed=False))
    frozen=dict(frozen_at=now(),files={rel:sha256(ROOT/rel) for rel in CODE},preparation_sha256=sha256(STUDY/'PREPARATION.json'),input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'))
    write_new(STUDY/'FREEZE_MANIFEST.json',frozen)
    study=read_json(STUDY/'study.json');study.update(plan_status='frozen',execution_status='not_started',artifact_status='frozen_sources_and_inputs',
        frozen_at=frozen['frozen_at'],protocol_sha256=sha256(STUDY/'PROTOCOL.md'),current_phase='smoke_ready')
    save(STUDY/'study.json',study)
    print(json.dumps(dict(status='frozen',source_files=len(CODE),input_files=len(expected['files']),images=len(records),protocol_sha256=study['protocol_sha256'])))

if __name__=='__main__':main()
