"""Read-only input/resource preflight, then freeze the new protocol and sources."""
from common import *
import os
import psutil
import socket


def main():
    require(socket.gethostname().upper() == 'YANGLEI', 'authorized laptop only')
    require(not list(ROOT.glob('experiments/*/.*.lock')), 'existing experiment lock')
    require(not list(ROOT.glob('experiments/*/.AUTONOMOUS_OWNER.guard')), 'existing execution owner')
    others = []
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.info['pid'] != os.getpid() and 'python' in (p.info['name'] or '').lower():
            others.append(p.info)
    require(not others, 'other Python task exists: ' + str(others))
    expected, profile, environment = inspect_inputs()
    panel = read_json(S1/'TARGET_PANEL.json')['targets']
    ids = sorted({r['image_id'] for r in panel})
    require(len(panel)==256 and len(ids)==243, 'original S1 denominator changed')
    trace = read_json(BASE/'trace/TRACE_MANIFEST.json')
    records = [r for r in trace['images'] if r['image_id'] in set(ids)]
    require(len(records)==243, 'missing S1 cache')
    for row in records:
        require(sha256(BASE/'trace'/row['cache']) == row['cache_sha256'], 'cache drift')
        image = ROOT / 'assets/datasets/coco_native_s0_20261013/images/val2017' / f"{row['image_id']:012d}.jpg"
        require(sha256(image) == row['image_sha256'], 'original image drift')
    ram = psutil.virtual_memory().available
    disk = shutil.disk_usage(ROOT).free
    require(ram >= 3*1024**3 and disk >= 2*1024**3, 'insufficient resources')
    write_new(STUDY/'PREPARATION.json', dict(status='passed', at=now(), host=socket.gethostname(),
        environment=environment, profile_sha256=sha256(PROFILE), pairs=256, images=243,
        smoke_image_ids=ids[:8], image_ids=ids, verified_cache_and_image_records=records,
        available_ram_bytes=ram, free_disk_bytes=disk, other_python_processes=others,
        base_model_forward_calls=0, new_training_steps=0, outcome_cross_tabulation_performed=False))
    frozen = dict(frozen_at=now(), files={rel:sha256(ROOT/rel) for rel in CODE},
                  input_manifest_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'))
    write_new(STUDY/'FREEZE_MANIFEST.json', frozen)
    study = read_json(STUDY/'study.json')
    study.update(plan_status='frozen', frozen_at=frozen['frozen_at'], protocol_sha256=sha256(STUDY/'PROTOCOL.md'),
                 preparation_status='passed')
    save(STUDY/'study.json', study)
    print(json.dumps(dict(status='frozen', at=frozen['frozen_at'], protocol_sha256=study['protocol_sha256'], files=len(CODE), pairs=256, images=243)))


if __name__ == '__main__':
    main()
