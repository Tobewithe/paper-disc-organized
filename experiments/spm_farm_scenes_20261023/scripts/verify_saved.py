"""Verify/extract the received checkpoint without requiring Blender or Torch."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARCHIVE_SHA = 'ba33fbc5d3ed2b71963e3bdd99d07cfa904d167392fe523866780a83f702df00'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    archive = ROOT/'SPM_FARM_CHECKPOINT_20261023.tar.gz'
    assert sha(archive) == EXPECTED_ARCHIVE_SHA
    target = ROOT/'server_snapshot'; target.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            assert (target/member.name).resolve().is_relative_to(target.resolve())
            assert not member.issym() and not member.islnk()
        tar.extractall(target, filter='data')
    for name in ['runs', 'server_logs']:
        for src in (target/name).rglob('*'):
            if src.is_file():
                dst = ROOT/name/src.relative_to(target/name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    assert sha(dst) == sha(src), f'existing local artifact differs: {dst}'
                else:
                    shutil.copyfile(src, dst)
    for name in ['SMOKE_POINTER.json']:
        shutil.copyfile(target/name, ROOT/name)
    verified = 0; run_checks = []
    for run in sorted((ROOT/'runs').glob('RUN_SPM_FARM_*')):
        manifest = json.loads((run/'ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))
        for name, record in manifest.items():
            path = run/name
            assert path.stat().st_size == record['size_bytes'] and sha(path) == record['sha256'], str(path)
            verified += 1
        meta = json.loads((run/'run.json').read_text(encoding='utf-8'))
        source_dir = {187: ROOT/'revisions/render187', 188: ROOT/'revisions/render188', 189: ROOT}[int(meta['slurm_job_id'])]
        assert sha(source_dir/'FREEZE_MANIFEST.json') == meta['freeze_manifest_sha256']
        assert sha(source_dir/'CONFIG.json') == meta['config_sha256']
        run_checks.append({'run_id': run.name, 'job': int(meta['slurm_job_id']), 'status': meta['execution_status'],
                           'files_verified': len(manifest), 'frozen_source_snapshot': str(source_dir.relative_to(ROOT))})
    parent = ROOT.parent/'spm_prototype_training_20261022'
    old_weights = {'artifacts/pig_prototype_memory.pt': '6c6ca1237b1500eed317b93c47c01ebbba1118fd16cd2c009224d354779c69cd',
                   'artifacts/pig_projector.pt': '4e32f2ca3c1cd8a1998b184a825510a427bd37dc9efa5f68e7377c62034fe939'}
    for name, digest in old_weights.items():
        assert sha(parent/name) == digest, name
    preflight_records = []
    lines = (ROOT/'server_logs/SACCT_SAVE.txt').read_text().splitlines()
    for line in lines[1:]:
        cols = line.split('|')
        if cols[0] not in ['185', '186']:
            continue
        job = int(cols[0]); run = ROOT/'runs'/f'RUN_SPM_ASSET_PRECHECK_JOB{job}'
        run.mkdir(exist_ok=True)
        record = {'study_id': 'STUDY_SPM_FARM_SCENES_20261023', 'run_id': run.name,
                  'stage': 'read_only_asset_preflight', 'slurm_job_id': job,
                  'execution_status': 'completed' if cols[2] == 'COMPLETED' else 'failed',
                  'exit_code': int(cols[3].split(':')[0]), 'started_at_server_reported': cols[4],
                  'ended_at_server_reported': cols[5], 'elapsed': cols[6],
                  'timestamp_source': 'sacct server-reported; no timezone conversion inferred',
                  'metadata_recorded_after_execution': True,
                  'source_sha256': sha(ROOT/'scripts/inspect_asset.py'),
                  'command': 'sbatch scripts/inspect.sbatch',
                  'optimizer_steps': 0, 'scientific_status': 'engineering_inventory_only',
                  'reason': 'completed_asset_inspection' if job == 186 else 'missing_existing_user_runtime_library_search_path'}
        dump(run/'run.json', record)
        shutil.copyfile(ROOT/f'server_logs/spm-farm-inspect-{job}.log', run/'EXECUTION.log')
        if job == 186:
            shutil.copyfile(ROOT/'ASSET_INSPECTION.json', run/'ASSET_INSPECTION.json')
        dump(run/'EVAL_CONTRACT.json', {'task': 'read_registered_asset_rig_without_modification', 'GPU_used': False,
                                        'asset_sha256': '8aec801123f31ca475000c738287dcd075dda6be4c5650c03b5657d801a847ef'})
        dump(run/'ARTIFACT_MANIFEST.json', {p.name: {'sha256': sha(p), 'size_bytes': p.stat().st_size}
                                           for p in run.iterdir() if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json'})
        preflight_records.append(record)
    summary = json.loads((ROOT/'runs/RUN_SPM_FARM_SMOKE_c065813560a6/SUMMARY.json').read_text())
    assert summary['geometry_scenes'] == 8 and summary['RGB_images'] == 16
    assert summary['ray_check_samples'] == 32768 and summary['ray_check_mismatches'] == 0
    receipt = {'saved_date': '2026-10-23', 'verified_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'archive': archive.name, 'archive_sha256': EXPECTED_ARCHIVE_SHA, 'archive_size_bytes': archive.stat().st_size,
               'archive_SHA_matched_server': True, 'render_manifest_files_verified': verified,
               'render_runs': run_checks, 'asset_preflight_records': preflight_records,
               'old_P1_core_weights_SHA_verified': old_weights,
               'new_training_steps': 0, 'formal_generation_started': False,
               'source_revisions_preserved': ['revisions/render187', 'revisions/render188'],
               'scope': 'local/server checkpoint integrity; not a new model-utility evaluation'}
    dump(ROOT/'TRANSFER_RECEIPT.json', receipt)
    print(json.dumps({'archive_bytes': archive.stat().st_size, 'render_files_SHA_verified': verified,
                      'render_runs': run_checks, 'old_P1_weights_verified': 2}, ensure_ascii=False))


if __name__ == '__main__':
    main()
