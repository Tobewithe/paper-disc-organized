"""Transfer packaging only. Does not change models, metrics or decisions."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import tarfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rt import dump, read, sha, utc


p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
a = p.parse_args(); root = a.root.resolve()
source = root/'runs'/read(root/'FORMAL_POINTER.json')['run_id']
accept = root/'runs'/read(root/'ACCEPTANCE_POINTER.json')['run_id']
assert read(accept/'SUMMARY.json')['independent_acceptance_passed']
bank = read(root/'BANK_POINTER.json')
artifacts = root/'artifacts'; artifacts.mkdir(exist_ok=True)
targets = {'structured_last.pt': source/'structured_last.pt',
           'scrambled_control_last.pt': source/'scrambled_control_last.pt',
           'object_bank.pt': Path(bank['file'])}
for name, path in targets.items():
    target = artifacts/name
    assert not target.exists(), 'Preserve an existing delivery instead of overwrite'
    try: os.link(path, target)
    except OSError: shutil.copy2(path, target)
model_manifest = dict(source_run_id=source.name, source_job_id=read(source/'run.json')['slurm_job_id'],
    files={name: {'sha256': sha(path), 'bytes': path.stat().st_size, 'server_file': str(artifacts/name)} for name, path in targets.items()},
    native_checkpoint=read(root/'CONFIG.json')['native_checkpoint'],
    source_native_checkpoint_sha256=read(root/'INVENTORY.json')['external_files'][str(Path(read(root/'CONFIG.json')['parent_root'])/read(root/'CONFIG.json')['native_checkpoint'])],
    scientific_status='negative real segmentation pilot; research artifacts, keep native model for use',
    complete_body_bank_is_not_visible_GT=True)
dump(artifacts/'MODEL_MANIFEST.json', model_manifest)
model_files = list(targets)+['MODEL_MANIFEST.json']
weights_tar = root/'TRAINED_MODELS.tar.gz'
with tarfile.open(weights_tar, 'w:gz', compresslevel=3) as tar:
    for name in model_files: tar.add(artifacts/name, arcname='artifacts/'+name)
metadata = []
for folder in sorted((root/'runs').iterdir()):
    if not folder.is_dir(): continue
    assert read(folder/'run.json')['execution_status'] in ('completed', 'failed')
    for path in sorted(folder.rglob('*')):
        if not path.is_file(): continue
        if path.suffix not in ('.json', '.py', '.md', '.log', '.png', '.sbatch'): continue
        if path.name.endswith('_predictions.json') or path.name == 'OUTPUT_IDENTITIES.json': continue
        metadata.append(path)
for name in ('BANK_POINTER.json', 'ENGINEERING_POINTER.json', 'FORMAL_POINTER.json',
             'ACCEPTANCE_POINTER.json', 'PRESENTATION_POINTER.json'):
    metadata.append(root/name)
metadata.append(artifacts/'MODEL_MANIFEST.json')
receipt = dict(source='completed Slurm233-236; raw caches/full predictions remain server',
    metadata_files=[dict(path=p.relative_to(root).as_posix(), sha256=sha(p), bytes=p.stat().st_size) for p in metadata],
    model_files=[dict(path='artifacts/'+name, sha256=sha(artifacts/name), bytes=(artifacts/name).stat().st_size) for name in model_files],
    weights_archive={'file': weights_tar.name, 'sha256': sha(weights_tar), 'bytes': weights_tar.stat().st_size})
dump(root/'RESULT_RECEIPT.json', receipt)
with tarfile.open(root/'RESULTS_LIGHT.tar.gz', 'w:gz', compresslevel=3) as tar:
    for path in metadata: tar.add(path, arcname=path.relative_to(root).as_posix())
    tar.add(root/'RESULT_RECEIPT.json', arcname='RESULT_RECEIPT.json')
dump(root/'PACKAGING.json', dict(completed_at=utc(), slurm_job_id=os.getenv('SLURM_JOB_ID'),
    source_sha256=sha(__file__), metadata_files=len(metadata),
    light_archive_sha256=sha(root/'RESULTS_LIGHT.tar.gz'), light_archive_bytes=(root/'RESULTS_LIGHT.tar.gz').stat().st_size,
    weights_archive_sha256=sha(weights_tar), weights_archive_bytes=weights_tar.stat().st_size))
print('PACKAGED', read(root/'PACKAGING.json'), flush=True)
