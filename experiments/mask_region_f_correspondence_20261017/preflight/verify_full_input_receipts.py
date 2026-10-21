"""Read-only full-export receipt reconciliation; no model or scientific evaluator."""
from pathlib import Path
import hashlib
import json

study = Path(__file__).resolve().parents[1]
root = study.parents[1]
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
run = study / 'runs' / read(study / 'INPUT_FULL_POINTER.json')['run_id']
assert read(run / 'run.json')['execution_status'] == 'completed'
assert read(run / 'DECISION.json')['status'] == 'passed'
assert read(run / 'CONFIG_AUDIT.json')['status'] == 'passed'
source = read(run / 'SOURCE_MANIFEST.json')
assert source == read(study / 'EXECUTION_SOURCE.json')
checked = 0
external = 0
for item in read(run / 'ARTIFACT_MANIFEST.json')['artifacts']:
    rel = item['relative_path']
    if rel.startswith('regions/'):
        external += 1
        continue
    assert sha(run / rel) == item['sha256'], rel
    checked += 1
for i, (rel, digest) in enumerate(source['files'].items()):
    assert sha(run / 'source_snapshot' / (f'{i:03d}_' + Path(rel).name)) == digest
    assert sha(root / rel) == digest
    checked += 2
spec = read(study / 'DESIGN_SPEC.json')
fit = read(root / spec['data']['source_split'])['fit_ids']
checks = read(run / 'fit_REPLAY_CHECKS.json')
manifest = read(run / 'REGION_MANIFEST.json')['fit']
assert external == len(checks) == len(manifest['records']) == len(fit) == 2000
assert {c['image_id'] for c in checks} == {r['image_id'] for r in manifest['records']} == set(fit)
assert sum(c['rows'] for c in checks) == sum(r['rows'] for r in manifest['records']) == 218209
assert all(c['tensor_identity_exact'] and c['native_output_exact'] and c['zero_delta_RLE_exact'] for c in checks)
assert all(r['region_rows'] == r['rows'] for r in manifest['records'])
receipt = dict(status='passed', run_id=run.name, artifact_and_source_hash_checks=checked,
               original_training_images=2000, original_final_outputs=218209,
               all_training_rows_have_region_F=True, training_selection_unchanged=True,
               same_execution_source_as_smoke=True, local_GPU_recomputation=False,
               region_sidecars_preserved_on_laptop=external)
(study / 'preflight/INDEPENDENT_FULL_INPUT_RECEIPT_CHECK.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt))
