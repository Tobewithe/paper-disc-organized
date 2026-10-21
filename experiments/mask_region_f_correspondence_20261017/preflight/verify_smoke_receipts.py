"""Independent receipt/contract/transfer checks; no inference, training or new metric."""
from pathlib import Path
import hashlib
import json

study = Path(__file__).resolve().parents[1]
root = study.parents[1]
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = read(study / 'EXECUTION_SOURCE.json')
spec = read(study / 'DESIGN_SPEC.json')
checked = 0
external_F = []
runs = {}
for name in ('PREFLIGHT', 'INPUT_SMOKE', 'SMOKE_A', 'SMOKE_Q'):
    run = study / 'runs' / read(study / (name + '_POINTER.json'))['run_id']
    runs[name] = run
    assert read(run / 'run.json')['execution_status'] == 'completed'
    assert read(run / 'DECISION.json')['status'] == 'passed'
    assert read(run / 'CONFIG_AUDIT.json')['status'] == 'passed'
    assert read(run / 'SOURCE_MANIFEST.json') == source
    for item in read(run / 'ARTIFACT_MANIFEST.json')['artifacts']:
        rel = item['relative_path']
        if rel.startswith('regions/'):
            external_F.append(dict(run_id=run.name, **item))
            continue
        assert sha(run / rel) == item['sha256'], (name, rel)
        checked += 1
    for i, (rel, digest) in enumerate(source['files'].items()):
        assert sha(run / 'source_snapshot' / (f'{i:03d}_' + Path(rel).name)) == digest
        assert sha(root / rel) == digest
        checked += 2

input_run = runs['INPUT_SMOKE']
replay_rows = 0
for split in ('fit', 'dev'):
    checks = read(input_run / (split + '_REPLAY_CHECKS.json'))
    assert len(checks) == 128 and len({c['image_id'] for c in checks}) == 128
    assert all(c['tensor_identity_exact'] and c['native_output_exact'] and c['zero_delta_RLE_exact'] for c in checks)
    replay_rows += sum(c['rows'] for c in checks)
    manifest = read(input_run / 'REGION_MANIFEST.json')[split]
    assert len(manifest['records']) == 8
    assert {r['image_id'] for r in manifest['records']} == set(manifest['wanted_ids'])

contracts = []
for name in ('SMOKE_A', 'SMOKE_Q'):
    run = runs[name]
    contract = read(run / 'EVAL_CONTRACT.json')
    summary = read(run / 'SUMMARY.json')
    controls = read(run / 'REAL_FORWARD_CONTROLS.json')
    audit = read(run / 'CONFIG_AUDIT.json')
    assert summary['optimizer_steps'] == 2 and summary['images_seen'] == 8
    assert len(summary['parameters_with_nonzero_gradient']) == 17
    assert summary['initial_state_sha256'] == spec['training']['initial_tensor_state_sha256']
    assert sha(run / 'initial.pt') == spec['training']['initialization_sha256']
    assert sha(run / 'last.pt') == summary['checkpoint_sha256'] == audit['checkpoint_sha256']
    assert contract['optimizer'] == spec['training']['optimizer']
    assert contract['environment'] == spec['training']['numeric_environment_exact_reference']
    assert len(controls['images']) == 16 and controls['layout_effect_images'] > 0
    assert all(c['zero_delta_RLE_exact'] and c['broadcast_old_G_exact'] for c in controls['images'])
    assert controls['extra_optimizer_steps'] == 0 and controls['existing_G_checkpoint_unchanged']
    assert audit['parameters'] == 167872 and audit['trainable_tensors'] == 17
    contracts.append(contract)
for key in ('image_orders_sha256', 'initialization_file_sha256', 'optimizer', 'gt_loss',
            'region_manifest_sha256', 'design_sha256', 'environment', 'trainable_whitelist'):
    assert contracts[0][key] == contracts[1][key], key

receipt = dict(status='passed', check_scope='read_only_smoke_receipt_and_configuration_reconciliation',
               new_inference_or_training=False, artifact_and_source_hash_checks=checked,
               original_batch_context_images=256, original_output_rows_replayed=replay_rows,
               selected_region_images=16, smoke_steps_per_arm=2,
               frozen_source_manifest_sha256=sha(study / 'EXECUTION_SOURCE.json'),
               runs={name: run.name for name, run in runs.items()},
               regional_F_preserved_on_laptop_not_recomputed_locally=external_F)
out = study / 'preflight/INDEPENDENT_SMOKE_RECEIPT_CHECK.json'
out.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in receipt.items() if k != 'regional_F_preserved_on_laptop_not_recomputed_locally'}))
