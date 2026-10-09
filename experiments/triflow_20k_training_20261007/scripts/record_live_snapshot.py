"""Label a partial live metadata return without declaring a running writer complete."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
for identifier in ('RUN_TRIFLOW_20K_PIPELINE_S0', 'RUN_TRIFLOW_20K_TRAIN_S0'):
    run = a.root / 'runs' / identifier
    raw = (run / 'run.json').read_bytes()
    (run / '_remote_run_snapshot.json').write_bytes(raw)
    metadata = json.loads(raw)
    observed = datetime.now(timezone.utc).isoformat()
    metadata.update(transfer_status='partial_metadata_verified', local_directory=str(run.resolve()),
                    snapshot_observed_at=observed, snapshot_scope='live execution metadata only; final artifacts unavailable')
    (run / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    files = sorted(file for file in run.iterdir() if file.is_file() and file.name not in ('transfer.json', 'manifest.sha256'))
    manifest = ''.join(hashlib.sha256(file.read_bytes()).hexdigest() + '  ' + file.name + '\n' for file in files)
    (run / 'manifest.sha256').write_text(manifest, encoding='utf-8')
    transfer = {'run_id': identifier, 'status': 'partial_metadata_verified', 'source_host': '28358lan',
                'source_directory': metadata['remote_directory'], 'local_directory': str(run.resolve()),
                'observed_at': observed, 'files': [file.name for file in files], 'remote_writer_active': True,
                'scope': 'Atomic live metadata and immutable SOURCE fingerprint; no final model or scientific readout returned'}
    (run / 'transfer.json').write_text(json.dumps(transfer, indent=2), encoding='utf-8')
print(json.dumps({'live_metadata_runs_recorded': 2, 'scientific_outcome': None}))
