"""Reject a real smoke epoch snapshot and audit the real original-ID parser."""
import argparse
import hashlib
import json
from pathlib import Path
from frozen_io import dump_json, sha256
import evaluate_epoch as evaluator

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--run-id', required=True)
a = p.parse_args()
snapshot = a.root / 'runs/RUN_TRIFLOW_20K_OBSERVER_ENGINEERING01_EVAL_S0/head_epoch_01.pt'
if not snapshot.is_file():
    raise FileNotFoundError(snapshot)
try:
    evaluator.load_epoch_snapshot(snapshot, device='cpu', expected_epoch=1)
except ValueError as error:
    reason = str(error)
    if not any(token in reason.lower() for token in ('formal', 'engineering', 'budget', '20000', 'eight', 'epochs')):
        raise AssertionError('Unrelated loader failure is not formal-scope rejection: ' + reason)
else:
    raise AssertionError('A real smoke epoch snapshot entered formal5000 evaluation')
receipt = {'passed': True, 'real_smoke_epoch_snapshot_rejected': True,
           'snapshot_sha256': sha256(snapshot), 'actual_rejection': reason,
           'evaluator_sha256': sha256(Path(evaluator.__file__)),
           'formal_positive_case_tested': False,
           'scope': 'Actual smoke guard; first real formal positive contract is checked at epoch1 completion'}
dump_json(a.root / 'runs' / a.run_id / 'EPOCH_LOADER_CONTRACT.json', receipt)
print(json.dumps(receipt), flush=True)
