"""Compare actual observer training to the existing unchanged32-image reference."""
import argparse
import json
from pathlib import Path
from frozen_io import dump_json, sha256
from train_epoch_observer import exact_tree_sha

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--run-id', required=True)
a = p.parse_args()
import torch
observed_run = a.root / 'runs/RUN_TRIFLOW_20K_OBSERVER_ENGINEERING_S0'
reference_run = a.root / 'runs/RUN_TRIFLOW_20K_ENGINEERING32_S0_R1'
observed = torch.load(observed_run / 'checkpoint_latest.pt', map_location='cpu', weights_only=False)
reference = torch.load(reference_run / 'checkpoint_latest.pt', map_location='cpu', weights_only=False)
checks = {}
for key in ('contract', 'head_state_dict', 'optimizer_state_dict', 'initial_head_state_dict', 'rng', 'head_state_sha256'):
    checks[key] = exact_tree_sha(observed[key]) == exact_tree_sha(reference[key])
for key in ('attempts', 'applied', 'instances', 'order', 'cursor', 'task_evidence', 'task_probed_epoch',
            'finite_groups', 'gradient_steps', 'frozen_input_checks', 'initial_head_state_sha256'):
    checks['state/' + key] = exact_tree_sha(observed['state'][key]) == exact_tree_sha(reference['state'][key])
for key in reference['state']['epochs'][0]:
    if key != 'seconds':
        checks['epoch/' + key] = exact_tree_sha(observed['state']['epochs'][0][key]) == exact_tree_sha(reference['state']['epochs'][0][key])
callback = a.root / 'runs/RUN_TRIFLOW_20K_OBSERVER_ENGINEERING01_EVAL_S0'
after = json.loads((callback / 'PARENT_STATE_FINAL_PARITY.json').read_text())
sink = json.loads((callback / 'ENGINEERING_CALLBACK.json').read_text())
checks['actual_parent_state_parity_passed'] = after.get('passed') is True
checks['actual_engineering_sink_not_coco_claim'] = sink.get('passed') is True and sink.get('engineering_only') is True
if not all(checks.values()):
    raise AssertionError('Observer changed scientific behavior: ' + str([key for key, value in checks.items() if not value]))
result = {'passed': True, 'engineering_only': True, 'images': 32, 'epochs': 1,
          'actual_epoch_callback_performed': True, 'formal5000_evaluation_performed': False,
          'observer_source_sha256': sha256(Path(__file__).with_name('train_epoch_observer.py')),
          'baseline_snapshot_sha256': sha256(reference_run / 'checkpoint_latest.pt'),
          'observer_snapshot_sha256': sha256(observed_run / 'checkpoint_latest.pt'),
          'checks': checks, 'final_module_state_sha256': observed['head_state_sha256']}
dump_json(a.root / 'runs' / a.run_id / 'OBSERVER_ENGINEERING_VERIFICATION.json', result)
print(json.dumps(result), flush=True)
