"""Check a real R3 32-image run against the preserved original reference."""
import argparse
import json
from pathlib import Path
from frozen_io import dump_json, sha256
from train_epoch_observer_r3 import exact_tree_sha

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--run-id', required=True)
args = parser.parse_args()
import torch

observed_run = args.root / 'runs/RUN_TRIFLOW_20K_DIAGNOSTICS_OBSERVER_ENGINEERING_S0'
reference_run = args.root / 'runs/RUN_TRIFLOW_20K_ENGINEERING32_S0_R1'
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
callback = args.root / 'runs/RUN_TRIFLOW_20K_DIAGNOSTICS_OBSERVER_ENGINEERING_S0_EPOCH01_EVAL_S0'
after = json.loads((callback / 'PARENT_STATE_FINAL_PARITY.json').read_text(encoding='utf-8'))
sink = json.loads((callback / 'ENGINEERING_CALLBACK.json').read_text(encoding='utf-8'))
execution = json.loads((observed_run / 'OBSERVER_COMPLETE.json').read_text(encoding='utf-8'))
checks['actual_parent_state_parity_passed'] = after.get('passed') is True
checks['actual_engineering_sink_not_coco_claim'] = sink.get('passed') is True and sink.get('engineering_only') is True
counts = observed.get('execution_diagnostics_mode_counts_this_execution', {})
checks['actual_minimal_and_sampled_full'] = counts.get('full') == 1 and counts.get('minimal') == observed['state']['applied'] - 1
checks['declared_runtime_bound_to_snapshot'] = observed.get('execution_diagnostics_metadata') == execution.get('diagnostics_execution')
result = {'passed': all(checks.values()), 'smoke_only': True, 'engineering_only': True,
          'images': 32, 'epochs': 1, 'actual_epoch_callback_performed': True,
          'formal5000_evaluation_performed': False,
          'observer_source_sha256': sha256(Path(__file__).with_name('train_epoch_observer_r3.py')),
          'baseline_snapshot_sha256': sha256(reference_run / 'checkpoint_latest.pt'),
          'observer_snapshot_sha256': sha256(observed_run / 'checkpoint_latest.pt'),
          'checks': checks, 'diagnostics_mode_counts': counts,
          'final_module_state_sha256': observed['head_state_sha256'],
          'scope': 'Same observed original32-image scientific trajectory and real callback; diagnostic calculations/timing differ'}
dump_json(args.root / 'runs' / args.run_id / 'DIAGNOSTICS_OBSERVER_ENGINEERING_VERIFICATION.json', result)
print(json.dumps({'passed': result['passed'], 'failed_checks': [key for key, value in checks.items() if not value],
                  'actual_optimizer_updates': observed['state']['applied'], 'diagnostics_mode_counts': counts}), flush=True)
raise SystemExit(0 if result['passed'] else 1)
