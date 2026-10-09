"""Reject actual engineering/pilot heads, with unchanged native decoder sources."""
import argparse
import ast
import json
from pathlib import Path
from frozen_io import dump_json, sha256
from evaluate_20k import load_triflow_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    a = p.parse_args()
    pilot = Path('D:/coco_wire/experiments/triflow_potential_coefficient_20261006')
    cases = [(a.root / 'runs/RUN_TRIFLOW_20K_ENGINEERING32_S0_R1/head_final.pt',
              'Only fixed formal epoch8, smoke_only=false'),
             (pilot / 'runs/RUN_TRIFLOW_TRAIN_S0_R1/head_final.pt',
              'Expected triflow_20k_module_final checkpoint')]
    rejected = []
    for head, expected in cases:
        if not head.is_file():
            raise FileNotFoundError(head)
        try:
            load_triflow_checkpoint(head, device='cpu')
        except ValueError as e:
            if expected not in str(e):
                raise AssertionError('Unrelated loader error; not evidence of the intended guard: ' + str(e))
            rejected.append({'path': str(head), 'sha256': sha256(head), 'actual_error': str(e),
                             'expected_guard': expected, 'rejected': True})
        else:
            raise AssertionError('A real engineering/pilot head entered the formal fullscale loader')
    def functions(path):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.FunctionDef)}
    new = functions(Path(__file__).with_name('evaluate_20k.py'))
    old = functions(pilot / 'scripts/evaluate_triflow.py')
    names = ['decode', 'detection_records', 'identity_payload', 'assert_same_baseline', 'assemble_predictions']
    # Check every shared non-loader function, so renamed helpers cannot silently
    # evade the native output equivalence check.
    shared = sorted(set(new) & set(old) - {'main', 'load_triflow_checkpoint'})
    if 'decode' not in shared or 'detection_records' not in shared:
        raise AssertionError('Native decoder implementation is unavailable')
    for name in shared:
        if new[name] != old[name]:
            raise AssertionError('Shared native helper changed: ' + name)
    result = {'passed': True, 'actual_heads_rejected': rejected,
              'unchanged_shared_native_helpers': shared,
              'evaluator_sha256': sha256(Path(__file__).with_name('evaluate_20k.py')),
              'formal_positive_case_tested': False,
              'formal_positive_case_reason': 'A complete trained 118287-image eight-epoch head does not exist yet',
              'scope': 'Engineering guard and unchanged decoder source; no new method AP result'}
    dump_json(a.root / 'runs' / a.run_id / 'CONTRACT_VERIFICATION.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
