"""Sequential three-arm seed0 ablation. Each actual execution uses the study runner.

All paths must be supplied on the executing machine. This file has no credentials.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    parser = argparse.ArgumentParser()
    for key in ['root', 'runner', 'bank', 'split', 'learned-template', 'data', 'weights', 'reference', 'scalar-reference']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    study = json.loads((root/'study.json').read_text())['study_id']
    assert study.startswith('STUDY_')
    for key in ['runner', 'bank', 'split', 'learned_template', 'weights']:
        assert getattr(args, key).is_file(), key
    for key in ['data', 'reference', 'scalar_reference']:
        assert getattr(args, key).is_dir(), key
    (root/'runs').mkdir(exist_ok=True)
    ids_path = root/'execution_ids.json'
    arms = ['analytic_center', 'shuffled_learned', 'fit_residual']
    names = ['template_prepare']+[arm+'_'+phase for arm in arms for phase in ['geometry', 'train', 'eval']]
    if ids_path.exists():
        ids = json.loads(ids_path.read_text())
        assert set(ids) == set(names)
    else:
        ids = {name: 'RUN_'+uuid.uuid4().hex for name in names}
        ids_path.write_text(json.dumps(ids, indent=2))
    scripts = root/'scripts'
    snapshots = [root/'PROTOCOL.json', *sorted(scripts.glob('*.py'))]
    status_path = root/'queue_status.json'

    def execute(name, script, options, inputs, outputs):
        out = root/'runs'/ids[name]
        record_path = out/'run.json'
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            if previous.get('status') == 'completed' and all((out/x).exists() for x in outputs):
                return out
            raise RuntimeError(f'{name} already has a non-complete record; preserve it, use a separate retry Run.')
        command = [sys.executable, str(args.runner), '--study', study, '--run-id', ids[name],
                   '--output', str(out), '--cwd', str(root)]
        for path in inputs:
            command += ['--input', str(path)]
        for path in snapshots:
            command += ['--snapshot', str(path)]
        for output in outputs:
            command += ['--expect', str(out/output)]
        command += ['--', sys.executable, '-u', str(scripts/script), *map(str, options), '--out', str(out)]
        status_path.write_text(json.dumps({'status': 'running', 'stage': name, 'run_id': ids[name]}))
        subprocess.run(command, check=True)
        record = json.loads(record_path.read_text())
        assert record['status'] == 'completed' and record['return_code'] == 0
        return out

    try:
        template_dir = execute('template_prepare', 'prepare_template_sources.py',
            ['--bank', args.bank, '--split', args.split, '--learned-template', args.learned_template],
            [args.bank, args.split, args.learned_template], ['COMPLETE.json'])
        evaluated = {}
        for arm in arms:
            template = template_dir/(arm+'.json')
            common = ['--bank', args.bank, '--split', args.split, '--template', template]
            geometry = execute(arm+'_geometry', 'shared_shape_head.py',
                ['prepare', *common, '--weights', args.weights, '--data', args.data],
                [args.bank, args.split, template, args.weights], ['geometry.pt', 'COMPLETE.json'])
            train = execute(arm+'_train', 'shared_shape_head.py',
                ['train', *common, '--geometry', geometry/'geometry.pt', '--seed', '0'],
                [args.bank, args.split, template, geometry/'geometry.pt'], ['epoch8.pt', 'SELECTION.json', 'COMPLETE.json'])
            evaluation = execute(arm+'_eval', 'evaluate_shared2.py',
                ['--train', train, '--template', template, '--data', args.data, '--weights', args.weights,
                 '--reference', args.reference, '--scalar-reference', args.scalar_reference],
                [train/'epoch8.pt', train/'SELECTION.json', template, args.weights,
                 args.reference/'RESULTS.json', args.reference/'MATCHED_GT75.json',
                 args.scalar_reference/'RESULTS.json', args.scalar_reference/'MATCHED_GT75.json'],
                ['RESULTS.json', 'MATCHED_GT75.json', 'COMPLETE.json'])
            evaluated[arm] = str(evaluation)
        status_path.write_text(json.dumps({'status': 'completed', 'evaluations': evaluated,
            'note': 'Execution complete only. Scientific comparison remains to be analyzed; no automatic more seeds.'}, indent=2))
    except Exception as exc:
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        status_path.write_text(json.dumps({**state, 'status': 'failed', 'error': repr(exc)}, indent=2))
        raise


if __name__ == '__main__':
    main()
