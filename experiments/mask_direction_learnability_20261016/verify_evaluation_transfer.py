"""Local CLD evaluation transfer, lineage and frozen statistical verification. No model execution."""
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
FIT = ROOT / 'experiments/mask_soft_fit_audit_20261016/runs/RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770'
CACHE = ROOT / 'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, reason):
    if not ok:
        raise AssertionError(reason)


def identity(row):
    return row['image_id'], row['annotation_id']


def macro(rows_to_reduce, names):
    groups = defaultdict(list)
    for row in rows_to_reduce:
        groups[row['image_id']].append(row)
    return {name: math.fsum(math.fsum(r[name] for r in group) / len(group)
                           for group in groups.values()) / len(groups) for name in names}


def main():
    transport = read_json(STUDY / 'EVALUATION_TRANSFER_MANIFEST.json')
    require(transport['run_ids'] == ['RUN_CLD_EVALUATION_3bf05f04bb6e47c9abf9bb46cc4e1f47'], 'wrong actual Run')
    for record in transport['files']:
        item = STUDY / record['path']
        require(item.resolve().is_relative_to(STUDY), 'transfer path escaped Study')
        require(item.stat().st_size == record['size_bytes'] and sha(item) == record['sha256'], record['path'])
    freeze = read_json(STUDY / 'EXECUTION_FREEZE.json')
    for rel, digest in freeze['files'].items():
        require(sha(ROOT / rel) == digest, rel)
    for item in read_json(STUDY / 'SCIENTIFIC_FREEZE.json')['files']:
        require(sha(STUDY / item['path']) == item['sha256'], item['path'])
    for rel, digest in read_json(STUDY / 'EVALUATION_CONTROL_FREEZE.json')['files'].items():
        require(sha(STUDY / rel) == digest, rel)
        require(sha(STUDY / 'evaluation_control_snapshot' / rel.replace('/', '__')) == digest, 'control snapshot')
    run = STUDY / 'runs' / transport['run_ids'][0]
    meta, contract, summary, independent = [read_json(run / name) for name in
                                           ('run.json', 'EVAL_CONTRACT.json', 'SUMMARY.json', 'INDEPENDENT.json')]
    require(meta['execution_status'] == 'completed' and meta['exit_code'] == 0, 'Run not complete')
    require(read_json(run / 'CONFIG_AUDIT.json')['status'] == 'passed' and independent['status'] == 'passed', 'audit failed')
    require(contract['stage'] == 'CLD_evaluation' and contract['new_optimizer_steps'] == meta['optimizer_steps'] == 0, 'wrong stage')
    require(meta['base_model_forward_calls'] == meta['oracle_optimization_calls'] == 0, 'unexpected base/oracle computation')
    require(contract['operation_authorization']['allow_full_evaluation'] is True
            and contract['operation_authorization']['allow_formal_training'] is False, 'wrong authorization')
    require(sha(run / 'EVAL_CONTRACT.json') == meta['contract_sha256'] == independent['contract_sha256'], 'contract SHA')
    require(sha(run / 'PROTOCOL.md') == sha(STUDY / 'PROTOCOL.md'), 'protocol SHA')
    require(read_json(run / 'SOURCE_MANIFEST.json')['files'] == freeze['files'], 'source manifest changed')
    for rel, digest in freeze['files'].items():
        require(sha(run / read_json(run / 'SOURCE_SNAPSHOT_MAP.json')[rel]) == digest, 'source snapshot '+rel)
    for record in read_json(run / 'ARTIFACT_MANIFEST.json')['artifacts']:
        require(sha(run / record['relative_path']) == record['sha256'], 'artifact hash')
    registry = read_json(STUDY / 'TRAINED_CHECKPOINTS.json')
    for arm, item in registry['checkpoints'].items():
        require(contract['checkpoint_runs'][arm] == item['run_id'], 'wrong checkpoint Run')
        require(sha(STUDY / item['relative_path']) == item['sha256'] == contract['checkpoints'][arm], 'checkpoint bytes changed')
    require(contract['amplitude'] == registry['common_amplitude'] == 2.1651711144258443, 'amplitude drift')
    require(contract['environment'] == read_json(STUDY / 'INHERITED_CONTRACT.json')['source_environment'], 'runtime drift')
    held = set(read_json(STUDY / 'SPLIT.json')['heldout_image_ids'])
    direction, pairs, tasks = [rows(run / name) for name in ('DIRECTION_ROWS.jsonl', 'FIXED_PAIRS.jsonl', 'TASKS.jsonl')]
    require(all((r['split'] == 'heldout') == (r['image_id'] in held) for r in direction), 'direction split mismatch')
    require(len(set(identity(r) for r in direction)) == len(direction) == 3091, 'duplicate/missing direction identity')
    require(all(r['image_id'] in held for r in pairs + tasks), 'evaluation denominator escaped heldout')
    expected_pairs = {identity(r): r for r in rows(FIT / 'FIT_FIXED_PAIRS.jsonl') if r['image_id'] in held}
    require(set(expected_pairs) == set(identity(r) for r in pairs) and len(pairs) == 2246, 'fixed pair identities changed')
    for r in pairs:
        old = expected_pairs[identity(r)]
        require((r['N'], r['G'], r['population'], r['output_detection_id']) ==
                (old['baseline_iou'], old['G_iou'], old['population'], old['baseline_output_detection_id']), 'historical pair differs')
    historic = {identity(r): r for r in rows(FIT / 'FIT_TASKS.jsonl') if r['image_id'] in held}
    ledgers = {k: {identity(r): r for r in rows(run / f'{k}_GT_LEDGER.jsonl')} for k in ('N', 'M', 'S')}
    task_ids = {identity(r) for r in tasks}
    require(len(task_ids) == len(tasks) and task_ids == {i for i, r in ledgers['N'].items() if not r['is_ignored']}, 'task identities differ')
    for arm in ('M', 'S'):
        require(set(ledgers[arm]) == set(ledgers['N']), 'ledger GT identities differ')
        for i, n in ledgers['N'].items():
            require(all(ledgers[arm][i][k] == n[k] for k in ('is_ignored', 'task_box75', 'bbox75_match_output_id')), 'Box/ignore guard')
    for r in tasks:
        require(all(r[k] == (ledgers[k][identity(r)]['task_mask75'] == 'success') for k in ('N', 'M', 'S')), 'task and ledger differ')
        require(r['G'] == historic[identity(r)]['G_success'], 'historical G task differs')
    mapping = read_json(run / 'ROW_MAPPING.json')['global_indices']
    trace = read_json(CACHE / 'trace/TRACE_MANIFEST.json')
    expected_mapping = sorted(g for im in trace['images'] if im['image_id'] in held for g in im['prediction_indices'])
    require(mapping == expected_mapping and len(mapping) == len(set(mapping)) == 42167, 'output row mapping differs')
    predictions = {k: read_json(run / f'{k}_predictions.json') for k in ('N', 'M', 'S')}
    for arm in ('M', 'S'):
        require(len(predictions[arm]) == len(predictions['N']) == 42167, 'incomplete mixed outputs')
        for original, changed in zip(predictions['N'], predictions[arm]):
            require({k: v for k, v in original.items() if k != 'segmentation'} ==
                    {k: v for k, v in changed.items() if k != 'segmentation'}, 'non-mask prediction identity changed')
    # Execute only the immutable pure-statistics function; never import torch or a model.
    verifier = STUDY / 'scripts/verify_evaluation.py'
    parsed = ast.parse(verifier.read_text(encoding='utf-8'))
    function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'statistical_check')
    namespace = dict(defaultdict=defaultdict, math=math, np=np, read_json=read_json, require=require)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(verifier), 'exec'), namespace)
    stats = namespace['statistical_check'](run, direction, pairs, tasks)
    require(independent['kind'] == 'full_evaluation' and independent['additional_optimizer_steps'] == 0, 'wrong independent scope')
    require(independent['adapter_forward_calls'] == meta['adapter_forward_calls'] == 2534, 'forward accounting differs')
    require(all(stats[k] == independent[k] for k in stats), 'two-machine scientific decision differs')
    target = [r for r in pairs if r['population'] == 'T']
    result = dict(status='passed', at=datetime.now(timezone.utc).isoformat(), run_id=run.name,
                  verifier_sha256=sha(Path(__file__)), frozen_statistics_source_sha256=sha(verifier),
                  local_environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__),
                  scope='Local full-file hashes, lineage, complete prediction identities, GT ledgers and frozen statistical recomputation; model/RLE reexecution performed independently on laptop',
                  files_verified=len(transport['files']), bytes_verified=sum(r['size_bytes'] for r in transport['files']),
                  checkpoint_hashes=contract['checkpoints'], direction_instances=3091, heldout_direction_instances=581,
                  target_instances=490, overall_pairs=2246, heldout_images=400, complete_output_rows=42167,
                  zero_additional_training=True, laptop_primary_adapter_forwards=2534, laptop_independent_adapter_forwards=2534,
                  statistics=stats, raw_summary_sha256=sha(run / 'SUMMARY.json'),
                  descriptive_absolute_macro_iou=dict(target=macro(target, ('N', 'M', 'S', 'G')), overall=macro(pairs, ('N', 'M', 'S', 'G'))))
    with (STUDY / 'EVALUATION_LOCAL_VERIFICATION.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
