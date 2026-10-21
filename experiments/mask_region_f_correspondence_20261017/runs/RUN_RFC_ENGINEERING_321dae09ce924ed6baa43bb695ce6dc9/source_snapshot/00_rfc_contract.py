"""Read-only design and configuration checks. This module cannot launch a workload."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from rfc_paths import ROOT, STUDY


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def compare_tree(expected, observed, prefix=''):
    """Report missing/unknown fields as well as type-sensitive value differences."""
    errors = []
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in sorted(set(expected) | set(observed)):
            name = f'{prefix}.{key}' if prefix else key
            if key not in observed:
                errors.append(f'{name}: missing')
            elif key not in expected:
                errors.append(f'{name}: unexpected')
            else:
                errors.extend(compare_tree(expected[key], observed[key], name))
    elif json.dumps(expected, sort_keys=True, allow_nan=False) != json.dumps(observed, sort_keys=True, allow_nan=False):
        errors.append(f'{prefix}: differs')
    return errors


def require_no_drift(expected, observed, label):
    differences = compare_tree(expected, observed)
    if differences:
        raise ValueError(f'{label} drift: ' + '; '.join(differences))


def check_native_args(expected, observed):
    # Output locations may differ. Dataset/model identity is checked by hashes,
    # never inferred from path equality. All remaining actual keys are compared.
    paths = {'model', 'data', 'project', 'name', 'save_dir'}
    require_no_drift({k: v for k, v in expected.items() if k not in paths},
                     {k: v for k, v in observed.items() if k not in paths}, 'native arguments')


def check_native_runtime(expected, observed):
    require_no_drift(expected, observed, 'actual native runtime')


def check_source_references(references=None, root=ROOT):
    references = references or read_json(STUDY/'INPUT_REFERENCES.json')
    files = references['existing_files']
    for item in files:
        path = (root/item['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError('Pinned source changed/missing: ' + item['path'])
    return {'status': 'passed', 'files': len(files)}


def check_training_design(spec=None):
    spec = spec or read_json(STUDY/'DESIGN_SPEC.json')
    old = read_json(ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/runs/'
                    'RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb/EVAL_CONTRACT.json')
    keys = ('seed', 'epochs', 'fit_images', 'image_microbatch', 'gradient_accumulation',
            'effective_image_batch', 'expected_optimizer_steps', 'optimizer', 'amp',
            'dtype', 'scheduler', 'early_stopping', 'checkpoint_selection',
            'augmentation', 'preservation_weight')
    require_no_drift({k: old[k] for k in keys}, {k: spec['training'].get(k) for k in keys}, 'G training')
    require_no_drift(old['environment'], spec['training']['numeric_environment_exact_reference'], 'G numeric contract')
    split = read_json(ROOT/spec['data']['source_split'])
    orders = read_json(ROOT/spec['training']['order_reference'])
    sequences = list(orders.values()) if isinstance(orders, dict) else orders
    if (len(split['fit_ids']) != 2000 or len(split['dev_ids']) != 512 or
            not set(split['fit_ids']).isdisjoint(split['dev_ids']) or len(sequences) != 3 or
            any(len(x) != 2000 or set(x) != set(split['fit_ids']) for x in sequences)):
        raise ValueError('Original split/order mismatch')
    if spec['training']['gate_restricts_training'] or spec['architecture']['total_trainable_parameters'] != 167872:
        raise ValueError('Training scope or parameter budget changed')
    if spec['training']['formal_steps_total'] != 3000 or spec['training']['formal_steps_per_arm'] != 1500:
        raise ValueError('New formal training budget changed')
    return {'status': 'passed', 'fit_images': 2000, 'dev_images': 512, 'steps_per_arm': 1500}


def execution_readiness(study=None):
    """Do not confuse component tests with permission/readiness for real execution."""
    study = study or read_json(STUDY/'study.json')
    missing = []
    for key in ('new_cache_export_authorized', 'training_authorized'):
        if not study.get(key, False):
            missing.append(key)
    for path in ('EXECUTION_FREEZE.json', 'REAL_INPUT_REPLAY.json', 'REAL_SMOKE_VERIFICATION.json'):
        if not (STUDY/path).is_file():
            missing.append(path)
    return {'real_execution_ready': not missing, 'outstanding': missing}
