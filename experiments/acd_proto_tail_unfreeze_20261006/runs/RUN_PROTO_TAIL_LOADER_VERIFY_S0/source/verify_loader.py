"""Synthetic loader-contract checks; the fixture is not a trained epoch-3 model."""
import copy
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / 'RUN_PROTO_TAIL_LOADER_VERIFY_S0'
VENDOR = Path('D:/coco_wire/vendor_8.4.100')
WEIGHT = Path('D:/coco_wire/models/yolo26m-seg.pt')
sys.path.insert(0, str(VENDOR))
os.environ.update(YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
import torch
from ultralytics import YOLO

RUN.mkdir(parents=True, exist_ok=True)
assert not (RUN / 'LOADER_VERIFICATION.json').exists(), 'Use a new Run for a retry'
source = RUN / 'source'
source.mkdir(exist_ok=True)
for name in ('evaluate.py', 'verify_loader.py'):
    (source / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
(source / 'PROTOCOL.md').write_bytes((ROOT / 'PROTOCOL.md').read_bytes())
sys.path.insert(0, str(source))
import evaluate as ev

sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
now = lambda: datetime.now(timezone.utc).isoformat()
meta = {'run_id': RUN.name, 'status': 'running', 'execution_status': 'running',
        'artifact_status': 'partial', 'transfer_status': 'not_started', 'started_at': now(),
        'purpose': 'Synthetic overlay-loader contract verification, not training or AP',
        'interpreter': sys.executable, 'remote_directory': str(RUN), 'environment': 'laptop_28358lan',
        'local_directory': 'C:/Dpan/codexproject/paper-disc-organized/experiments/acd_proto_tail_unfreeze_20261006/runs/' + RUN.name}
(RUN / 'run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
(RUN / 'SOURCE.json').write_text(json.dumps({'script_sha256': sha(source / 'verify_loader.py'),
    'executed_snapshot': str(source / 'verify_loader.py'), 'evaluator_sha256': sha(source / 'evaluate.py'),
    'protocol_sha256': sha(source / 'PROTOCOL.md'), 'captured_at': now()}, indent=2), encoding='utf-8')
receipt = {'status': 'FAIL', 'evaluator_sha256': sha(source / 'evaluate.py'),
           'scope': 'Synthetic contract: epoch=3 is a schema fixture value, not a claim of training',
           'training_claim': False, 'checks': {}}
try:
    torch.set_num_threads(4)
    base, _ = ev.load_model(str(WEIGHT), YOLO, torch)
    expected, mutable, bn = ev.overlay_state(base, torch)
    initial_frozen = ev.frozen_digest(base, torch)[0]
    state = {key: base.state_dict()[key].detach().cpu().clone() for key in expected}
    key = sorted(mutable)[0]
    state[key].reshape(-1)[0] += .001
    payload = {'state_dict': state, 'kind': ev.OVERLAY_KIND, 'epoch': 3, 'audit_passed': True,
               'base_weights_sha256': sha(WEIGHT), 'scope': list(ev.OVERLAY_PREFIXES),
               'synthetic_contract_fixture': True}
    fixture = RUN / 'synthetic_loader_fixture.pt'
    def load(value):
        torch.save(value, fixture)
        return ev.load_model({'base_weights': str(WEIGHT), 'coefficients': str(fixture)}, YOLO, torch)
    model, metadata = load(payload)
    assert metadata['reconstructed_mask_tail_overlay']
    assert set(metadata['overlay_state_keys']) == expected
    assert ev.frozen_digest(model, torch)[0] == initial_frozen
    assert torch.equal(model.state_dict()[key], state[key])
    receipt['checks']['valid_full_scope_fp32_overlay'] = True
    cases = {
        'legacy_kind': lambda x: x.update(kind='coefficient_only_ema_final'),
        'wrong_epoch': lambda x: x.update(epoch=1),
        'failed_audit': lambda x: x.update(audit_passed=False),
        'wrong_base_hash': lambda x: x.update(base_weights_sha256='0' * 64),
        'missing_key': lambda x: x['state_dict'].pop(sorted(expected)[0]),
        'extra_frozen_key': lambda x: x['state_dict'].update({'model.23.proto.cv1.conv.weight': base.state_dict()['model.23.proto.cv1.conv.weight'].clone()}),
        'fp16': lambda x: x['state_dict'].update({key: x['state_dict'][key].half()}),
    }
    for name, mutate in cases.items():
        value = copy.deepcopy(payload)
        mutate(value)
        try:
            load(value)
        except ValueError:
            receipt['checks']['reject_' + name] = True
        else:
            raise AssertionError('Loader accepted invalid ' + name)
    for name, changed_key in (('nan', key), ('bn_change', sorted(expected & bn)[0])):
        value = copy.deepcopy(payload)
        value['state_dict'][changed_key].reshape(-1)[0] += float('nan') if name == 'nan' else .001
        try:
            load(value)
        except ValueError:
            receipt['checks']['reject_' + name] = True
        else:
            raise AssertionError('Loader accepted invalid ' + name)
    for changed_key in ('model.23.proto.cv1.conv.weight', 'model.23.proto.semseg.2.weight', sorted(bn)[0]):
        tensor = base.state_dict()[changed_key]
        original = tensor.clone()
        with torch.no_grad():
            tensor.reshape(-1)[0] += 1
        assert ev.frozen_digest(base, torch)[0] != initial_frozen
        with torch.no_grad():
            tensor.copy_(original)
        assert ev.frozen_digest(base, torch)[0] == initial_frozen
    receipt['checks']['digest_detects_other_proto_semantic_and_bn_changes'] = True
    receipt.update(status='PASS', overlay_keys=len(expected), mutable_keys=len(mutable),
                   base_weights_sha256=sha(WEIGHT), completed_at=now())
    code = 0
except BaseException as error:
    receipt.update(exception=str(error), traceback=traceback.format_exc(), completed_at=now())
    code = 1
(RUN / 'LOADER_VERIFICATION.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
meta.update(status='completed' if code == 0 else 'failed', execution_status='completed' if code == 0 else 'failed',
            exit_code=code, ended_at=now(), artifact_status='generated' if code == 0 else 'partial')
(RUN / 'run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print(json.dumps(receipt), flush=True)
sys.exit(code)
