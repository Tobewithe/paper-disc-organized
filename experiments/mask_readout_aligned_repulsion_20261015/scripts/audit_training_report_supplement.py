"""Post-run supplement; preserve the frozen auditor and its exact report mismatch.

Run only on already completed training. All frozen substantive assertions execute
first. The only permitted discrepancy is a descriptive epoch loss mean whose
laptop value exactly equals math.fsum and whose local value equals local sum.
No scientific metric, gate, optimizer value, or hash receives a tolerance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import audit_training


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def verify(run):
    receipt = run.parent.parent / ('AUDIT_' + run.name + '.json')
    laptop = read(receipt)
    exception = None
    local = None
    try:
        audit_training.audit(run)
    except ValueError as exc:
        if str(exc) != 'prior audit differs':
            raise
        exception = str(exc)
        tb = exc.__traceback__
        while tb:
            if tb.tb_frame.f_code is audit_training.audit.__code__:
                local = canonical(tb.tb_frame.f_locals['report'])
            tb = tb.tb_next
        assert local is not None
    if exception is None:
        local = laptop
    field = 'per_epoch_mean_training_loss'
    assert {k: v for k, v in local.items() if k != field} == {
        k: v for k, v in laptop.items() if k != field
    }, 'a non-descriptive report field differs'
    rows = [json.loads(line) for line in (run / 'training.jsonl').read_text(encoding='utf-8').splitlines()]
    means = {}
    for epoch in range(3):
        values = [row['loss'] for row in rows if row['epoch'] == epoch]
        assert len(values) == 2000
        plain = sum(values) / 2000
        compensated = math.fsum(values) / 2000
        assert local[field][str(epoch)] == plain
        assert laptop[field][str(epoch)] == compensated
        # Standard summation forward-error bound, including final division.
        u = sys.float_info.epsilon / 2
        gamma = (len(values) * u) / (1 - len(values) * u)
        bound = gamma * math.fsum(abs(v) for v in values) / 2000
        difference = abs(plain - compensated)
        assert difference <= bound
        means[str(epoch)] = dict(local_sum_mean=plain, math_fsum_mean=compensated,
            laptop_report_mean=laptop[field][str(epoch)], absolute_difference=difference,
            forward_error_bound=bound, exact_fsum_equality=True)
    return dict(
        status='passed_substantive_checks_with_descriptive_sum_exception' if exception else 'passed',
        run_id=run.name,
        frozen_audit_exception_retained=exception,
        frozen_auditor_sha256=hashlib.sha256(Path(audit_training.__file__).read_bytes()).hexdigest(),
        laptop_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        local_python=sys.version,
        all_frozen_substantive_assertions_passed=True,
        all_other_JSON_report_fields_identical=True,
        scientific_metrics_or_gates_changed=False,
        checkpoint_or_training_artifacts_changed=False,
        per_epoch_descriptive_sum_comparison=means,
        local_actual_report=local,
        scope='Independent checkpoint/log audit. Supplement only explains the final stored-report equality failure; the original auditor and laptop receipt are unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)
