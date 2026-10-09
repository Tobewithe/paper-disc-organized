"""Report actual per-epoch5000 evaluations and the fixed final-epoch readout."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from frozen_io import dump_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--training-run', default='RUN_TRIFLOW_20K_TRAIN_S0_R1')
    a = p.parse_args()
    root = a.root.resolve()
    train = root / 'runs' / a.training_run
    audit = json.loads((train / 'TRAINING_AUDIT.json').read_text())
    complete = json.loads((train / 'TRAINING_COMPLETE.json').read_text())
    if audit.get('passed') is not True or audit.get('completed_epochs') != 8 or audit.get('images_per_epoch') != 20000:
        raise ValueError('Actual final eight-epoch20000-image training audit required')
    if complete.get('head_sha256') != sha256(train / 'head_final.pt'):
        raise ValueError('Actual final head differs from completion receipt')
    rows, evidence = [], {}
    last = None
    for epoch in range(1, 9):
        name = 'RUN_TRIFLOW_20K_EPOCH%02d_EVAL_S0' % epoch
        evaluation = root / 'runs' / name
        verification = root / 'runs' / ('RUN_TRIFLOW_20K_EPOCH%02d_READOUT_VERIFY_S0' % epoch)
        report_path = verification / 'READOUT_VERIFICATION.json'
        verified = json.loads(report_path.read_text())
        summary_path = evaluation / 'SUMMARY.json'
        summary = json.loads(summary_path.read_text())
        if verified.get('passed') is not True or summary.get('status') != 'complete' or summary.get('image_count') != 5000:
            raise ValueError('Actual independently verified complete5000 epoch readout required: ' + str(epoch))
        provenance = summary['head_provenance']
        if provenance.get('epoch') != epoch or provenance.get('kind') != 'triflow_20k_training_snapshot':
            raise ValueError('Actual epoch-boundary snapshot identity differs')
        meta = summary['metrics']
        row = {'epoch': epoch, 'evaluation_run': name,
               'baseline_mask_ap': 100*meta['baseline']['segm']['AP'],
               'mask_ap': 100*meta['triflow']['segm']['AP'],
               'mask_ap75': 100*meta['triflow']['segm']['AP75'],
               'mask_apsmall': 100*meta['triflow']['segm']['APsmall'],
               'box_ap': 100*meta['triflow']['bbox']['AP'],
               'delta_ap_points': 100*(meta['triflow']['segm']['AP']-meta['baseline']['segm']['AP'])}
        rows.append(row)
        evidence[name] = {'summary_sha256': sha256(summary_path), 'verification_sha256': sha256(report_path),
                          'snapshot_sha256': provenance['head_sha256']}
        last = summary
    if last['head_provenance']['loaded_state_sha256'] != audit['final_head_state_sha256']:
        raise ValueError('Epoch8 actual evaluated module differs from original audited final parameters')
    stats = last['paired']['triflow']['statistics']
    damage = stats['damage_rate_of_baseline_success']['value']
    success_delta = stats['baseline_success_mean_iou_delta']['value']
    dm = last['delta_vs_baseline']['triflow']['segm']
    gates = {'mask_ap_points_at_least_0_3': dm['AP_points'] >= .3,
             'ap75_and_apsmall_not_both_negative': not (dm['AP75_points'] < 0 and dm['APsmall_points'] < 0),
             'baseline_success_damage_at_most_1pct': damage <= .01,
             'baseline_success_mean_iou_delta_at_least_minus_0_005': success_delta >= -.005}
    outcome = 'valid_positive' if all(gates.values()) else 'valid_negative'
    decision = {'outcome': outcome, 'primary_epoch': 8, 'epoch8_parameters_equal_actual_final_audit': True,
                'epochs': rows, 'scientific_gates': gates, 'paired_final_epoch': stats,
                'evidence': evidence, 'training_audit_sha256': sha256(train / 'TRAINING_AUDIT.json'),
                'final_checkpoint_sha256': complete['head_sha256'],
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'scope': 'Fixed epoch8 primary result; all epochs are observed validation curves, no best-epoch selection',
                'unknown': ['AP confidence interval', 'training seed uncertainty', 'raw geometry five states', 'AUC', 'crop ceiling']}
    run = root / 'runs' / a.run_id
    dump_json(run / 'EPOCH_CURVE.json', decision)
    lines = ['# TriFlow 2万图逐轮评估', '',
             '每轮均评价原始COCO val2017完整5000图。主结果保持固定第8轮；该轮实际评估参数与最终训练审计逐字节状态哈希一致。', '',
             '| epoch | Mask AP | Δ AP点 | AP75 | APsmall | Box AP |',
             '|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| %d | %.4f | %+.4f | %.4f | %.4f | %.4f |' % (
            row['epoch'], row['mask_ap'], row['delta_ap_points'], row['mask_ap75'], row['mask_apsmall'], row['box_ap']))
    lines += ['', '固定第8轮结论：'+outcome+'；baseline成功损伤率 %.4f%%。' % (100*damage), '',
              '前7轮仅正常任务AP；未重算配对修复/损伤及bootstrap。第8轮包含原配对诊断，允许同类GT重复匹配；该诊断不替代COCO AP或raw几何。', '',
              '2万张为本项目当前配置结论阶段；单seed结果不代表方法能力上限，未计算的指标及不确定性保持未知。']
    report = '\n'.join(lines)+'\n'
    (run / 'REPORT.md').write_text(report, encoding='utf-8')
    (root / 'REPORT.md').write_text(report, encoding='utf-8')
    dump_json(run / 'REPORT_RECEIPT.json', {'outcome': outcome, 'report_sha256': sha256(run / 'REPORT.md'),
                                         'epoch_curve_sha256': sha256(run / 'EPOCH_CURVE.json'), 'primary_epoch': 8})
    print(json.dumps({'outcome': outcome, 'primary_epoch': 8, 'verified_epoch_count': 8}), flush=True)


if __name__ == '__main__':
    main()
