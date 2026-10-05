"""Compare every prespecified arm; no validation-based model selection."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    for key in ['root', 'learned-reference', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    ids = read(args.root/'execution_ids.json')
    reference = read(args.learned_reference/'RESULTS.json')
    reference_matches = read(args.learned_reference/'MATCHED_GT75.json')
    assert read(args.learned_reference/'run.json')['status'] == 'completed'
    baseline_matches = set(reference_matches['baseline'])
    methods = {name: reference[key] for name, key in
               [('baseline', 'baseline'), ('scalar', 'scalar'),
                ('local16', 'local'), ('learned_template', 'shared2')]}
    matches = {name: set(reference_matches[key]) for name, key in
               [('baseline', 'baseline'), ('scalar', 'scalar'),
                ('local16', 'local'), ('learned_template', 'shared2')]}
    sources = {'reference': str(args.learned_reference)}
    selections = {}
    for arm in ['analytic_center', 'shuffled_learned', 'fit_residual']:
        run = args.root/'runs'/ids[arm+'_eval']
        record = read(run/'run.json')
        assert record['status'] == 'completed' and record['return_code'] == 0
        result = read(run/'RESULTS.json')
        matched = read(run/'MATCHED_GT75.json')
        assert set(matched['baseline']) == baseline_matches
        assert result['baseline']['metrics'] == reference['baseline']['metrics']
        assert result['shared2']['seed'] == 0
        methods[arm] = result['shared2']
        matches[arm] = set(matched['shared2'])
        sources[arm] = {'run_id': record['run_id'], 'path': str(run)}
        selections[arm] = read(args.root/'runs'/ids[arm+'_train']/'SELECTION.json')
    assert len(baseline_matches) == 22767
    summary = {}
    for name, result in methods.items():
        metrics = {k: 100*v for k, v in result['metrics'].items()}
        repairs = len(matches[name]-baseline_matches)
        harms = len(baseline_matches-matches[name])
        assert repairs == result['repaired75'] and harms == result['damaged75']
        summary[name] = {'metrics_0_100': metrics, 'matched75': len(matches[name]),
                         'repairs75': repairs, 'harms75': harms, 'net75': repairs-harms,
                         'alpha': result.get('alpha'),
                         'delta_ap_baseline': metrics['AP']-100*reference['baseline']['metrics']['AP'],
                         'delta_ap_scalar': metrics['AP']-100*reference['scalar']['metrics']['AP'],
                         'delta_ap_learned': metrics['AP']-100*reference['shared2']['metrics']['AP']}
    contrasts = {}
    for arm in ['analytic_center', 'shuffled_learned', 'fit_residual']:
        contrasts[arm+'_minus_learned'] = {
            'delta_metrics_0_100': {k: summary[arm]['metrics_0_100'][k]-summary['learned_template']['metrics_0_100'][k]
                                  for k in summary[arm]['metrics_0_100']},
            'newly_matched75': len(matches[arm]-matches['learned_template']),
            'lost_matched75': len(matches['learned_template']-matches[arm])}
    limits = [
        'Single seed and one fixed permutation; no equivalence or AP significance claim.',
        'Same 800 fit / 200 selection images and repeatedly explored val2017, not independent confirmation.',
        'Learning-template source uses previous local-head outputs on selection images; residual source uses fit images only.',
        'Permutation changes spatial arrangement and possibly interpolated smoothness.',
        'Repair/harm are official COCOeval Mask75 matching identity changes, not fixed-candidate causal pixel attribution.',
        'All arms reported; no automatic model choice, extra seeds, or hyperparameter search.']
    args.out.mkdir(parents=True, exist_ok=True)
    output = {'sources': sources, 'summary': summary, 'contrasts': contrasts,
              'train_selection': selections, 'limits': limits}
    (args.out/'ANALYSIS.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    labels = {'baseline': '冻结基线', 'scalar': '仅偏置', 'local16': '16格局部头',
              'learned_template': '原学习模板', 'analytic_center': '解析中心模板',
              'shuffled_learned': '打乱学习模板', 'fit_residual': '训练残差模板'}
    lines = ['# 模板来源与空间结构：完整 COCO 对照', '',
             '固定 seed0、800张拟合＋200张训练内选择、8轮。全部5000张val2017正常输出，官方COCOeval。', '',
             '| 方法 | AP | AP75 | AP小 | AP中 | AP大 | ΔAP 对基线 | ΔAP 对学习模板 | 修复 | 误伤 | 净增 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summary.items():
        m = row['metrics_0_100']
        lines.append(f"| {labels[name]} | {m['AP']:.4f} | {m['AP75']:.4f} | {m['APS']:.4f} | {m['APM']:.4f} | {m['APL']:.4f} | {row['delta_ap_baseline']:+.4f} | {row['delta_ap_learned']:+.4f} | {row['repairs75']} | {row['harms75']} | {row['net75']:+d} |")
    lines += ['', '修复/误伤相对同一冻结基线；前三个对照和原学习模板复用已完成seed0结果。完整12项指标与逐组差值见 [ANALYSIS.json](ANALYSIS.json)。', '',
              '## 判断边界', '',
              '这是一次预先规定的单种子对照。点估计接近不能称统计等效；打乱组的差异也不能全部归因于语义空间结构。',
              '本表不自动选择获胜方法，不触发增加种子或调参。结果需要结合修复与误伤、模板来源和运行成本解释。', '',
              '原模板利用旧网络输出，训练残差模板利用拟合集标签；不是同数据来源的纯随机来源对照。',
              'val2017已经反复用于探索，本轮不能替代独立确认。']
    (args.out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (args.out/'COMPLETE.json').write_text(json.dumps({'methods': list(summary), 'all_arms_reported': True}), encoding='utf-8')
    print(json.dumps({name: {'AP': row['metrics_0_100']['AP'], 'net75': row['net75']} for name, row in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
