"""Paired, image-clustered descriptive analysis of the frozen diagnostic cohort."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(x) for x in (a.input / 'instances.jsonl').read_text().splitlines()]
    assert len(rows) == 144 and len({r['annotation_id'] for r in rows}) == 144
    modes = ['scalar', 'coeff', 'coeff_bias', 'local4', 'coeff_local4']
    rng = np.random.default_rng(20260916)

    def group(rs):
        ids = sorted({r['image_id'] for r in rs})
        lookup = {v: i for i, v in enumerate(ids)}
        clusters = np.array([lookup[r['image_id']] for r in rs])
        draws = rng.multinomial(len(ids), [1 / len(ids)] * len(ids), size=2000)
        weights = draws[:, clusters]
        denom = weights.sum(1)

        def paired(left, right, stage):
            def value(r, m):
                return r['baseline']['iou'] if m == 'baseline' else r['arms'][m][stage]['iou']
            diff = np.array([value(r, left) - value(r, right) for r in rs]) * 100
            boot = weights @ diff / denom
            return {'delta_iou_points': float(diff.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
                    'improved': int((diff > 1e-9).sum()), 'worsened': int((diff < -1e-9).sum())}

        return {'n': len(rs), 'images': len(ids),
                'contrasts': {stage: {f'{left}_minus_{right}': paired(left, right, stage)
                    for left, right in [(m, 'baseline') for m in modes] +
                    [('coeff', 'scalar'), ('local4', 'coeff'), ('coeff_local4', 'coeff_bias')]}
                    for stage in ['last', 'best']},
                'optimization': {m: {
                    'best_at_final_step': sum(r['arms'][m]['best_step'] == 120 for r in rs),
                    'mean_loss_change_last_20_steps': float(np.mean([
                        r['arms'][m]['trajectory'][-1]['loss'] - r['arms'][m]['trajectory'][-3]['loss'] for r in rs])),
                    'mean_iou_change_last_20_steps': float(np.mean([
                        r['arms'][m]['trajectory'][-1]['iou'] - r['arms'][m]['trajectory'][-3]['iou'] for r in rs])),
                    'coverage_delta_points': 100 * float(np.mean([
                        r['arms'][m]['last']['coverage'] - r['baseline']['coverage'] for r in rs])),
                    'purity_delta_points': 100 * float(np.mean([
                        r['arms'][m]['last']['purity'] - r['baseline']['purity'] for r in rs]))}
                    for m in modes}}

    result = {'all': group(rows), 'groups': {outcome: group([r for r in rows if r['outcome'] == outcome])
              for outcome in ['failure', 'success']},
              'size_fill_failure': {f'{size}_{fill}': group([r for r in rows if r['area_group'] == size
                  and r['fill_group'] == fill and r['outcome'] == 'failure'])
                  for size in ['small', 'medium', 'large'] for fill in ['low', 'high']},
              'interpretation': 'Exploratory conditional-cohort image bootstrap; not population COCO AP or method generalization.'}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'ANALYSIS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['groups']['failure'], ensure_ascii=False))


if __name__ == '__main__':
    main()
