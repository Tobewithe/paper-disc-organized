"""Fit a single prespecified seed; no validation data or hyperparameter search."""
from pathlib import Path
import argparse
import json
import time
from common import setup, atomic, progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); p, out = setup(args.protocol, args.output)
    import numpy as np
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from calibrator import export_hgb, decompose
    from portable_risk import PortableRisk
    from data_access import load_bank
    started = time.perf_counter()
    split = json.loads(Path(p['split']).read_text())
    fit_ids, selection_ids = set(split['fit_ids']), set(split['selection_ids'])
    assert len(fit_ids) == 1500 and len(selection_ids) == 500 and not fit_ids & selection_ids
    bank = load_bank(p['train_bank'])
    assert set(bank['ids']) == fit_ids | selection_ids
    selected = [r for r in bank['matched'] if r['image_id'] in fit_ids]
    indices = np.asarray([r['index'] for r in selected]); x = bank['features'][indices]
    target = {'direct': np.asarray([r['trial_iou'] - r['baseline_iou'] for r in selected]),
              'purity': np.asarray([r['baseline_purity'] for r in selected]),
              'coverage': np.asarray([r['baseline_coverage'] for r in selected]),
              'removed_purity': np.asarray([r['removed_purity'] for r in selected])}
    # Exact-label algebra check, including the feasibility constraints and empty fallback.
    effective_r = np.where(bank['eligible'][indices], x[:, 4], 0.)
    exact = decompose(target['purity'], target['coverage'], target['removed_purity'], effective_r)
    positive = (target['purity'] > 1e-6) & (target['coverage'] > 1e-6)
    algebra_error = float(np.max(np.abs(exact['gain'][positive] - target['direct'][positive])))
    coverage_error = float(np.max(np.abs(exact['coverage_cost'][positive]
                        - np.asarray([r['coverage_cost'] for r in selected])[positive])))
    assert algebra_error < 1e-8 and coverage_error < 1e-8
    atomic(out / 'training_definition.json', {'fit_ids': sorted(fit_ids), 'selection_ids': sorted(selection_ids),
        'n_rows': len(selected), 'features': p['feature_names'], 'models': p['models'], 'estimator': p['estimator'],
        'label_equations_max_error': algebra_error, 'coverage_equation_max_error': coverage_error,
        'zero_overlap_rows': int((~positive).sum()), 'trained_from_all_matched_success_and_failure': True})
    summary = {'training_rows': len(selected), 'training_images': len({r['image_id'] for r in selected}),
               'models': {}, 'protocol': p['protocol_id'], 'label_equations_max_error': algebra_error}
    for name, y in target.items():
        progress(out, 'fit', model=name, rows=len(x))
        trees = p['models']['direct']['trees'] if name == 'direct' else p['models']['decomposed']['trees_per_target']
        model = HistGradientBoostingRegressor(max_iter=trees, **p['estimator'])
        t = time.perf_counter(); model.fit(x, y); seconds = time.perf_counter() - t
        joblib.dump(model, out / (name + '.joblib'))
        model_summary = export_hgb(model, out / (name + '.json'))
        portable = PortableRisk(out / (name + '.json'))
        actual = np.concatenate([portable.predict(x[i:i+2048]) for i in range(0, len(x), 2048)])
        error = float(np.max(np.abs(actual - model.predict(x))))
        assert error < 1e-12
        model_summary.update(fit_seconds=seconds, training_mse=float(np.mean((actual-y)**2)),
                             portable_max_abs_error=error)
        summary['models'][name] = model_summary
        atomic(out / 'metrics_partial.json', summary)
    summary.update(elapsed_seconds=time.perf_counter()-started,
                   direct_total_trees=summary['models']['direct']['trees'],
                   decomposed_total_trees=sum(summary['models'][k]['trees'] for k in ['purity','coverage','removed_purity']))
    assert summary['direct_total_trees'] == summary['decomposed_total_trees'] == 300
    atomic(out / 'SUMMARY.json', summary); progress(out, 'completed', **summary)


if __name__ == '__main__': main()
