"""Prediction-only direct and pixel-cost-decomposed calibration of one fixed action."""
from pathlib import Path
import json
import numpy as np
from portable_risk import PortableRisk


def bank_features(rows):
    a = np.asarray([float(r['baseline_area']) for r in rows])
    return np.column_stack([
        np.log1p(a), np.log1p([float(r['mask_elongation']) for r in rows]),
        [float(r['mask_extent']) for r in rows],
        np.log1p([float(r['grid_compactness']) for r in rows]),
        [float(r['removed_fraction']) for r in rows]]).astype(np.float64)


def decompose(purity, coverage, removed_purity, removed_fraction):
    p = np.clip(purity, 1e-6, 1.)
    c = np.clip(coverage, 1e-6, 1.)
    z = np.clip(removed_purity, 0., 1.)
    r = np.clip(removed_fraction, 0., 1.)
    # Enforce nested-mask feasibility in units of original predicted area.
    u = np.clip(z * r, np.maximum(0., p + r - 1.), np.minimum(p, r))
    v = r - u
    g = p / c
    before = p / (g + 1. - p)
    after = (p - u) / (g + 1. - r - p + u)
    return {'gain': after - before, 'coverage_cost': u / g,
            'removed_tp_per_gt': u / g, 'removed_fp_per_gt': v / g,
            'baseline_iou': before, 'purity': p, 'coverage': c, 'removed_purity': z}


class Calibrator:
    def __init__(self, model_dir, frozen_rcmc):
        root = Path(model_dir)
        self.models = {name: PortableRisk(root / (name + '.json'))
                       for name in ['direct', 'purity', 'coverage', 'removed_purity']}
        self.rcmc = PortableRisk(frozen_rcmc)

    def predict(self, features):
        x = np.asarray(features, dtype=np.float64)
        arrays = {name: [] for name in [*self.models, 'frozen_rcmc']}
        for start in range(0, len(x), 2048):
            chunk = x[start:start + 2048]
            for name, model in self.models.items():
                arrays[name].append(model.predict(chunk))
            arrays['frozen_rcmc'].append(self.rcmc.predict(chunk))
        arrays = {k: np.concatenate(v) if v else np.empty(0) for k, v in arrays.items()}
        result = decompose(arrays['purity'], arrays['coverage'], arrays['removed_purity'], x[:, 4])
        result.update(direct=arrays['direct'], frozen_rcmc=arrays['frozen_rcmc'])
        return result


def decisions(scores, eligible, cutoff, protected=False, budget=.02):
    use = eligible & (scores['gain'] > cutoff)
    if protected:
        use &= scores['coverage_cost'] <= budget
    return use


def export_hgb(model, path):
    trees = [stage[0].nodes for stage in model._predictors]
    assert all(len(stage) == 1 for stage in model._predictors)
    assert all(not t['is_categorical'].any() for t in trees)
    fields = {'value': 'value', 'feature': 'feature_idx', 'threshold': 'num_threshold',
              'missing_left': 'missing_go_to_left', 'left': 'left', 'right': 'right', 'leaf': 'is_leaf'}
    size = max(len(t) for t in trees)
    nodes = {key: [np.pad(t[field], (0, size - len(t))).tolist() for t in trees]
             for key, field in fields.items()}
    spec = dict(format='numeric_hgb_v1', features=int(model.n_features_in_),
                baseline=float(model._baseline_prediction[0, 0]),
                max_depth=int(max(t['depth'].max() for t in trees)), nodes=nodes)
    Path(path).write_text(json.dumps(spec, allow_nan=False), encoding='utf-8')
    return {'trees': len(trees), 'leaves': int(sum(t['is_leaf'].sum() for t in trees)),
            'nodes': sum(len(t) for t in trees), 'json_bytes': Path(path).stat().st_size}


def make_predictor(model_dir, frozen_rcmc, selection, variant, chunk=24):
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops
    import torch
    from mask_calibration import input_logits, export_masks, apply_threshold
    from risk_calibration import prediction_features

    calibrator = Calibrator(model_dir, frozen_rcmc)
    policies = json.loads(Path(selection).read_text())['policies']

    class Calibrated(SegmentationPredictor):
        def construct_result(self, pred, img, orig_img, img_path, proto):
            if self.args.retina_masks:
                raise ValueError('Only the frozen non-retina decoder is evaluated')
            batches = []
            for offset in range(0, len(pred), chunk):
                rows = pred[offset:offset + chunk]
                logits = input_logits(proto, rows[:, 6:], rows[:, :4], img.shape[2:])
                base = (logits > 0).byte()
                exported = export_masks(base, orig_img.shape[:2])
                trial = apply_threshold(logits, exported.sum((1, 2)).float(), 'smooth')
                trial_export = export_masks(trial, orig_img.shape[:2])
                x = prediction_features(base, exported, trial_export, 'response')
                eligible = trial_export.flatten(1).any(1).cpu().numpy().astype(bool)
                # Run only the estimator needed by this deployed variant.
                if variant == 'frozen_rcmc':
                    use = (calibrator.rcmc.predict(x) > 0) & eligible
                elif variant == 'direct':
                    use = (calibrator.models['direct'].predict(x) > policies[variant]['cutoff']) & eligible
                else:
                    parts = {k: calibrator.models[k].predict(x) for k in ['purity', 'coverage', 'removed_purity']}
                    scores = decompose(parts['purity'], parts['coverage'], parts['removed_purity'], x[:, 4])
                    use = decisions(scores, eligible, policies[variant]['cutoff'],
                                    protected=variant == 'decomposed_protected', budget=.02)
                use = torch.as_tensor(use, device=base.device)
                batches.append(torch.where(use[:, None, None], trial, base))
            masks = torch.cat(batches) if batches else None
            pred = pred[:, :6].clone()
            if len(pred):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                keep = masks.amax((-2, -1)) > 0
                pred, masks = pred[keep], masks[keep]
            return Results(orig_img, path=img_path, names=self.model.names, boxes=pred, masks=masks)
    return Calibrated
