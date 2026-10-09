"""Numeric-only frozen HGB inference; no scikit-learn pickle on the target host."""
import json
from pathlib import Path
import numpy as np


class PortableRisk:
    def __init__(self, path):
        spec = json.loads(Path(path).read_text(encoding='utf-8'))
        if spec['format'] != 'numeric_hgb_v1':
            raise ValueError('Unknown format')
        self.baseline = spec['baseline']
        self.features = spec['features']
        self.depth = spec['max_depth']
        self.arrays = {k: np.asarray(v, dtype=np.float64 if k in ('value', 'threshold') else np.int64)
                       for k, v in spec['nodes'].items()}

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.features:
            raise ValueError('Feature shape mismatch')
        a = self.arrays
        tree = np.arange(len(a['value']))[:, None]
        node = np.zeros((len(tree), len(x)), dtype=np.int64)
        row = np.arange(len(x))[None, :]
        for _ in range(self.depth):
            value = x[row, a['feature'][tree, node]]
            left = np.where(np.isnan(value), a['missing_left'][tree, node],
                            value <= a['threshold'][tree, node])
            next_node = np.where(left, a['left'][tree, node], a['right'][tree, node])
            node = np.where(a['leaf'][tree, node], node, next_node)
        if not a['leaf'][tree, node].all():
            raise RuntimeError('Incomplete traversal')
        result = np.full(len(x), self.baseline)
        # Same accumulation order as sklearn's raw prediction.
        for values in a['value'][tree, node]:
            result += values
        return result


def load_estimator(path):
    if Path(path).suffix == '.json':
        return PortableRisk(path)
    import joblib
    return joblib.load(path)
