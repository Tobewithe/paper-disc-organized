"""Explicit references to the original, immutable pilot modules."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'experiments/mask_readout_mechanism_v2_20261012/scripts'
SOFT = ROOT / 'experiments/mask_readout_soft_iou_pilot_20261014/scripts'

for directory in (ORIGINAL, SOFT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def require_original_module(module, filename):
    if Path(module.__file__).resolve() != (ORIGINAL / filename).resolve():
        raise ValueError(f'Unexpected module import: {module.__file__}')
