"""Descriptive smoke metrics; GT is used only after predictions are saved."""
import collections
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from pycocotools import mask as mask_api
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"sources"))
from topology import Context


def boxes_iou(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    intersection = np.maximum(np.minimum(a[:,None,2:], b[None,:,2:])-np.maximum(a[:,None,:2], b[None,:,:2]), 0).prod(2)
    area_a = np.maximum(a[:,2:]-a[:,:2], 0).prod(1)
    area_b = np.maximum(b[:,2:]-b[:,:2], 0).prod(1)
    return intersection/np.maximum(area_a[:,None]+area_b[None,:]-intersection, 1e-12)


def associate(gt, preds):
    pb = [p["bbox"][:2]+[p["bbox"][0]+p["bbox"][2],p["bbox"][1]+p["bbox"][3]] for p in preds]
    iou = boxes_iou([g["bbox"] for g in gt], pb)
    if not iou.size:
        return {}
    valid = iou >= .5
    # Maximise valid match cardinality before overlap, never use GT class/mask.
    score = valid*(max(iou.shape)+1+iou)
    rows, cols = linear_sum_assignment(-score)
    return {int(r):int(c) for r,c in zip(rows,cols) if valid[r,c]}

