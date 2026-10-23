"""Load the saved native/64-basis model and predict visible instances from RGB only."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from common import dump
from data import rle
from evaluate import predict
from task_model import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cpu')
    args = parser.parse_args(); torch.set_num_threads(4); cv2.setNumThreads(1)
    rgb = cv2.imread(args.image); assert rgb is not None, args.image
    height, width = rgb.shape[:2]; ratio = min(640/height, 640/width)
    h, w = round(height*ratio), round(width*ratio); top, left = (640-h)//2, (640-w)//2
    canvas = np.full((640, 640, 3), 114, np.uint8); canvas[top:top+h, left:left+w] = cv2.resize(rgb, (w, h))
    image = torch.from_numpy(canvas[:, :, ::-1].transpose(2, 0, 1).copy()).to(args.device).float().unsqueeze(0)/255
    model = load(args.checkpoint, device=args.device).eval(); det, masks = predict(model, image)
    output = []
    for d, mask in zip(det, masks):
        original = cv2.resize(mask[top:top+h, left:left+w], (width, height), interpolation=cv2.INTER_NEAREST)
        box = ((d[:4]-np.array([left, top, left, top]))/ratio).tolist()
        box[0], box[2] = np.clip([box[0], box[2]], 0, width); box[1], box[3] = np.clip([box[1], box[3]], 0, height)
        output.append({'score': float(d[4]), 'class_id': int(d[5]), 'box_xyxy': box, 'visible_mask_rle': rle(original)})
    dump(Path(args.output), {'image': str(Path(args.image).resolve()), 'width': width, 'height': height,
                             'checkpoint': str(Path(args.checkpoint).resolve()), 'predictions': output,
                             'GT_required': False, 'model_format': 'spm_v4_1_task_mask_bases_fp32'})


if __name__ == '__main__':
    main()
