"""Load the delivered FP32 checkpoints and run native 640-canvas pig segmentation."""
import argparse
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.nn.tasks import SegmentationModel

from common import dump, sha
from data import letterbox, rle
from evaluate import predict
from model import attach


def load_checkpoint(checkpoint, device):
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SegmentationModel(deepcopy(data['yaml']), nc=1, verbose=False).to(device).float()
    model.names = {0: 'pig'}
    if data['mode'] in ['spm', 'adapter']:
        attach(model, data['memory'], data['mode'])
    model.load_state_dict(data['state_dict'], strict=True)
    return model.eval(), data


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--source', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--display-confidence', type=float, default=.25)
    args = p.parse_args(); torch.set_num_threads(4)
    raw = cv2.imread(args.source); assert raw is not None
    image, _, rect = letterbox(raw, [])
    tensor = torch.from_numpy(image[:, :, ::-1].transpose(2, 0, 1).copy()).unsqueeze(0).to(args.device).float()/255
    model, data = load_checkpoint(args.checkpoint, args.device)
    with torch.no_grad():
        detections, masks = predict(model, tensor)
    canvas = image.copy(); results = []
    for i, (d, mask) in enumerate(zip(detections, masks)):
        results.append({'output_detection_id': i, 'score': float(d[4]), 'xyxy_canvas': d[:4].tolist(), 'segmentation': rle(mask)})
        if d[4] < args.display_confidence:
            continue
        color = np.random.default_rng(i).integers(65, 256, 3)
        canvas[mask > 0] = (.55*canvas[mask > 0]+.45*color).astype(np.uint8)
        x0, y0, x1, y1 = d[:4].astype(int)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color.tolist(), 1)
        cv2.putText(canvas, f'pig {d[4]:.2f}', (x0, max(12, y0-3)), cv2.FONT_HERSHEY_SIMPLEX, .4, color.tolist(), 1)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); assert cv2.imwrite(str(out), canvas)
    dump(out.with_suffix('.json'), {'checkpoint_sha256': sha(args.checkpoint), 'source_sha256': sha(args.source),
                                  'mode': data['mode'], 'selected_training_step': data['step'], 'canvas': [640, 640],
                                  'valid_rect': rect, 'native_inference_confidence': .001,
                                  'display_confidence': args.display_confidence, 'detections': results})
    print('INFERENCE_SAVED', str(out), 'displayed', int((detections[:, 4] >= args.display_confidence).sum()), flush=True)


if __name__ == '__main__':
    main()
