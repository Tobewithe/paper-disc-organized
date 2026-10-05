"""Independent two-output mask refiner over a fixed training-derived shape."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

import learn_refinement as original
from component_seed_probe import image_correct, setup
from repair_refinement import per_loss, hard_iou


class SharedShapeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(67, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(), nn.Linear(512, 128), nn.SiLU())
        self.output = nn.Linear(128, 2)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(self.features(x)).tanh()


def grid_tensor(path, device='cuda'):
    return torch.tensor(json.loads(Path(path).read_text())['field'], dtype=torch.float32, device=device).reshape(1, 16)


def roi_template(grid):
    t = (torch.arange(32, device=grid.device) + .5) / 32 * 2 - 1
    yy, xx = torch.meshgrid(t, t, indexing='ij')
    coords = torch.stack((xx, yy), -1)[None]
    return F.grid_sample(4 * grid.reshape(1, 1, 4, 4), coords,
        padding_mode='border', align_corners=True)[0, 0]


def full_template(grid, boxes, shape):
    zero = torch.zeros((len(boxes), *shape), dtype=boxes.dtype, device=boxes.device)
    field = image_correct(zero, None, None, boxes, shape, grid.expand(len(boxes), -1), 'local4', 'pixel_center')
    support = ops.crop_mask(torch.ones_like(field), boxes).bool()
    count = support.sum((1, 2)); n = count.clamp_min(1)
    mean = (field * support).sum((1, 2)) / n
    std = (((field - mean[:, None, None]).square() * support).sum((1, 2)) / n).sqrt()
    basis = (field - mean[:, None, None]) / std.clamp_min(1e-12)[:, None, None]
    return basis, mean, std, count


def correct(base, basis, values):
    return base + 4 * (values[:, 0, None, None] + values[:, 1, None, None] * basis)


def bank_indices(bank, split):
    fit_ids, dev_ids = set(split['fit_image_ids']), set(split['selection_image_ids'])
    assert len(fit_ids) == 800 and len(dev_ids) == 200 and not fit_ids & dev_ids
    owners = defaultdict(set)
    for row in bank['records']:
        owners[row['image_id'], row['raw_id']].add(row['annotation_id'])
    fit, dev, seen = [], [], set()
    for index, row in enumerate(bank['records']):
        key = row['image_id'], row['raw_id']
        if len(owners[key]) != 1 or key in seen:
            continue
        seen.add(key)
        assert row['image_id'] in fit_ids | dev_ids
        (fit if row['image_id'] in fit_ids else dev).append(index)
    assert (len(fit), len(dev)) == (14204, 3723)
    return fit, dev


@torch.no_grad()
def prepare(args):
    setup(); torch.backends.cudnn.benchmark = False; start = time.monotonic()
    bank = torch.load(args.bank, map_location='cpu', weights_only=False, mmap=True)
    split = json.loads(args.split.read_text()); fit, dev = bank_indices(bank, split)
    rows_by_image = defaultdict(list)
    for index in fit + dev: rows_by_image[bank['records'][index]['image_id']].append(index)
    means = torch.zeros(len(bank['records'])); stds = torch.ones(len(bank['records']))
    done = torch.zeros(len(bank['records']), dtype=torch.bool)
    model = YOLO(str(args.weights)).model.cuda().float().eval(); head = model.model[-1]; assert head.end2end
    grid = grid_tensor(args.template)
    transform = LetterBox((640, 640), auto=True, stride=32)
    empty = 0; min_std = float('inf')
    for number, image_id in enumerate(sorted(rows_by_image)):
        im = cv2.imread(str(args.data / 'images/train2017' / f'{image_id:012d}.jpg')); assert im is not None
        res = transform.apply_image({'img': im}, transform.get_params({'img': im}))['img']
        inp = torch.from_numpy(np.ascontiguousarray(res[:, :, ::-1].transpose(2, 0, 1))).cuda().float()[None] / 255
        shape = tuple(inp.shape[-2:]); _, raw = model(inp); p = raw['one2one']
        boxes = head._get_decode_boxes(p)[0].T
        indexes = rows_by_image[image_id]
        for first in range(0, len(indexes), 16):
            batch = indexes[first:first+16]
            ids = torch.tensor([bank['records'][i]['raw_id'] for i in batch], device='cuda')
            _, mean, std, count = full_template(grid, boxes[ids], shape)
            empty += int((count == 0).sum()); min_std = min(min_std, float(std.min()))
            means[batch] = mean.cpu(); stds[batch] = std.cpu(); done[batch] = True
        if number % 100 == 0 or number + 1 == len(rows_by_image):
            progress = {'stage': 'geometry', 'images': number+1, 'total': len(rows_by_image), 'elapsed_s': time.monotonic()-start}
            original.save(args.out/'progress.json', progress); print(json.dumps(progress), flush=True)
    assert bool(done[fit+dev].all()) and empty == 0 and min_std > 0
    torch.save({'mean': means, 'std': stds, 'prepared': done, 'records': bank['records'],
        'template': json.loads(args.template.read_text()), 'fit': fit, 'dev': dev}, args.out/'geometry.pt')
    original.save(args.out/'COMPLETE.json', {'prepared': len(fit)+len(dev), 'images': len(rows_by_image),
        'empty_crop': empty, 'min_std': min_std, 'elapsed_s': time.monotonic()-start})


def train(args):
    setup(); start = time.monotonic()
    bank = torch.load(args.bank, map_location='cpu', weights_only=False, mmap=True)
    geometry = torch.load(args.geometry, map_location='cpu', weights_only=False)
    assert geometry['records'] == bank['records']
    assert geometry['template']['field'] == json.loads(args.template.read_text())['field']
    split = json.loads(args.split.read_text()); fit, dev = bank_indices(bank, split)
    assert fit == geometry['fit'] and dev == geometry['dev']
    rt = roi_template(grid_tensor(args.template, 'cpu'))
    norm = torch.stack((geometry['mean'], geometry['std']), 1)
    fitset = TensorDataset(*(bank[k][fit] for k in ['x', 'base', 'target']), norm[fit])
    devset = TensorDataset(*(bank[k][dev] for k in ['x', 'base', 'target']), norm[dev])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    net = SharedShapeHead().cuda(); rt = rt.cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(fitset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    original.save(args.out/'SPLIT.json', split)
    with torch.no_grad():
        x, b, _, nm = next(iter(DataLoader(fitset, batch_size=2)))
        basis = (rt[None] - nm[:, 0, None, None].cuda()) / nm[:, 1, None, None].cuda().clamp_min(1e-12)
        assert torch.equal(correct(b.cuda().float(), basis, net(x.cuda().float())), b.cuda().float())
    history = []
    for epoch in range(1, 9):
        net.train(); losses = []
        for x, b, target, nm in loader:
            x, b, target, nm = [v.cuda().float() for v in (x, b, target, nm)]
            basis = (rt[None] - nm[:, 0, None, None]) / nm[:, 1, None, None].clamp_min(1e-12)
            z = correct(b, basis, net(x)); loss = per_loss(z, target).mean()
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10, error_if_nonfinite=True)
            opt.step(); losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        torch.save({'state_dict': net.state_dict(), 'seed': args.seed, 'epoch': epoch, 'mode': 'shared2',
            'template': geometry['template'], 'parameter_count': sum(p.numel() for p in net.parameters())}, args.out/f'epoch{epoch}.pt')
        progress = {'stage': 'train', 'seed': args.seed, 'epoch': epoch, 'loss': history[-1],
            'grad_norm': float(grad_norm), 'elapsed_s': time.monotonic()-start}
        original.save(args.out/'progress.json', progress); print(json.dumps(progress), flush=True)
    net.eval(); alphas = [0., .25, .5, 1.]
    stats = {str(alpha): {'total': 0., 'n': 0, 'repairs': 0, 'damages': 0, 'loss': 0.} for alpha in alphas}
    with torch.no_grad():
        for x, b, target, nm in DataLoader(devset, batch_size=128):
            x, b, target, nm = [v.cuda().float() for v in (x, b, target, nm)]
            basis = (rt[None] - nm[:, 0, None, None]) / nm[:, 1, None, None].clamp_min(1e-12)
            values = net(x); original_iou = hard_iou(b, target)
            for alpha in alphas:
                z = correct(b, basis, values*alpha); iou = hard_iou(z, target); item = stats[str(alpha)]
                item['total'] += float(iou.sum()); item['n'] += len(x)
                item['repairs'] += int(((iou >= .75) & (original_iou < .75)).sum())
                item['damages'] += int(((iou < .75) & (original_iou >= .75)).sum())
                item['loss'] += float(per_loss(z, target).sum())
    for item in stats.values(): item.update(mean_iou=item['total']/item['n'], mean_loss=item['loss']/item['n'])
    chosen = max(alphas, key=lambda alpha: (stats[str(alpha)]['mean_iou'], -alpha))
    selection = {'mode': 'shared2', 'seed': args.seed, 'alpha': chosen, 'stats': stats, 'history': history,
        'checkpoint': 'epoch8.pt', 'parameters': sum(p.numel() for p in net.parameters()), 'independent_head': True,
        'template_source': str(args.template), 'zero_identity': True, 'elapsed_s': time.monotonic()-start}
    original.save(args.out/'SELECTION.json', selection); original.save(args.out/'COMPLETE.json', selection)
    print(json.dumps(selection), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'train'])
    for key in ['bank', 'template', 'split', 'out']: parser.add_argument('--'+key, type=Path, required=True)
    for key in ['data', 'weights', 'geometry']: parser.add_argument('--'+key, type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    (prepare if args.action == 'prepare' else train)(args)


if __name__ == '__main__':
    main()
