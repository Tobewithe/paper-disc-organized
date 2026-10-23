from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT


def set_seed(seed=0):
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class Fusion(nn.Module):
    def __init__(self, memory, mode):
        super().__init__(); self.mode = mode; self.capture = False; self.maps = {}; self.intervention = 'none'
        self.register_buffer('mean', memory['mean'].view(1, -1, 1, 1))
        self.register_buffer('std', memory['std'].view(1, -1, 1, 1))
        self.projector = nn.Conv2d(256, 128, 1)
        self.projector.weight.data.copy_(memory['projector_weight'].view(128, 256, 1, 1))
        self.projector.bias.data.copy_(memory['projector_bias'])
        self.register_buffer('keys', memory['centers'].clone())
        self.register_buffer('values', memory['centers'].clone())
        self.readout = nn.Conv2d(256 if mode == 'spm' else 128, 256, 1)
        self.gate = nn.Conv2d(256, 1, 1)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.alpha = nn.Parameter(torch.tensor(0.))
        if mode == 'spm':
            self.projector.requires_grad_(False)

    def forward(self, x):
        normed = (x-self.mean)/self.std
        z = self.projector(normed)
        maps = {}
        if self.mode == 'spm':
            query = F.normalize(z, dim=1)
            sims = torch.einsum('bdhw,kd->bkhw', query, self.keys)
            responses = []
            for offset in [0, 16]:
                score, index = sims[:, offset:offset+16].topk(4, dim=1)
                weights = (score/.1).softmax(dim=1)
                retrieved = (self.values[index+offset]*weights.unsqueeze(-1)).sum(dim=1).permute(0, 3, 1, 2)
                responses.append(retrieved)
            z = torch.cat(responses, dim=1)
            if self.capture:
                maps = {'foreground_similarity': sims[:, :16].amax(1),
                        'background_similarity': sims[:, 16:].amax(1),
                        'top1': sims.argmax(1).float(), 'query': query}
        else:
            z = F.silu(z)
        residual = self.readout(z)
        # Both branches share a per-pixel RMS cap; alpha starts at exactly zero.
        feature_rms = x.square().mean(1, keepdim=True).add(1e-8).sqrt().detach()
        residual_rms = residual.square().mean(1, keepdim=True).add(1e-8).sqrt()
        residual = residual * torch.clamp(.25*feature_rms/residual_rms, max=1.)
        gate = self.gate(normed).sigmoid()
        delta = self.alpha.clamp(-.5, .5)*gate*residual
        if self.intervention == 'shuffle_spatial':
            delta = torch.roll(delta, shifts=(delta.shape[-2]//2, delta.shape[-1]//3), dims=(-2, -1))
        if self.capture:
            maps.update(gate=gate[:, 0], residual_rms=delta.square().mean(1).sqrt(),
                        feature_rms=feature_rms[:, 0])
            self.maps = {k: v.detach().cpu() for k, v in maps.items()}
        return x+delta


def attach(model, memory, mode):
    model.add_module('spm', Fusion(memory, mode).to(next(model.parameters()).device))
    def hook(head, args):
        levels = args[0]
        return ([model.spm(levels[0]), *levels[1:]],)
    model.model[-1].register_forward_pre_hook(hook)


def configure(model, steps):
    model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, 'overlap_mask': False, 'epochs': steps,
                                   'imgsz': 640, 'batch': 3, 'task': 'segment'})
    model.criterion = None
    return model


def build(weights, steps):
    set_seed()
    source = YOLO(str(weights)).model
    model = SegmentationModel(deepcopy(source.yaml), nc=1, verbose=False)
    model.names = {0: 'pig'}; model.load(source, verbose=False)
    model = configure(model, steps).to('cuda').float()
    assert model.model[-1].end2end and model.model[-1].nc == 1
    return model


def save(model, path, mode, step, memory=None):
    torch.save({'yaml': deepcopy(model.yaml), 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'mode': mode, 'step': step, 'memory': memory, 'names': {0: 'pig'},
                'format': 'spm_v4_fp32_state_dict', 'canvas': 640}, path)


def load(path, steps=400):
    data = torch.load(path, map_location='cpu', weights_only=False)
    model = SegmentationModel(deepcopy(data['yaml']), nc=1, verbose=False).to('cuda').float()
    model.names = {0: 'pig'}
    if data['mode'] in ['spm', 'adapter']:
        attach(model, data['memory'], data['mode'])
    model.load_state_dict(data['state_dict'], strict=True)
    return configure(model, steps)


def train_mode(model, freeze_body):
    model.train()
    if freeze_body:
        for layer in model.model[:-1]:
            layer.eval(); layer.requires_grad_(False)
        model.model[-1].requires_grad_(True)
        if hasattr(model, 'spm') and model.spm.mode == 'spm':
            model.spm.projector.requires_grad_(False)


@torch.no_grad()
def p3(model, images):
    captured = []
    handle = model.model[-1].register_forward_pre_hook(lambda head, args: captured.append(args[0][0].detach()))
    model.eval(); model(images); handle.remove()
    return captured[0]
