from copy import deepcopy
import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.tasks import SegmentationModel

from base_model import configure


class Projector(nn.Module):
    def __init__(self):
        super().__init__(); self.first = nn.Linear(512, 256); self.last = nn.Linear(256, 128)

    def forward(self, x):
        return self.last(F.silu(self.first(x)))


class Coefficients(nn.Module):
    def __init__(self, native, extra=32):
        super().__init__(); self.native = native
        self.extra = nn.Conv2d(native.in_channels, extra, native.kernel_size, native.stride, native.padding, bias=True)
        nn.init.zeros_(self.extra.weight); nn.init.zeros_(self.extra.bias)

    def forward(self, x):
        return torch.cat([self.native(x), self.extra(x)], dim=1)


class MemoryBases(nn.Module):
    def __init__(self, memory, mode):
        super().__init__(); self.mode = mode; self.intervention = 'none'; self.capture = False; self.maps = {}
        self.register_buffer('mean', memory['mean'].reshape(1, 512, 1, 1))
        self.register_buffer('std', memory['std'].reshape(1, 512, 1, 1))
        self.detail = None
        self.projector = Projector(); self.projector.load_state_dict(memory['projector']); self.projector.requires_grad_(False)
        generator = torch.Generator().manual_seed(23102351)
        random_keys = F.normalize(torch.randn(32, 128, generator=generator), dim=1)
        self.register_buffer('keys', random_keys.clone() if mode == 'random_memory' else memory['centers'].clone())
        self.register_buffer('random_keys', random_keys.clone())
        if mode == 'adapter':
            self.mix = nn.Sequential(nn.Conv2d(128, 64, 1), nn.SiLU(), nn.Conv2d(64, 32, 1))
        else:
            self.mix = nn.Conv2d(32, 32, 1)
            self.mix.weight.data.zero_(); self.mix.bias.data.zero_()
            self.mix.weight.data[:, :, 0, 0].copy_(torch.eye(32))

    def forward(self, x):
        assert self.detail is not None and self.detail.shape[-2:] == (160, 160)
        features = torch.cat([self.detail, F.interpolate(x, size=self.detail.shape[-2:], mode='bilinear', align_corners=False)], dim=1)
        query = self.projector(((features-self.mean)/self.std).permute(0, 2, 3, 1))
        query = F.normalize(query, dim=-1)
        if self.mode == 'adapter':
            raw = self.mix(query.permute(0, 3, 1, 2)); scores = None
        else:
            keys = self.random_keys if self.intervention == 'random_dictionary' else self.keys
            scores = torch.einsum('bhwd,kd->bkhw', query, keys)
            fg, bg = scores[:, :16], scores[:, 16:]
            contrast = torch.cat([fg-bg.amax(1, keepdim=True), bg-fg.amax(1, keepdim=True)], dim=1)
            raw = self.mix((contrast/.15).tanh())
        if self.capture:
            maps = {'query': query.permute(0, 3, 1, 2)}
            if scores is not None:
                maps.update(foreground_similarity=scores[:, :16].amax(1), background_similarity=scores[:, 16:].amax(1),
                            top1=scores.argmax(1))
            self.maps = {k: v.detach().cpu() for k, v in maps.items()}
        return raw

    def frozen_hash(self):
        h = hashlib.sha256()
        tensors = {**{'projector.'+k: v for k, v in self.projector.state_dict().items()},
                   'mean': self.mean, 'std': self.std, 'keys': self.keys, 'random_keys': self.random_keys}
        for name, tensor in sorted(tensors.items()):
            h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()


class ProtoWithMemory(nn.Module):
    def __init__(self, native, memory, mode):
        super().__init__(); self.native = native; self.memory = MemoryBases(memory, mode)

    def forward(self, x, return_semantic=True):
        native = self.native(x, return_semantic=return_semantic)
        base = native[0] if isinstance(native, tuple) else native
        extra = F.interpolate(self.memory(x[0]), size=base.shape[-2:], mode='bilinear', align_corners=False)
        # The same spatial centering and RMS normalization is applied to F/G/R.
        work = extra.float(); centered = work-work.mean((-2, -1), keepdim=True)
        extra = centered/(centered.square().mean((-2, -1), keepdim=True)+1e-6).sqrt()
        mode = self.memory.intervention
        if mode == 'off':
            extra = torch.zeros_like(extra)
        elif mode == 'permute_channels':
            extra = torch.roll(extra, 1, 1)
        elif mode == 'shuffle_spatial':
            extra = torch.roll(extra, shifts=(extra.shape[-2]//2, extra.shape[-1]//3), dims=(-2, -1))
        if self.memory.capture:
            self.memory.maps['extra_bases'] = extra.detach().cpu()
        output = torch.cat([base, extra.to(base.dtype)], dim=1)
        return (output, native[1]) if isinstance(native, tuple) else output


def augment(model, memory, mode):
    head = model.model[-1]
    assert head.nm == 32 and head.end2end
    for name in ['cv4', 'one2one_cv4']:
        for level in getattr(head, name):
            assert isinstance(level[-1], nn.Conv2d) and level[-1].out_channels == 32
            level[-1] = Coefficients(level[-1])
    head.proto = ProtoWithMemory(head.proto, memory, mode)
    head.nm = 64
    memory_module = head.proto.memory
    def capture_detail(layer, inputs, output):
        memory_module.detail = output.detach()
    model.model[2].register_forward_hook(capture_detail)
    model.to(next(model.parameters()).device)


def load(path, steps=8000, device='cuda'):
    data = torch.load(path, map_location='cpu', weights_only=False)
    model = SegmentationModel(deepcopy(data['yaml']), nc=1, verbose=False).float().to(device)
    model.names = {0: 'pig'}
    if data['mode'] in ['task_spm', 'adapter', 'random_memory']:
        augment(model, data['memory'], data['mode'])
    model.load_state_dict(data['state_dict'], strict=True)
    return configure(model, steps)


def save(model, path, mode, step, memory=None):
    torch.save({'yaml': deepcopy(model.yaml), 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'mode': mode, 'step': step, 'memory': memory, 'format': 'spm_v4_1_task_mask_bases_fp32',
                'names': {0: 'pig'}, 'original_bases': 32, 'added_bases': 32 if memory is not None else 0}, path)


def train_mode(model, mask_only):
    model.train()
    for layer in model.model[:-1]:
        layer.eval(); layer.requires_grad_(False)
    head = model.model[-1]; head.requires_grad_(not mask_only)
    if mask_only:
        head.proto.requires_grad_(True); head.cv4.requires_grad_(True); head.one2one_cv4.requires_grad_(True)
        for name in ['cv2', 'cv3', 'one2one_cv2', 'one2one_cv3']:
            getattr(head, name).eval()
    if isinstance(head.proto, ProtoWithMemory):
        head.proto.memory.projector.requires_grad_(False)


def memory_module(model):
    proto = model.model[-1].proto
    return proto.memory if isinstance(proto, ProtoWithMemory) else None


@torch.no_grad()
def query_features(model, images):
    model.eval(); x = images; saved = []; detail = None
    for layer in model.model[:17]:
        if layer.f != -1:
            x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
        x = layer(x); saved.append(x if layer.i in model.save else None)
        if layer.i == 2:
            detail = x
    assert x.shape[1:] == (256, 80, 80) and detail.shape[1:] == (256, 160, 160)
    return torch.cat([detail, F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)], dim=1)
