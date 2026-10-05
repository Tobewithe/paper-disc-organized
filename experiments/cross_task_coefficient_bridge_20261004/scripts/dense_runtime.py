"""Same native branch and full official pixel loss; only support weights vary."""
from __future__ import annotations
import hashlib, json, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import ops
from joint_head import JointCoefficientReadout
from online_runtime import load_json, sha256, tensor_sha


def setup(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def new_model(replay, cfg):
    setup(cfg['seed'])
    return JointCoefficientReadout(replay.native_cv4, replay.feature_channels, 'N', cfg).to(replay.device).float()


def state_digest(model):
    return {k: tensor_sha(v) for k, v in model.state_dict().items()}


def buffers(model):
    return {k: v.detach().cpu().clone() for k, v in model.named_buffers()}


def verify_buffers(model, previous):
    assert set(previous) == {k for k, _ in model.named_buffers()}
    for k, v in model.named_buffers():
        torch.testing.assert_close(v.detach().cpu(), previous[k], atol=0, rtol=0)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert not module.training


def gpu_payload(image, device):
    keys = ('proto', 'masks', 'owners', 'target_boxes', 'segmentation_gain')
    return {k: image[k].to(device) if torch.is_tensor(image[k]) else image[k] for k in keys}


def pixel_values(image, coefficients, proto640):
    owner = image['owners']; box = image['target_boxes']
    area = ((box[:, 2:] - box[:, :2])/640).prod(1)
    if not bool((area > 0).all()):
        raise ValueError('Non-positive original GT-box area')
    target = (image['masks'][None] == (owner + 1)[:, None, None]).float()
    logits = torch.einsum('in,nhw->ihw', coefficients.float(), proto640)
    pixels = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return ops.crop_mask(pixels, box).mean((1, 2))/area*float(image['segmentation_gain'])


def weighted_loss_and_grad(image, coefficients, normalizer, weights):
    """Exact coefficient chain rule; normalizer counts ORIGINAL covered GTs."""
    if normalizer <= 0 or weights.shape != (len(coefficients),):
        raise ValueError('Bad normalization/weight dimensions')
    weights = weights.to(coefficients.device, coefficients.dtype)
    if not bool(torch.isfinite(weights).all() and (weights > 0).all()):
        raise ValueError('All preserved support weights must be positive finite')
    proto = F.interpolate(image['proto'][None].float(), (640, 640), mode='bilinear', align_corners=False)[0]
    gradient = torch.zeros_like(coefficients); total = 0.0
    for lo in range(0, len(coefficients), 8):
        hi = min(lo + 8, len(coefficients))
        leaf = coefficients[lo:hi].detach().requires_grad_(True)
        part = dict(image, owners=image['owners'][lo:hi], target_boxes=image['target_boxes'][lo:hi])
        value = (pixel_values(part, leaf, proto)*weights[lo:hi]).sum()
        if not bool(torch.isfinite(value)):
            raise FloatingPointError('Nonfinite exact weighted mask loss')
        gradient[lo:hi] = torch.autograd.grad(value/normalizer, leaf)[0]
        total += float(value.detach())
    return total, gradient


@torch.no_grad()
def official_values(image, coefficients):
    proto = F.interpolate(image['proto'][None].float(), (640, 640), mode='bilinear', align_corners=False)[0]
    values = []
    for lo in range(0, len(coefficients), 8):
        hi = min(lo + 8, len(coefficients))
        part = dict(image, owners=image['owners'][lo:hi], target_boxes=image['target_boxes'][lo:hi])
        values.append(pixel_values(part, coefficients[lo:hi], proto).detach().cpu())
    return torch.cat(values) if values else torch.empty(0)


def optimizer_for(model, cfg):
    groups = {}
    for name, p in model.named_parameters():
        assert name.startswith('native_cv4.') and p.requires_grad
        groups.setdefault(p.ndim > 1, []).append(p)
    return torch.optim.AdamW([dict(params=ps, lr=cfg['branch_lr'], initial_lr=cfg['branch_lr'],
        weight_decay=cfg['weight_decay'] if decay else 0., group_name='native_decay' if decay else 'native_no_decay')
        for decay, ps in groups.items()], betas=(.9, .999), eps=1e-8)


def lr_factor(progress, cfg):
    warm = cfg['warmup_epochs']
    if progress < warm:
        return max(.01, progress/warm)
    phase = min(1., (progress-warm)/max(1, cfg['epochs']-warm))
    return cfg['eta_ratio'] + (1-cfg['eta_ratio'])*.5*(1+math.cos(math.pi*phase))


def check_control(cfg, model=None):
    old = load_json(cfg['control_config'])
    for key in cfg['paired_keys']:
        if cfg[key] != old[key]:
            raise AssertionError('Paired setting differs: '+key)
    for key in ('cache','assets','weights','source_python','annotations_train'):
        assert cfg[key] == old[key], key
    assert load_json(cfg['split']) == load_json(old['split'])
    complete = load_json(Path(cfg['control_run'])/'COMPLETE.json')
    assert complete['completed'] and complete['epochs'] == 3 and complete['mode'] == 'N'
    info = load_json(Path(cfg['control_run'])/'MODEL.json')
    if model is not None:
        initial_sha = hashlib.sha256(json.dumps(state_digest(model), sort_keys=True).encode()).hexdigest()
        assert initial_sha == info['initialization_state_sha256'], 'Initialization differs from N'
        assert model.parameter_counts() == info['counts']
        assert model.trainable_parameter_names() == info['trainable']
    return dict(checkpoint=cfg['control_checkpoint'], checkpoint_sha256=sha256(cfg['control_checkpoint']),
                run=cfg['control_run'], run_complete_sha256=sha256(Path(cfg['control_run'])/'COMPLETE.json'),
                paired_settings={key:cfg[key] for key in cfg['paired_keys']},
                reason='Repeated original support is analytically original N; value and full-parameter gradient smoke required')
