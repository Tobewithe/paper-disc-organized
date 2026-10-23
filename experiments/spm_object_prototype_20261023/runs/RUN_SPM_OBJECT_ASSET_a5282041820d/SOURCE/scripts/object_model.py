"""Matched-capacity image-to-object models; geometry binding is the intervention."""
import torch
from torch import nn
from torch.nn import functional as F

from geometry import skin


class ObjectModel(nn.Module):
    def __init__(self, template, cfg, structured):
        super().__init__()
        self.structured = structured
        self.shape_dim = cfg['shape_dim']
        for name in ['canonical', 'weights', 'part_weights']:
            self.register_buffer(name, torch.as_tensor(template[name], dtype=torch.float32))
        self.register_buffer('basis_init', torch.as_tensor(template['shape_basis'], dtype=torch.float32))
        bones = self.weights.shape[1]
        self.base_delta = nn.Parameter(torch.zeros_like(self.canonical))
        self.basis_delta = nn.Parameter(torch.zeros_like(self.basis_init))
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Conv2d(96, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        h = cfg['state_hidden_dim']
        self.state = nn.Sequential(nn.Linear(512+12, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.shape_head = nn.Linear(h, self.shape_dim)
        self.pose_head = nn.Linear(h, bones*12)
        self.uncertainty_head = nn.Linear(h, 1)
        self.residual = nn.Sequential(nn.Linear(3+self.shape_dim+bones*12, 128), nn.SiLU(),
                                      nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3))
        self.field = nn.Sequential(nn.Linear(3+self.shape_dim, 64), nn.SiLU(),
                                   nn.Linear(64, cfg['feature_dim']), nn.SiLU())
        self.part_classifier = nn.Linear(cfg['feature_dim'], 8)
        self.dense_head = nn.Conv2d(64, 11, 1)  # 3 canonical coordinates + 8 part logits
        nn.init.zeros_(self.pose_head.weight); nn.init.zeros_(self.pose_head.bias)
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)

    def encode(self, rgb, camera):
        x = rgb
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i == 5:
                dense = self.dense_head(x)
        state = self.state(torch.cat([self.pool(x).flatten(1), camera.flatten(1)], 1))
        z = torch.tanh(self.shape_head(state))
        pose = self.pose_head(state).reshape(-1, self.weights.shape[1], 3, 4)
        identity = torch.eye(4, device=rgb.device)[:3]
        pose = pose+identity
        return z, pose, dense, self.uncertainty_head(state).clamp(-8, 0)

    def decode(self, z, pose, wrong_binding=False):
        base = self.canonical+.025*torch.tanh(self.base_delta)
        basis = self.basis_init+.01*torch.tanh(self.basis_delta)
        rest = base[None]+torch.einsum('bd,dvc->bvc', z, basis)
        vertices = skin(rest, pose, self.weights.roll(1, 0) if wrong_binding else self.weights) if self.structured else rest
        shared = torch.cat([base[None].expand(z.shape[0], -1, -1), z[:, None].expand(-1, len(base), -1)], -1)
        point_state = torch.cat([shared, pose.flatten(1)[:, None].expand(-1, len(base), -1)], -1)
        # Same flexible residual and exact same trainable parameters in both arms.
        vertices = vertices+.75*torch.tanh(self.residual(point_state))
        features = self.field(shared)
        return vertices, features, self.part_classifier(features)

    def forward(self, rgb, camera, wrong_binding=False):
        z, pose, dense, log_sigma = self.encode(rgb, camera)
        vertices, features, part_logits = self.decode(z, pose, wrong_binding)
        return dict(vertices=vertices, shape=z, pose=pose, dense=dense, features=features,
                    part_logits=part_logits, log_sigma=log_sigma)


def objective(output, pair_output, batch, template, cfg):
    from geometry import project
    weights = cfg['loss_weights']
    gt = batch['vertices']
    losses = {'vertices': F.smooth_l1_loss(output['vertices'], gt, beta=.05),
              'shape': F.mse_loss(output['shape'], batch['shape']),
              'pose': F.mse_loss(output['pose'], batch['pose']),
              'shape_consistency': F.mse_loss(output['shape'], pair_output['shape'])}
    pred_uv, _ = project(output['vertices'], batch['target_camera'], batch['focal'][0], cfg['image_size'])
    true_uv, _ = project(gt, batch['target_camera'], batch['focal'][0], cfg['image_size'])
    losses['novel_view_points'] = F.smooth_l1_loss(pred_uv/cfg['image_size'], true_uv/cfg['image_size'], beta=.03)
    canonical = F.interpolate(batch['canonical'].permute(0, 3, 1, 2), output['dense'].shape[-2:], mode='nearest')
    valid = F.interpolate(batch['visible'][:, None].float(), output['dense'].shape[-2:], mode='nearest')[:, 0].bool()
    parts = F.interpolate(batch['parts'][:, None].float(), output['dense'].shape[-2:], mode='nearest')[:, 0].long()
    pred = output['dense']
    losses['canonical'] = F.smooth_l1_loss(pred[:, :3].permute(0, 2, 3, 1)[valid], canonical.permute(0, 2, 3, 1)[valid], beta=.05)
    losses['part'] = F.cross_entropy(pred[:, 3:].permute(0, 2, 3, 1)[valid], parts[valid])
    part_labels = template['part_weights'].argmax(-1)[None].expand(pred.shape[0], -1)
    losses['part'] = losses['part'] + F.cross_entropy(output['part_logits'].transpose(1, 2), part_labels)
    error = (output['vertices'].detach()-gt).norm(dim=-1).mean(-1, keepdim=True)
    losses['uncertainty'] = F.mse_loss(output['log_sigma'], error.clamp_min(.0004).log().clamp(-8, 0))
    return sum(weights[k]*v for k, v in losses.items()), losses
