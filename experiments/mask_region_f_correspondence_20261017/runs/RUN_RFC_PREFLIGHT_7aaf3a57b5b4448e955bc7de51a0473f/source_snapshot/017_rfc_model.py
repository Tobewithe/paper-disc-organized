"""RFC-v1: original 167872 parameters, shared regional-F injection, A/Q/J controls."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from rfc_paths import require_original_module
import adapter_model
import adapter_primitives
from rfc_inputs import RegionInputs, spatial_indices, require

require_original_module(adapter_model, 'adapter_model.py')
require_original_module(adapter_primitives, 'adapter_primitives.py')


class RegionFAdapter(adapter_model.CoefficientAdapter):
    def __init__(self, arm):
        require(arm in ('A', 'Q'), 'A or Q required')
        super().__init__('B')
        self.arm = arm

    def forward(self, inputs, phase, *, intervention=None):
        require(isinstance(inputs, RegionInputs), 'validated RegionInputs required')
        require(phase in ('train:0', 'train:1', 'train:2', 'eval:0'), 'invalid phase')
        require(inputs.scope == ('eval_gated_rows' if phase == 'eval:0' else 'train_all_rows'),
                'training/evaluation input scope mismatch')
        require(intervention is None or
                (intervention == 'J' and self.arm == 'A' and phase == 'eval:0'),
                'J is only a same-A-checkpoint evaluation intervention')
        device = self.readout[0].weight.device
        require(all(p.dtype == torch.float32 for p in self.parameters()), 'FP32 adapter required')
        p, c, boxes = (x.detach().to(device).float() for x in
                       (inputs.proto, inputs.c0, inputs.boxes_input))
        n = len(c)
        if n == 0:
            # The last linear makes the zero-output training loss differentiable.
            return self.readout(c.new_empty((0, 181)))
        spatial = adapter_primitives.sample_spatial_roi(p, c, boxes, inputs.input_hw)
        h1 = self.spatial[1](self.spatial[0](spatial))
        points = c.new_zeros((n, 64))
        regional = c.new_zeros((n, 64, 8, 8))
        preserve = self.arm == 'A' and intervention is None
        for level, projection in enumerate(self.feature):
            ranks = [i for i, actual_level in enumerate(inputs.levels) if actual_level == level]
            if not ranks:
                continue
            indices = torch.tensor(ranks, device=device, dtype=torch.long)
            old_f = torch.stack([inputs.features[i].detach() for i in ranks]).to(device).float()
            grid_f = torch.stack([inputs.regions[i].detach() for i in ranks]).to(device).float()
            permutation = torch.tensor([spatial_indices(inputs.image_id, i, preserve) for i in ranks],
                                       device=device, dtype=torch.long)
            grid_f = grid_f.flatten(2).gather(2, permutation[:, None].expand(-1, grid_f.shape[1], -1))
            difference = grid_f.transpose(1, 2) - old_f[:, None, :]
            delta_features = F.linear(difference, projection.weight, bias=None)
            delta_features = delta_features.transpose(1, 2).reshape(len(ranks), 64, 8, 8)
            points = points.index_copy(0, indices, projection(old_f))
            regional = regional.index_copy(0, indices, delta_features)
        spatial = self.spatial[3](self.spatial[2](h1 + regional)).mean(dim=(-2, -1))
        height, width = inputs.input_hw
        normalized_boxes = boxes / boxes.new_tensor([width, height, width, height])
        joined = torch.cat((spatial, points, self.coefficient(c), normalized_boxes,
                            inputs.scores.detach().to(device)[:, None],
                            self.category(inputs.classes.detach().to(device).long())), dim=1)
        result = self.readout(joined)
        require(result.shape == (n, 32) and bool(torch.isfinite(result).all()), 'invalid delta_c')
        return result


def parameter_receipt(model):
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'parameter_tensors': len(list(model.parameters())),
            'state_shapes': {k: list(v.shape) for k, v in model.state_dict().items()}}
