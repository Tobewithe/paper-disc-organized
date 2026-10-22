"""Frozen local prototype retrieval; accepts raw YOLO P3 feature vectors."""
import torch
from torch import nn
import torch.nn.functional as F


class FrozenPrototypeMemory(nn.Module):
    def __init__(self, payload):
        super().__init__()
        for name in ["keys_fg", "values_fg", "keys_bg", "values_bg", "input_mean", "input_std"]:
            self.register_buffer(name, payload[name].clone())
        self.register_buffer("weight", payload["projector_state"]["weight"].clone())
        self.register_buffer("bias", payload["projector_state"]["bias"].clone())
        self.k = payload["topk"]; self.temperature = payload["temperature"]

    def forward(self, features):
        assert features.ndim == 2 and features.shape[1] == self.weight.shape[1]
        q = F.normalize(F.linear((features-self.input_mean)/self.input_std, self.weight, self.bias), dim=1)
        result = {}
        for part, key, value in [("foreground", self.keys_fg, self.values_fg), ("background", self.keys_bg, self.values_bg)]:
            cos = q@key.T; score, idx = cos.topk(min(self.k, len(key)), dim=1)
            result[part] = (F.softmax(score/self.temperature, dim=1).unsqueeze(-1)*value[idx]).sum(1)
            result[part+"_indices"] = idx; result[part+"_max_cosine"] = score[:, 0]
        return result
