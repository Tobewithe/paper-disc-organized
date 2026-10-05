import sys
import torch

p = sys.argv[1]
d = torch.load(p, map_location="cpu", weights_only=False)
print("type", type(d).__name__)
print("keys", list(d.keys()) if isinstance(d, dict) else None)
if isinstance(d, dict):
    for key, value in d.items():
        print(key, type(value).__name__)
    model = d.get("model")
    if hasattr(model, "state_dict"):
        keys = list(model.state_dict().keys())
        print("model_state_keys", len(keys), "quality", [k for k in keys if "quality_head" in k][:8])
    state = d.get("state_dict")
    if isinstance(state, dict):
        print("state_dict_keys", len(state), "quality", [k for k in state if "quality_head" in k][:8])
