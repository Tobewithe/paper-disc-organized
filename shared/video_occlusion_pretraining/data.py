from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from .common import read, sha


def labels(folder, flip):
    a = dict(np.load(folder/"tracks.npz"))
    maps = np.stack([np.load(folder/"labels"/f"{f:03d}.npz")["owner"] for f in range(48)])
    xy = a["xy"].astype(np.float32)
    if flip:
        xy[...,0] = 1-xy[...,0]
        maps = maps[:,:,::-1].copy()
    boxes, region_active = [], []
    for owner in range(1,4):
        y,x = np.where(maps[0] == owner)
        region_active.append(bool(len(x)))
        boxes.append([x.min()/128,y.min()/80,(x.max()+1)/128,(y.max()+1)/80] if len(x) else [0,0,0,0])
    initial_points = xy[0].copy()
    # Inactive queries have no observed initial prompt; replace their coordinates.
    initial_points[~a["active"]] = 0
    result = dict(xy=xy,state=a["state"].astype(np.int64),valid=a["valid"].astype(bool),
        active=a["active"].astype(bool),purity=a["purity"].astype(np.float32),
        owners=a["owners"].astype(np.int64),owner_maps=maps.astype(np.int64),
        inframe=a["instance_inframe"].astype(bool),ratio=a["instance_ratio"].astype(np.float32),
        region_active=np.asarray(region_active),initial_boxes=np.asarray(boxes,np.float32),
        initial_points=initial_points,timestamps=a["timestamps"].astype(np.float32))
    return {k:torch.from_numpy(v.copy()) for k,v in result.items()}


class FamilyData:
    def __init__(self, dataset, features):
        self.dataset,self.features = Path(dataset),Path(features)
        self.cache = OrderedDict()
        self.label_cache = {}
        manifest=self.features.parent/"FEATURE_MANIFEST.json"
        self.expected={(r["family"],r["variant"]):r["sha256"] for r in read(manifest)}
        self.verified=set()

    def load(self, family, variant, flip):
        key=(family,variant,flip)
        if key not in self.cache:
            artifact=(family,variant)
            if artifact not in self.verified:
                assert sha(self.features/family/f"{variant}.pt")==self.expected[artifact]
                self.verified.add(artifact)
            if key not in self.label_cache:
                self.label_cache[key]=labels(self.dataset/family/str(variant),flip)
            lab=self.label_cache[key]
            # Bound CPU cache. Clone the selected view so eviction releases its
            # full underlying storage, instead of retaining both encoded views.
            feat=torch.load(self.features/family/f"{variant}.pt",weights_only=True,
                            map_location="cpu",mmap=True)[flip].clone()
            self.cache[key]=(feat,lab)
            while len(self.cache)>32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def four_views(self, family, variant, device="cuda"):
        pairs=[self.load(family,v,f) for v,f in [(0,0),(0,1),(variant,0),(variant,1)]]
        features=torch.stack([p[0] for p in pairs]).to(device)
        targets={k:torch.stack([p[1][k] for p in pairs]).to(device) for k in pairs[0][1]}
        # Only this explicitly constructed tuple is passed into the model.
        inputs=dict(features=features,timestamps=targets["timestamps"],
                    initial_points=targets["initial_points"],initial_boxes=targets["initial_boxes"])
        return inputs,targets
