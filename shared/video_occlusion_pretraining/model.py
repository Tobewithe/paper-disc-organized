import torch
from torch import nn
from torch.nn import functional as F


def grid_xy(height, width, device):
    y, x = torch.meshgrid((torch.arange(height, device=device)+.5)/height,
                          (torch.arange(width, device=device)+.5)/width, indexing="ij")
    return torch.stack((x, y), -1).reshape(-1, 2)


class TemporalBlock(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.qnorm, self.knorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))

    def forward(self, query, context):
        value = self.knorm(context)
        query = query + self.attn(self.qnorm(query), value, value, need_weights=False)[0]
        return query + self.ffn(self.norm(query))


class VideoRepresentation(nn.Module):
    """Only RGB-derived features, timestamps and initial observed prompts enter.

    Region prompts are initial visible bounding boxes, not per-frame GT crops.
    The feature extractor is frozen; adapter and causal memory are reusable.
    """
    def __init__(self, backbone_dim=384, dim=256, height=20, width=32):
        super().__init__()
        self.height, self.width, self.dim = height, width, dim
        self.adapter = nn.Sequential(nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, 64), nn.GELU(), nn.Linear(64, backbone_dim))
        self.project = nn.Linear(backbone_dim, dim)
        self.position = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, dim))
        self.prompt_type = nn.Embedding(2, dim)
        self.time = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, dim))
        self.blocks = nn.ModuleList([TemporalBlock(dim), TemporalBlock(dim)])
        self.match = nn.Linear(dim, dim)
        self.null = nn.Linear(dim, 1)
        self.offset = nn.Linear(dim, 2)
        self.state = nn.Linear(dim, 4)
        self.region = nn.Linear(dim, 2)
        self.content = nn.Linear(dim, 128)
        self.confidence = nn.Sequential(nn.Linear(dim+3, 64), nn.GELU(), nn.Linear(64, 1))
        self.log_scale = nn.Parameter(torch.tensor(2.3))

    def encode(self, features):
        features = features.to(self.project.weight.dtype)
        return self.project(features + self.adapter(features))

    def anchors(self, spatial, points, boxes):
        b, _, d = spatial.shape
        maps = spatial.transpose(1, 2).reshape(b, d, self.height, self.width)
        sampled = F.grid_sample(maps, points[:, :, None, :]*2-1,
                                align_corners=False).squeeze(-1).transpose(1, 2)
        coords = grid_xy(self.height, self.width, spatial.device)
        mask = ((coords[None,None,:,0] >= boxes[:,:,None,0]) &
                (coords[None,None,:,1] >= boxes[:,:,None,1]) &
                (coords[None,None,:,0] <= boxes[:,:,None,2]) &
                (coords[None,None,:,1] <= boxes[:,:,None,3])).float()
        region = torch.einsum("bqn,bnd->bqd", mask, spatial)/mask.sum(-1,keepdim=True).clamp_min(1)
        prompt_xy = torch.cat((points, (boxes[:,:,:2]+boxes[:,:,2:])/2), 1)
        types = torch.cat((torch.zeros(points.shape[1],device=spatial.device,dtype=torch.long),
                           torch.ones(boxes.shape[1],device=spatial.device,dtype=torch.long)))
        # Keep appearance and position separate in matching.
        appearance = torch.cat((sampled, region), 1)
        anchor = appearance + self.position(prompt_xy) + self.prompt_type(types)[None]
        return anchor, appearance

    def forward(self, features, timestamps, initial_points, initial_boxes, arm="P",
                return_spatial=False, memory_intervention="none"):
        assert arm in ("B", "P", "I", "Hpos")
        assert features.ndim == 4 and timestamps.shape == features.shape[:2]
        b, t, n, _ = features.shape
        assert n == self.height*self.width
        spatial = self.encode(features)
        anchor, appearance = self.anchors(spatial[:,0], initial_points, initial_boxes)
        q = anchor.shape[1]
        # Current observations use the preserved initial appearance; no future data.
        spatial_n, appearance_n = F.normalize(spatial,dim=-1), F.normalize(appearance,dim=-1)
        attention = torch.einsum("bqd,btnd->btqn", appearance_n, spatial_n)*10
        obs_weights = attention.softmax(-1)
        pos = self.position(grid_xy(self.height, self.width, features.device))
        obs = torch.einsum("btqn,btnd->btqd", obs_weights, spatial+pos[None,None])
        memory, rows = anchor, []
        for index in range(t):
            if arm == "I":
                memory = anchor
                recent = obs[:,index:index+1].expand(-1,8,-1,-1)
                dt = timestamps[:,index]-timestamps[:,max(0,index-1)]
                if index == 0:
                    dt = torch.full_like(dt,1/12)
                # Slot ages use only current elapsed time, not past imagery.
                # Keeping time projections active avoids dead-parameter padding.
                ages = dt[:,None]*torch.arange(7,-1,-1,device=features.device)[None]
            else:
                start = max(0,index-7)
                recent = obs[:,start:index+1]
                ages = timestamps[:,index:index+1] - timestamps[:,start:index+1]
            if index > 0 and memory_intervention != "none" and arm != "I":
                point_count = initial_points.shape[1]
                exchanged = torch.cat((memory[:,:point_count].roll(8,dims=1),
                                       memory[:,point_count:].roll(1,dims=1)),dim=1)
                if memory_intervention == "other_query":
                    memory = exchanged
                elif memory_intervention == "matched_norm":
                    # Deterministic sign pattern; same perturbation norm as the
                    # identity exchange, without importing another query's state.
                    direction = torch.sin(torch.arange(self.dim,device=memory.device)*1.7)
                    direction = F.normalize(direction,dim=0)
                    amplitude = (exchanged-memory).norm(dim=-1,keepdim=True)
                    memory = memory + amplitude*direction
                else:
                    raise ValueError(memory_intervention)
            age_features = torch.stack((ages, ages.square()),-1)
            recent = recent + self.time(age_features)[:,:,None]
            context = torch.cat((anchor[:,:,None], memory[:,:,None],
                                 recent.transpose(1,2)),2).reshape(b*q,-1,self.dim)
            current = (obs[:,index] + anchor).reshape(b*q,1,self.dim)
            for block in self.blocks:
                current = block(current,context)
            memory = current.reshape(b,q,self.dim)
            rows.append(memory)
        hidden = torch.stack(rows,1)
        scores = torch.einsum("btqd,btnd->btqn",
                             F.normalize(self.match(hidden),dim=-1), spatial_n)
        scores = scores*self.log_scale.exp().clamp(max=30)
        logits = torch.cat((scores,self.null(hidden)), -1)
        probabilities = logits.softmax(-1)
        cell_xy = grid_xy(self.height,self.width,features.device)
        cell_xy = cell_xy[None,None] + self.offset(spatial).tanh()*.5/torch.tensor(
            [self.width,self.height],device=features.device)
        conditional = scores.softmax(-1)
        xy_mean = torch.einsum("btqn,btnd->btqd",conditional,cell_xy)
        idx = scores.argmax(-1)
        xy = cell_xy[:,:,None].expand(-1,-1,q,-1,-1).gather(
            3,idx[:,:,:,None,None].expand(-1,-1,-1,1,2)).squeeze(3)
        probability_info = torch.stack((
            probabilities[...,-1], conditional.max(-1).values,
            -(conditional*conditional.clamp_min(1e-8).log()).sum(-1)/math_log(n)), -1)
        conf = self.confidence(torch.cat((hidden.detach(),probability_info.detach()),-1)).squeeze(-1)
        result = dict(logits=logits, xy=xy, xy_mean=xy_mean, state=self.state(hidden),
                      region=self.region(hidden), confidence=conf, h=hidden,
                      z=F.normalize(self.content(hidden),dim=-1))
        if return_spatial:
            result["spatial"] = spatial
        return result


def math_log(n):
    import math
    return math.log(n)

