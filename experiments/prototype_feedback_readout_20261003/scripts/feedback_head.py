"""Two-step coefficient refinement: only pooling-state freshness differs F/R."""
from __future__ import annotations
import torch
from torch import nn
from torchvision.ops import roi_align
from evidence_head import EvidenceReadout
from joint_head import JointCoefficientReadout


class FeedbackEvidence(EvidenceReadout):
    def __init__(self, channels, cfg):
        super().__init__(channels, 'S', cfg)
        self.query_mlp = nn.Sequential(nn.Linear(64+32+16+128,128),nn.SiLU(),nn.Linear(128,64))
        self.pooling_floor=float(cfg['pooling_floor'])
        assert self.pooling_floor == .05 and cfg['feedback_steps'] == 2

    def forward_steps(self, features, selected, mode, hold=False):
        key_f=self._encode(features,len(selected))
        counts=[f.shape[-2]*f.shape[-1] for f in features]
        coefficients=[]; intermediates=[]; traces=[]
        for image,row in enumerate(selected):
            cstart=row['c_current']; h=row['h_current']; n=len(cstart)
            boxes=row['boxes'].clamp(0,640)
            valid=row['valid'] & torch.isfinite(row['boxes']).all(1)
            valid=valid & ((boxes[:,2:]-boxes[:,:2])>0).all(1)
            pos=torch.where(valid)[0]
            raw=row['raw_ids']; levels=(raw>=counts[0]).long()+(raw>=counts[0]+counts[1]).long()
            c1=cstart.clone(); c2=cstart.clone(); summaries=[]
            for start in range(0,len(pos),self.roi_chunk_size):
                ix=pos[start:start+self.roi_chunk_size]
                rois=torch.cat([boxes[ix].new_zeros((len(ix),1)),boxes[ix]],1)
                roi=roi_align(key_f[image:image+1],rois,(16,16),spatial_scale=key_f.shape[-2]/640,
                              sampling_ratio=2,aligned=True)
                tokens=roi.flatten(2); a=row['A'][ix]; k=row['K'][ix]
                c=cstart[ix]; zstart=torch.bmm(a,c.unsqueeze(-1)).squeeze(-1)
                initial_context=None; changes=[]
                for step in range(2):
                    z=torch.bmm(a,c.unsqueeze(-1)).squeeze(-1)
                    gate=zstart if mode=='F' or hold else z
                    s=gate.detach().sigmoid()
                    wp=self.pooling_floor+(1-self.pooling_floor)*s
                    wn=self.pooling_floor+(1-self.pooling_floor)*(1-s)
                    fg=(tokens*wp[:,None]).sum(-1)/wp.sum(-1,keepdim=True)
                    bg=(tokens*wn[:,None]).sum(-1)/wn.sum(-1,keepdim=True)
                    context=torch.cat([fg,bg],1)
                    if initial_context is None: initial_context=context.detach()
                    changes.append((context.detach()-initial_context).abs().max())
                    q=self.query_mlp(torch.cat([h[ix],c,self.level_embedding(levels[ix]),context],1))
                    gamma,beta=self.query_film(q).chunk(2,1)
                    conditioned=roi*(1+gamma[:,:,None,None])+beta[:,:,None,None]
                    xy=self.relative_xy.expand(len(ix),-1,-1,-1)
                    logits=self.evidence_decoder(torch.cat([conditioned,xy,z.reshape(-1,1,16,16)],1))
                    residual=4*torch.tanh(logits.float().flatten(1)/4)
                    c=c+.5*torch.bmm(k,residual.unsqueeze(-1)).squeeze(-1)
                    if step==0: c1=c1.index_copy(0,ix,c)
                c2=c2.index_copy(0,ix,c)
                summaries.append(torch.stack(changes))
            coefficients.append(c2);intermediates.append(c1)
            traces.append(dict(valid=int(valid.sum()),context_change_step2=float(torch.stack(summaries)[:,1].max()) if summaries else 0.))
        return coefficients,intermediates,traces


class FeedbackReadout(JointCoefficientReadout):
    def __init__(self,native,channels,mode,cfg):
        if mode not in ('N','F','R'): raise ValueError(mode)
        super().__init__(native,channels,'N',cfg)
        self.mode=mode
        self.evidence=None if mode=='N' else FeedbackEvidence(channels,cfg)
        self.train()

    def forward_trace(self,features,selected,hold=False):
        current=self.current_selection(features,selected)
        if self.mode=='N':
            cs=[row['c_current'] for row in current]
            return cs,cs,[],current
        final,first,traces=self.evidence.forward_steps(features,current,self.mode,hold)
        return final,first,traces,current

    def forward(self,features,selected):
        return self.forward_trace(features,selected)[0]
