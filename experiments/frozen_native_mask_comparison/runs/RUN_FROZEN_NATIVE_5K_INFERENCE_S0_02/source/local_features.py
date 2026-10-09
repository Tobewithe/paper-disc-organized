"""Prediction-only response statistics; no annotations or GT arguments."""
import torch
import torch.nn.functional as F


def local_features(logits, base, trial):
    values=logits.clamp(-20,20)
    positive=base.bool(); kept=trial.bool(); removed=positive & ~kept
    erosion=-F.max_pool2d(-F.pad(base[:,None].float(),(1,1,1,1),value=0),3,stride=1)[:,0]
    boundary=positive & ~erosion.bool()
    near_kept=F.max_pool2d(kept[:,None].float(),3,stride=1,padding=1)[:,0].bool()
    def count(region):return region.sum((1,2)).float()
    def moments(region):
        n=count(region).clamp_min(1)
        mean=(values*region).sum((1,2))/n
        var=(values.square()*region).sum((1,2))/n-mean.square()
        return [mean,var.clamp_min(0).sqrt()]
    nr=count(removed).clamp_min(1);nb=count(positive).clamp_min(1)
    columns=moments(removed)+moments(kept)+moments(boundary)+[
        count(removed & boundary)/nr,count(removed & near_kept)/nr,count(boundary)/nb]
    columns += [count(positive & (logits<=t))/nb for t in [.25,.75,1.0]]
    return torch.stack(columns,1).cpu().numpy().astype('float64')
