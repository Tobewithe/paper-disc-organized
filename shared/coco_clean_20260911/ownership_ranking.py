"""Loss-only ownership trial. The deployable path has no GT or pair inputs."""
import torch
import torch.nn.functional as F
from local_coefficient_head import LocalCoefficientHead

def input_features(item,device='cuda'):
    h,w=map(int,item['input_shape']);base=torch.as_tensor(item['features'],device=device).float()
    level=torch.as_tensor(item['level'],device=device).long();boxes=torch.as_tensor(item['boxes'],device=device).float()
    scaled=boxes/boxes.new_tensor([w,h,w,h]);wh=(scaled[:,2:]-scaled[:,:2]).clamp_min(1e-5)
    return torch.cat([base,F.one_hot(level,3).float(),scaled,wh],1)

def coefficients(x,original,model,norm):
    return original+model((x-norm['mean'])/norm['std']).mean(1)

def losses(c,data):
    z=torch.einsum('nsk,nk->ns',data['p'],c);y=data['y'].float();prob=z.sigmoid()
    bce=F.binary_cross_entropy_with_logits(z,y)
    dice=(1-(2*(prob*y).sum(1)+1)/(prob.sum(1)+y.sum(1)+1)).mean()
    positive=torch.einsum('nsk,nk->ns',data['positive_p'],c).sigmoid().mean(1)
    valid=data['positive_valid'];zero=z.sum()*0
    coverage=F.relu(data['original_positive']-positive-.0025)[valid].mean() if valid.any() else zero
    src,dst=data['pair_source'],data['pair_neighbor']
    if len(src):
        zi=torch.einsum('psk,pk->ps',data['pair_p'],c[src]);zj=torch.einsum('psk,pk->ps',data['pair_p'],c[dst])
        rank=F.softplus(1-zi+zj).mean();pair_bce=(F.softplus(-zi)+F.softplus(zj)).mean()/2
    else:rank=pair_bce=zero
    edges=data['edges']
    ccl=F.relu(F.cosine_similarity(c[edges[:,0]],c[edges[:,1]],dim=1)-.1).mean() if len(edges) else zero
    return dict(bce=bce,dice=dice,coverage=coverage,rank=rank,pair_bce=pair_bce,ccl=ccl)

def objective(parts,arm):
    value=parts['bce']+parts['dice']
    if arm in ['coverage_only','rank_coverage','pair_bce_coverage','ccl_coverage']:value=value+parts['coverage']
    if arm in ['rank_only','rank_coverage']:value=value+.1*parts['rank']
    if arm=='pair_bce_coverage':value=value+.1*parts['pair_bce']
    if arm=='ccl_coverage':value=value+.1*parts['ccl']
    return value
