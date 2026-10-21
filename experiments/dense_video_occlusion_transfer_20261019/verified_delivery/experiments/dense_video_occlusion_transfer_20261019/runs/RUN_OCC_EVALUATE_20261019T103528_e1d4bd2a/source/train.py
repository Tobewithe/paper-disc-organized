import copy,time
import numpy as np
import torch
from torch.nn import functional as F
from .common import ROOT,Run,read,save,sha
from .model import VisibilityHead,windows

SEED=20261019
ARMS=['FULL','NOW','NOCTX']

def dataset(entries):
 xs=[];ys=[];vis=[];ids=[];offset=0
 for e in entries:
  p=ROOT/e['path'];assert sha(p)==e['sha256'];d=np.load(p);xs.append(d['x']);ys.append(d['y']);vis.append(d['visibility']);ids.append(d['ids']+offset);offset+=int(d['ids'].max())+1
 x=torch.from_numpy(np.concatenate(xs));y=torch.from_numpy(np.concatenate(ys)).long();v=torch.from_numpy(np.concatenate(vis));index=windows(x,np.concatenate(ids))
 return x.cuda(),y.cuda(),v.cuda(),index.cuda()

@torch.inference_mode()
def assess(model,arm,data):
 x,y,v,index=data;conf=np.zeros((4,4),int);mae=0.;preds=[]
 for i in range(0,len(x),512):
  logits,value=model(x[index[i:i+512]],arm);pred=logits.argmax(-1);yy=y[i:i+512]
  conf+=np.bincount((yy*4+pred).cpu().numpy(),minlength=16).reshape(4,4);mae+=float((value-v[i:i+512]).abs().sum());preds.append({'start':i,'prob':logits.softmax(-1).cpu().numpy(),'visibility':value.cpu().numpy()})
 f1=2*np.diag(conf)/np.maximum(1,conf.sum(0)+conf.sum(1))
 return {'macro_F1':float(f1.mean()),'class_F1':f1.tolist(),'confusion':conf.tolist(),'visibility_MAE':mae/len(x),'samples':len(x)},preds

def main():
 torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False
 pointer=read(ROOT/'features_pointer.json');fp=__import__('pathlib').Path(pointer['path']);catalog=read(fp/'FEATURE_MANIFEST.json')
 contract={'arms':ARMS,'seed':SEED,'steps_per_arm':2000,'smoke_steps_per_arm':20,'batch_size':128,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'visibility_loss':'.5*SmoothL1 including out=0','checkpoint':'last only','input_manifest_sha256':sha(fp/'FEATURE_MANIFEST.json'),'real_GT_training':False,'feature_run':pointer}
 with Run('train',contract) as run:
  sets={s:dataset([e for e in catalog if e['kind']=='synthetic' and e['split']==s]) for s in ['train','validation','synthetic_test']}
  x,y,v,index=sets['train'];counts=torch.bincount(y,minlength=4).float();cw=(counts.sum()/torch.clamp(counts,min=1)/4).clamp(max=5);cw[counts==0]=0
  model=VisibilityHead().cuda();initial=copy.deepcopy(model.state_dict());torch.save(initial,run.path/'initial.pt')
  parameter_count=sum(p.numel() for p in model.parameters());save(run.path/'MODEL.json',{'trainable_parameters':parameter_count,'input_dim':392,'class_counts':counts,'class_weights':cw,'initial_sha256':sha(run.path/'initial.pt'),'capacity_matched':True})
  rng=np.random.default_rng(SEED);batches=torch.from_numpy(rng.integers(0,len(x),size=(2000,128))).cuda();save(run.path/'BATCH_MANIFEST.json',{'sha256':__import__('hashlib').sha256(batches.cpu().numpy().tobytes()).hexdigest(),'shape':[2000,128]})
  results={};smoke=[]
  run.meta['smoke_training_steps']=0
  for arm in ARMS:
   model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
   for step in range(20):
    ix=batches[step];logit,value=model(x[index[ix]],arm);loss=F.cross_entropy(logit,y[ix],weight=cw)+.5*F.smooth_l1_loss(value,v[ix]);opt.zero_grad();loss.backward();assert torch.isfinite(loss);assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters());torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();run.meta['smoke_training_steps']+=1
   smoke.append({'arm':arm,'steps':20,'loss':float(loss),'all_parameters_nonzero_finite_gradients':True});print('TRAIN_SMOKE',smoke[-1],flush=True)
  save(run.path/'SMOKE.json',smoke)
  for arm in ARMS:
   model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);curve=[];start=time.perf_counter();model.train()
   for step in range(2000):
    ix=batches[step];logit,value=model(x[index[ix]],arm);classification=F.cross_entropy(logit,y[ix],weight=cw);regression=F.smooth_l1_loss(value,v[ix]);loss=classification+.5*regression
    opt.zero_grad();loss.backward();assert torch.isfinite(loss);torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();run.meta['training_steps']+=1
    if step%50==0 or step==1999:
     row={'step':step+1,'loss':float(loss),'classification':float(classification),'visibility':float(regression)};curve.append(row);print('TRAIN',arm,row,flush=True)
   model.eval();ck=run.path/(arm+'.pt');torch.save({'state':model.state_dict(),'arm':arm,'steps':2000,'seed':SEED,'trainable_parameters':parameter_count,'initial_sha256':sha(run.path/'initial.pt'),'features_manifest_sha256':contract['input_manifest_sha256']},ck)
   evaluations={}
   for split in ['validation','synthetic_test']:
    result,preds=assess(model,arm,sets[split]);evaluations[split]=result
    np.savez_compressed(run.path/(arm+'_'+split+'_predictions.npz'),prob=np.concatenate([p['prob'] for p in preds]),visibility=np.concatenate([p['visibility'] for p in preds]),y=sets[split][1].cpu().numpy(),truth_visibility=sets[split][2].cpu().numpy())
   results[arm]={'checkpoint':str(ck),'sha256':sha(ck),'steps':2000,'seconds':time.perf_counter()-start,'evaluation':evaluations};save(run.path/(arm+'_curve.json'),curve);save(run.path/'SUMMARY.json',results);print('TRAIN_ARM_DONE',arm,results[arm],flush=True)
  run.meta['training_steps']=6000;run.meta['smoke_training_steps']=60;run.meta['scientific_status']='pending_real_transfer_evaluation'
  save(run.path/'DECISION.json',{'synthetic_capability':results['FULL']['evaluation']['synthetic_test']['macro_F1']>=.65,'next':'fixed real-video baselines and ablations; no checkpoint selection'})
