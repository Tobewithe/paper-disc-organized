"""Online native coefficient-head training; official data, assignments and base loss."""
import argparse, copy, hashlib, json, os, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('WANDB_MODE','disabled')
import torch
from torch.nn import functional as F
import ultralytics
from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import unwrap_model
from coefficient_objective import objective, solve, self_test

ACTIVE=None
ORIGINAL_LOSS=SegmentationModel.loss
CONFIG={}


def save_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str));tmp.replace(path)


def frozen_digest(model):
    h=hashlib.sha256();prefix=f'model.{len(model.model)-1}.cv4.'
    for k,v in model.state_dict().items():
        if k.startswith(prefix):continue
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class TrainingLoss:
    def __init__(self,model,out,mode):
        self.model=model;self.out=out;self.mode=mode;self.step=0;self.assignment=None;self.native_items=None
        model.criterion=model.init_criterion()
        assert hasattr(model.criterion,'one2many')
        self.criterion=model.criterion
        self.reference=copy.deepcopy(model.model[-1].cv4).eval().requires_grad_(False)
        crit=self.criterion.one2many
        original_assign=crit.get_assigned_targets_and_loss
        def capture(pred,batch):
            result=original_assign(pred,batch);self.assignment=result[0];return result
        crit.get_assigned_targets_and_loss=capture
        original_many=crit.loss
        def capture_many(pred,batch):
            result=original_many(pred,batch);self.native_items=result[1];return result
        crit.loss=capture_many
        self.log=open(out/'batch_metrics.jsonl','a',buffering=1)
        self.images_log=open(out/'batch_identity.jsonl','a',buffering=1)

    def auxiliary(self,p,batch):
        fg,gtidx,boxes,_,_=self.assignment
        masks=batch['masks'].to(fg.device)
        assert masks.ndim==3 and self.model.args.overlap_mask and self.model.args.mask_ratio==1
        proto=p['proto'][0] if isinstance(p['proto'],tuple) else p['proto']
        coefficients=p['mask_coefficient'].permute(0,2,1)
        with torch.no_grad():
            refs=torch.cat([self.reference[k](x.float()).flatten(2) for k,x in enumerate(p['feats'])],2).permute(0,2,1)
        generator=torch.Generator(device=fg.device).manual_seed(721903+self.step)
        points=[];targets=[];student=[];reference=[];group=[]
        h,w=masks.shape[-2:];group_index=0
        for bi in range(len(fg)):
            positive=fg[bi].nonzero().flatten()
            for gi in gtidx[bi,positive].unique(sorted=True):
                ids=positive[gtidx[bi,positive]==gi]
                ids=ids[torch.randperm(len(ids),device=fg.device,generator=generator)[:2]]
                box=boxes[bi,ids[0]].detach()
                lo=box[:2].ceil().long();hi=box[2:].ceil().long()
                lo[0].clamp_(0,w);lo[1].clamp_(0,h);hi[0].clamp_(0,w);hi[1].clamp_(0,h)
                if bool((hi<=lo).any()):continue
                xs=torch.randint(int(lo[0]),int(hi[0]),(CONFIG['points'],),device=fg.device,generator=generator)
                ys=torch.randint(int(lo[1]),int(hi[1]),(CONFIG['points'],),device=fg.device,generator=generator)
                grid=torch.stack((2*(xs.float()+.5)/w-1,2*(ys.float()+.5)/h-1),-1)[None,None]
                pp=F.grid_sample(proto[bi:bi+1].detach().float(),grid,align_corners=False,padding_mode='border')[0,:,0].T
                yy=(masks[bi,ys,xs]==(gi+1)).float()
                for rid in ids:
                    points.append(pp);targets.append(yy);student.append(coefficients[bi,rid].float());reference.append(refs[bi,rid].float());group.append(group_index)
                group_index+=1
        if not student:return coefficients.sum()*0,{'aux_gt':0,'aux_candidates':0}
        pmat=torch.stack(points);y=torch.stack(targets);c=torch.stack(student);refc=torch.stack(reference)
        z=torch.einsum('nsk,nk->ns',pmat,c)
        ref=torch.einsum('nsk,nk->ns',pmat,refc).detach()
        beta,rho=CONFIG['beta'],CONFIG['rho']
        info={'aux_gt':group_index,'aux_candidates':len(c)}
        if self.mode=='B':
            per=objective(z,y,ref,beta,rho)
        else:
            ct,diag=solve(pmat,c.detach(),y,ref,CONFIG['teacher_steps'],beta,rho,CONFIG['damping'])
            teacher=torch.einsum('nsk,nk->ns',pmat,ct).detach()
            teacher_prob=teacher.sigmoid()
            per=(F.binary_cross_entropy_with_logits(z,teacher_prob,reduction='none')-
                 F.binary_cross_entropy_with_logits(teacher,teacher_prob,reduction='none')).mean(-1)
            def iou(v):
                pred=v>0;truth=y>.5
                return (pred&truth).sum(-1)/(pred|truth).sum(-1).clamp_min(1)
            old,new=iou(z.detach()),iou(teacher)
            info.update(teacher_accepted=int(diag['accepted'].sum()),teacher_F_before=float(diag['before'].mean()),
                        teacher_F_after=float(diag['after'].mean()),teacher_sample_iou_before=float(old.mean()),
                        teacher_sample_iou_after=float(new.mean()),teacher_sample_repairs=int(((old<.75)&(new>=.75)).sum()),
                        teacher_sample_damages=int(((old>=.75)&(new<.75)).sum()))
        group=torch.tensor(group,device=c.device)
        sums=torch.zeros(group_index,device=c.device).scatter_add_(0,group,per)
        count=torch.bincount(group,minlength=group_index).float()
        return (sums/count).mean(),info

    def __call__(self,batch,preds=None):
        if preds is None:preds=self.model.forward(batch['img'])
        base,_=self.criterion(preds,batch)
        assert torch.isfinite(base).all(), 'nonfinite official loss'
        info={'step':self.step,'native_mask_loss':float(self.native_items[1]),'o2m_weight':self.criterion.o2m}
        if self.mode!='A':
            with torch.autocast('cuda',enabled=False):aux,extra=self.auxiliary(preds['one2many'],batch)
            assert torch.isfinite(aux),'nonfinite auxiliary loss'
            # Match official mask gain and branch schedule; lambda is fixed in protocol.
            base=base.clone();base[1]=base[1]+CONFIG['lambda']*self.model.args.box*self.criterion.o2m*len(batch['img'])*aux
            info.update(extra,aux=float(aux.detach()))
        self.log.write(json.dumps(info)+'\n')
        # Confirm training order and transformed tensors agree across paired conditions.
        image_hash=hashlib.sha256(batch['img'].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        self.images_log.write(json.dumps({'step':self.step,'files':batch.get('im_file'), 'input_sha256':image_hash})+'\n')
        self.step+=1
        return base,self.native_items


def patched_loss(self,batch,preds=None):
    if ACTIVE is not None and self is ACTIVE.model:return ACTIVE(batch,preds)
    return ORIGINAL_LOSS(self,batch,preds)


class CoefficientTrainer(SegmentationTrainer):
    def build_optimizer(self,model,**kwargs):
        m=unwrap_model(model);m.requires_grad_(False);m.model[-1].cv4.requires_grad_(True)
        # Build with official optimizer implementation on exactly the trainable subnetwork.
        return super().build_optimizer(m.model[-1].cv4,**kwargs)

    def _model_train(self):
        super()._model_train()
        for module in self.model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()

    def _setup_train(self):
        global ACTIVE
        super()._setup_train()
        m=unwrap_model(self.model)
        names=[n for n,p in m.named_parameters() if p.requires_grad]
        assert names and all(f'model.{len(m.model)-1}.cv4.' in n for n in names),names
        self.frozen_start=frozen_digest(m)
        ACTIVE=TrainingLoss(m,CONFIG['out'],CONFIG['mode'])
        save_json(CONFIG['out']/'SETUP.json',{'trainable_names':names,'parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
                  'batch':self.batch_size,'train_images':len(self.train_loader.dataset),'frozen_start':self.frozen_start,
                  'reference':'fixed initial native cv4, same image features; all BN running stats frozen',
                  'ultralytics':ultralytics.__version__,'torch':torch.__version__,'args':vars(self.args)})

    def validate(self):
        # Independent evaluation jobs use original COCO masks, never train-converted labels.
        return {},0.0

    def save_model(self):
        m=unwrap_model(self.model);state={k:v.detach().cpu() for k,v in m.model[-1].cv4.state_dict().items()}
        ema={k:v.detach().cpu() for k,v in self.ema.ema.model[-1].cv4.state_dict().items()} if self.ema else state
        result={'cv4':state,'ema_cv4':ema,'epoch':self.epoch+1,'mode':CONFIG['mode'],'seed':self.args.seed,
                'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'scaler':self.scaler.state_dict(),
                'criterion_updates':m.criterion.updates,'o2m':m.criterion.o2m,'rng_torch':torch.get_rng_state(),
                'base_weights':CONFIG['weights'],'config':{k:v for k,v in CONFIG.items() if k!='out'}}
        path=CONFIG['out']/f'epoch{self.epoch+1}.pt';torch.save(result,path)
        save_json(CONFIG['out']/'progress.json',{'epoch':self.epoch+1,'epochs':self.epochs,'batch_steps':ACTIVE.step,'checkpoint':str(path),
                  'mask_loss':float(self.tloss[1]),'elapsed_s':time.time()-self.train_time_start})

    def final_eval(self):
        return None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--weights',type=Path,required=True)
    ap.add_argument('--mode',choices=['A','B','C'],required=True);ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();assert ultralytics.__version__=='8.4.100'
    out=Path(os.environ['RESEARCH_RUN_DIRECTORY']);protocol=json.loads((a.root/'PROTOCOL.json').read_text())
    CONFIG.update(protocol['auxiliary']);CONFIG.update(out=out,mode=a.mode,weights=str(a.weights))
    torch.set_num_threads(4);settings.update({'wandb':False,'tensorboard':False,'mlflow':False,'comet':False})
    if a.smoke:save_json(out/'OBJECTIVE_TEST.json',self_test())
    yolo=YOLO(str(a.weights));saved=yolo.ckpt['train_args'];args={k:v for k,v in saved.items() if k in DEFAULT_CFG_DICT}
    args.update(model=str(a.weights),data=str(a.root/'data'/('smoke.yaml' if a.smoke else 'pilot.yaml')),
                epochs=a.epochs,batch=protocol['batch'],workers=protocol['workers'],device=0,cache=False,
                project=str(out),name='trainer',exist_ok=True,save=True,save_period=1,plots=False,
                val=False,save_json=False,patience=1000,resume=False,seed=0,verbose=False)
    save_json(out/'CONFIG.json',{'source_weight_train_args':saved,'resolved_overrides':args,'unsupported_saved_keys':sorted(set(saved)-set(DEFAULT_CFG_DICT)),
             'method':CONFIG,'note':'8-epoch diagnostic budget; otherwise supported saved checkpoint training settings.'})
    SegmentationModel.loss=patched_loss
    trainer=CoefficientTrainer(overrides=args);trainer.model=yolo.model
    trainer.train()
    final=frozen_digest(unwrap_model(trainer.model));assert final==trainer.frozen_start, 'non-coefficient model state changed'
    summary={'status':'completed','mode':a.mode,'epochs':a.epochs,'steps':ACTIVE.step,'finite':True,'frozen_state_unchanged':True,
             'checkpoint':f'epoch{a.epochs}.pt','primary_evaluation':'final epoch EMA cv4 inserted into original model; one-to-many + NMS'}
    save_json(out/'COMPLETE.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
