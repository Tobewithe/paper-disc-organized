"""Paired image batches keep every within-image ownership edge intact."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from local_coefficient_head import LocalCoefficientHead
from ownership_ranking import coefficients,losses,objective
from frozen_mechanism_probe import sha,write_json
from three_region_probe import write_csv

FLOATS=['x','c','p','y','positive_p','pair_p']
INDICES=['pair_source','pair_neighbor','edges']

def load_data(cache):
    receipt=json.loads((cache/'COMPLETE.json').read_text())
    for name,digest in receipt['hashes'].items():
        if sha(cache/name)!=digest:raise RuntimeError(('cache receipt mismatch',name))
    hashes=json.loads((cache/'CACHE_HASHES.json').read_text());selection=json.loads((cache/'selection.json').read_text());records=[];excluded=[]
    for iid in selection['train']:
        path=cache/'train'/f'{iid}.npz'
        if sha(path)!=hashes[path.name]:raise RuntimeError(('training cache mismatch',iid))
        with np.load(path) as q:
            if not len(q['annotation_ids']):excluded.append(iid);continue
            item={k:torch.tensor(q[k],device='cuda',dtype=torch.float32) for k in FLOATS}
            item.update({k:torch.tensor(q[k],device='cuda',dtype=torch.long) for k in INDICES});item['positive_valid']=torch.tensor(q['positive_valid'],device='cuda',dtype=torch.bool)
            item['original_positive']=torch.einsum('nsk,nk->ns',item['positive_p'],item['c']).sigmoid().mean(1).detach();item['image_id']=iid
            records.append(item)
    if not records:raise RuntimeError('No trainable matched instances')
    x=torch.cat([r['x'] for r in records]);norm=dict(mean=x.mean(0),std=x.std(0,unbiased=False).clamp_min(.01))
    return records,norm,excluded

def batch_data(records,ids):
    selected=[records[int(j)] for j in ids];batch={k:torch.cat([r[k] for r in selected]) for k in [*FLOATS,'positive_valid','original_positive']}
    offsets=np.cumsum([0]+[len(r['c']) for r in selected[:-1]])
    for key in INDICES:batch[key]=torch.cat([r[key]+int(offset) for r,offset in zip(selected,offsets)])
    if len(batch['pair_source']) and int(torch.maximum(batch['pair_source'].max(),batch['pair_neighbor'].max()))>=len(batch['c']):raise RuntimeError('Pair index outside batch')
    return batch

def witness(records,norm,out):
    pairimage=next((i for i,r in enumerate(records) if len(r['pair_source'])),None)
    if pairimage is None:raise RuntimeError('Smoke needs a real ownership pair')
    batch=batch_data(records,[pairimage]);torch.manual_seed(0);model=LocalCoefficientHead().cuda()
    c=coefficients(batch['x'],batch['c'],model,norm)
    if not torch.equal(c,batch['c']):raise RuntimeError('Nonzero initial residual')
    parts=losses(c,batch);expected=F.binary_cross_entropy_with_logits(torch.einsum('nsk,nk->ns',batch['p'],c),batch['y'])+parts['dice']
    if not torch.equal(objective(parts,'bce_dice'),expected):raise RuntimeError('Base objective mismatch')
    # Both branches of a real directed pair must receive opposing ranking gradients.
    src=int(batch['pair_source'][0]);dst=int(batch['pair_neighbor'][0]);p=batch['pair_p'][:1].double();ci=batch['c'][src].double().clone().requires_grad_();cj=batch['c'][dst].double().clone().requires_grad_()
    margin=1-torch.einsum('psk,k->ps',p,ci)+torch.einsum('psk,k->ps',p,cj);value=F.softplus(margin).mean();gi,gj=torch.autograd.grad(value,[ci,cj])
    analytic=-(margin.sigmoid()[...,None]*p).mean((0,1));torch.testing.assert_close(gi,analytic,atol=1e-10,rtol=1e-10);torch.testing.assert_close(gi,-gj,atol=1e-10,rtol=1e-10)
    derivatives={}
    for name in ['bce','dice','rank','pair_bce','ccl']:
        model.zero_grad();c=coefficients(batch['x'],batch['c'],model,norm);parts=losses(c,batch);parts[name].backward();normg=float(torch.sqrt(sum(v.grad.square().sum() for v in model.parameters() if v.grad is not None)))
        if not np.isfinite(normg) or normg<=0:raise RuntimeError(('Absent gradient',name,normg))
        derivatives[name]=normg
    # Coverage must penalize a reduction in positive logits; margin is a surrogate.
    score=torch.tensor([[2.,1.]],device='cuda',requires_grad=True);reference=score.detach().sigmoid().mean(1)
    cov=F.relu(reference-(score-1).sigmoid().mean(1)-.0025).mean();grad=torch.autograd.grad(cov,score)[0]
    if not bool((grad<0).all()):raise RuntimeError('Coverage derivative sign')
    # One actual Adam update with every arm must stay finite and change the head.
    for arm in ['bce_dice','coverage_only','rank_only','rank_coverage','pair_bce_coverage','ccl_coverage']:
        torch.manual_seed(0);m=LocalCoefficientHead().cuda();opt=torch.optim.Adam(m.parameters(),lr=1e-4);c=coefficients(batch['x'],batch['c'],m,norm);loss=objective(losses(c,batch),arm);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10,error_if_nonfinite=True);opt.step()
        if not all(torch.isfinite(v).all() for v in m.parameters()):raise RuntimeError(('nonfinite witness',arm))
        if torch.equal(coefficients(batch['x'],batch['c'],m,norm),batch['c']):raise RuntimeError('Update inactive')
    write_json(out/'WITNESS.json',dict(status='PASS',real_pair_image=records[pairimage]['image_id'],pair_gradient_max_abs=float((gi-analytic).abs().max()),gradient_norms=derivatives,
        zero_residual_exact=True,base_objective_exact=True,both_competitors_receive_gradient=True,coverage_sign=True,finite_update_arms=6,batch_images_preserve_all_edges=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    cache=args.cache.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    configfile=Path(__file__).with_name('ownership_ranking_protocol_20260912.json');config=json.loads(configfile.read_text());torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records,norm,excluded=load_data(cache);witness(records,norm,out)
    torch.save({k:v.cpu() for k,v in norm.items()},out/'normalizer.pt');epochs=2 if args.smoke else config['epochs'];seeds=config['seeds'];arms=config['arms'];batchsize=config['batch_images']
    write_json(out/'protocol.json',dict(config=config,smoke=args.smoke,epochs=epochs,seeds=seeds,arms=arms,training='frozen-head-only',
        source_sha256=sha(__file__),dependencies={n:sha(Path(__file__).with_name(n)) for n in ['ownership_ranking.py','local_coefficient_head.py','ownership_ranking_protocol_20260912.json']},
        cache_complete_sha256=sha(cache/'COMPLETE.json'),cache_protocol_sha256=sha(cache/'protocol.json'),selection_sha256=sha(cache/'selection.json'),
        normalizer_sha256=sha(out/'normalizer.pt'),images_with_targets=len(records),images_without_trainable_targets=excluded,
        targets=sum(len(r['c']) for r in records),directed_pairs=sum(len(r['pair_source']) for r in records),ccl_edges=sum(len(r['edges']) for r in records)))
    started=time.monotonic();runrecords=[];orders_by_seed={};initial_by_seed={}
    for seed in seeds:
        for arm in arms:
            rng=np.random.default_rng(seed);torch.manual_seed(seed);model=LocalCoefficientHead().cuda();opt=torch.optim.Adam(model.parameters(),lr=config['lr'])
            init=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            if seed in initial_by_seed and init!=initial_by_seed[seed]:raise RuntimeError('Unpaired initialization')
            initial_by_seed[seed]=init;run=out/f'{arm}_s{seed}';run.mkdir();(run/'checkpoints').mkdir();history=[];orderhash=hashlib.sha256();updates=0
            for epoch in range(epochs):
                order=rng.permutation(len(records));orderhash.update(order.tobytes());sums={k:0. for k in ['loss','bce','dice','coverage','rank','pair_bce','ccl']};count=0;paircount=0;edgecount=0
                for pos in range(0,len(order),batchsize):
                    batch=batch_data(records,order[pos:pos+batchsize]);c=coefficients(batch['x'],batch['c'],model,norm);parts=losses(c,batch);loss=objective(parts,arm)
                    if not torch.isfinite(loss):raise RuntimeError(('nonfinite loss',arm,seed,epoch))
                    opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step();updates+=1
                    if not all(torch.isfinite(v).all() for v in model.parameters()):raise RuntimeError('nonfinite parameter')
                    n=len(c);count+=n;paircount+=len(batch['pair_source']);edgecount+=len(batch['edges']);sums['loss']+=float(loss.detach())*n
                    for k,v in parts.items():sums[k]+=float(v.detach())*n
                if paircount!=sum(len(r['pair_source']) for r in records):raise RuntimeError('Lost training pairs')
                row=dict(seed=seed,arm=arm,epoch=epoch+1,updates=updates,targets_seen=count,directed_pairs_seen=paircount,ccl_edges_seen=edgecount,
                    **{k:v/count for k,v in sums.items()},seconds=time.monotonic()-started);history.append(row);write_csv(run/'history.csv',history)
                state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),seed=seed,epoch=epoch+1,updates=updates,
                    numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),normalizer_sha256=sha(out/'normalizer.pt'))
                torch.save(state,run/'checkpoints'/f'epoch{epoch+1:02d}.pt')
                write_json(out/'progress.json',dict(phase='training',seed=seed,arm=arm,epoch=epoch+1,epochs=epochs,runs_complete=len(runrecords),total_runs=len(seeds)*len(arms),seconds=time.monotonic()-started))
                if (epoch+1)%5==0 or epoch+1==epochs:print(json.dumps(row),flush=True)
            digest=orderhash.hexdigest()
            if seed in orders_by_seed and digest!=orders_by_seed[seed]:raise RuntimeError('Unpaired data permutations')
            orders_by_seed[seed]=digest;checkpoint=run/'checkpoints'/f'epoch{epochs:02d}.pt'
            record=dict(arm=arm,seed=seed,epochs=epochs,updates=updates,checkpoint=str(checkpoint.relative_to(out)),sha256=sha(checkpoint),initialization_sha256=init,order_sha256=digest)
            runrecords.append(record);write_json(run/'COMPLETE.json',dict(status='COMPLETE',**record));write_json(out/'completed_runs.json',runrecords)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',runs=runrecords,seconds=time.monotonic()-started,epochs=epochs,normalizer_sha256=sha(out/'normalizer.pt'),
        hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

if __name__=='__main__':main()
