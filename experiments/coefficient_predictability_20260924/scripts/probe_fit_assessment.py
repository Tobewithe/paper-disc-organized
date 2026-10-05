"""Compare original, dev-selected and final checkpoints on fixed fit/dev pixels."""
import argparse
import json
from pathlib import Path
import torch
from official_pipeline import setup, load, split_metadata, predict, target_rois, roi_losses, write
from feature_probes import Probe


def main(a):
    setup()
    index=json.loads((a.bank/'INDEX.json').read_text())
    meta,stats=split_metadata(a.bank,index)
    history=json.loads((a.probe/'TRAINING.json').read_text())
    models={}
    for arm, state in history.items():
        models[arm]={}
        for label,filename in [('dev_selected',f'{arm}_best.pt'),('last',f'{arm}_epoch{state["history"][-1]["epoch"]}.pt')]:
            net=Probe(arm,*stats).cuda().eval()
            net.load_state_dict(load(a.probe/filename)['state_dict'])
            models[arm][label]=net
    result={}
    for group in ('fit','dev'):
        totals={'original':0.0}
        totals.update({f'{arm}:{label}':0.0 for arm in models for label in models[arm]})
        n=0
        with torch.no_grad():
            for step,item in enumerate(meta[group],1):
                image=load(a.bank/'images'/f'{item["image_id"]:012d}.pt')
                rois=target_rois(image)
                ids=[r['raw_id'] for r in image['rows']]
                values=[image['coeff'][ids].cuda()]
                for arm in models:
                    for label,net in models[arm].items():
                        cc,_=predict({arm:net},image,item)
                        values.append(cc[0])
                loss=roi_losses(torch.stack(values),rois)*image['segmentation_gain']
                for key,val in zip(totals,loss.tolist()): totals[key]+=val
                n+=len(ids)
                if step%100==0: print(json.dumps(dict(group=group,images=step,total=len(meta[group]))),flush=True)
        result[group]=dict(n=n,loss={k:v/n for k,v in totals.items()})
    result['dev_selection_including_unmodified']={arm:('original' if result['dev']['loss']['original'] <= result['dev']['loss'][arm+':dev_selected'] else 'dev_selected') for arm in models}
    write(a.out/'SUMMARY.json',result)
    write(a.out/'COMPLETE.json',dict(groups=['fit','dev'],arms=list(models)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('bank','probe','out'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);main(args)
