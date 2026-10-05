"""Detached sequential queue: smoke, three trainings, six evaluations, summary."""
import argparse,json,os,sys,traceback
from pathlib import Path
from recording import atomic_json,now,record_run

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--continue-queue',action='store_true');a=p.parse_args();study=a.study.resolve()
    cfg=json.loads((study/'protocol.json').read_text());runs=cfg['runs'];qpath=study/'queue.json'
    if qpath.exists():
        if not a.continue_queue:raise FileExistsError('Queue already exists; use an explicit continuation after resolving failure')
        state=json.loads(qpath.read_text())
        assert state['status']=='failed','Only a failed, inactive queue can be continued'
        atomic_json(study/'queue_history'/('attempt_'+str(state['pid'])+'.json'),state)
        state.update(status='running',resumed_at=now(),pid=os.getpid(),current=None)
        state.pop('finished_at',None);state.pop('error',None)
    else:
        state=dict(status='running',started_at=now(),pid=os.getpid(),completed=[],current=None)
    atomic_json(qpath,state)
    def run(spec,script,argv,inputs):
        if spec['run_id'] in state['completed']:
            out=study/'runs'/spec['run_id']
            assert json.loads((out/'run.json').read_text())['status']=='completed'
            return out
        state['current']=spec;state['updated_at']=now();atomic_json(qpath,state)
        print('START',json.dumps(spec),flush=True)
        out=record_run(study,spec,script,argv,inputs)
        state['completed'].append(spec['run_id']);state['updated_at']=now();atomic_json(qpath,state)
        print('DONE',spec['run_id'],flush=True)
        return out
    try:
        prep=next(r for r in runs if r['kind']=='preparation');prepdir=study/'runs'/prep['run_id']
        run(prep,'prepare_data.py',['--study',study,'--out',prepdir],[Path(cfg['coco_root'])/'coco_full.yaml'])
        for spec in [r for r in runs if r['kind'] in ['smoke','training']]:
            smoke=spec['kind']=='smoke';arm='reg_only' if smoke else spec['arm'];out=study/'runs'/spec['run_id']
            data=prepdir/'data'/('smoke.yaml' if smoke else 'data.yaml')
            argv=['--study',study,'--out',out,'--data',data,'--arm',arm]
            if smoke:argv.append('--smoke')
            run(spec,'train_ablation.py',argv,[cfg['weight'],data])
        training={r['arm']:r for r in runs if r['kind']=='training'}
        # Fixed last checkpoints are evaluated first across all arms.
        for checkpoint in ['last','best']:
            for spec in [r for r in runs if r['kind']=='evaluation' and r['checkpoint']==checkpoint]:
                weight=study/'runs'/training[spec['arm']]['run_id']/'weights'/(checkpoint+'.pt');out=study/'runs'/spec['run_id']
                run(spec,'evaluate_checkpoint.py',['--study',study,'--out',out,'--weight',weight,'--arm',spec['arm'],
                     '--checkpoint',checkpoint],[weight,cfg['selection'],Path(cfg['coco_root'])/'annotations/instances_val2017.json'])
        summary=next(r for r in runs if r['kind']=='analysis');out=study/'runs'/summary['run_id']
        inputs=[study/'runs'/r['run_id']/'official_metrics.csv' for r in runs if r['kind']=='evaluation']
        run(summary,'summarize.py',['--study',study,'--out',out],inputs)
        state.update(status='completed',current=None,finished_at=now());atomic_json(qpath,state)
    except BaseException as error:
        state.update(status='failed',finished_at=now(),error=repr(error));atomic_json(qpath,state)
        traceback.print_exc();raise

if __name__=='__main__':main()
