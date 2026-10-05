"""Run one new schedule arm, then evaluate and compare completed references."""
import argparse,json,os,shutil,sys,traceback
from pathlib import Path
from types import SimpleNamespace
import runner
from recording import atomic_json,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();study=a.study.resolve()
    cfg=json.loads((study/'protocol.json').read_text());q=study/'queue.json'
    if q.exists():raise FileExistsError('Existing queue: inspect it, never overwrite or duplicate')
    state=dict(status='running',started_at=now(),pid=os.getpid(),current=None,completed=[])
    atomic_json(q,state)
    try:
        for spec in cfg['runs']:
            out=study/'runs'/spec['run_id'];out.mkdir(parents=True,exist_ok=False)
            source=out/'execution_source';source.mkdir()
            for f in (study/'scripts').glob('*.py'):shutil.copyfile(f,source/f.name)
            shutil.copyfile(study/'protocol.json',source/'protocol.json')
            state.update(current=spec,updated_at=now());atomic_json(q,state)
            kind=spec['kind'];inputs=[study/'protocol.json'];expect=[]
            if kind=='schedule_check':
                script='check_schedule.py';argv=['--out',out];expect=[out/'COMPLETE.json']
            elif kind in ['smoke','training']:
                script='train_schedule.py';argv=['--study',study,'--out',out]
                if kind=='smoke':argv.append('--smoke')
                inputs.extend([cfg['weight'],cfg['smoke_data'] if kind=='smoke' else cfg['data'],cfg['train_list'],
                               Path(cfg['reference_remote_root'])/'runs'/cfg['reference_training_run']/'args.yaml'])
                expect=[out/'TRAINING_COMPLETE.json',out/'auxiliary_epochs.json',out/'results.csv',out/'weights/last.pt']
            elif kind=='evaluation':
                training=next(r for r in cfg['runs'] if r['kind']=='training')
                weight=study/'runs'/training['run_id']/'weights'/(spec['checkpoint']+'.pt')
                script='evaluate_checkpoint.py';argv=['--study',study,'--out',out,'--weight',weight,'--arm','reg_scheduled','--checkpoint',spec['checkpoint']]
                inputs.extend([weight,cfg['selection'],Path(cfg['coco_root'])/'annotations/instances_val2017.json'])
                expect=[out/'COMPLETE.json',out/'official_metrics.csv',out/'targeted_per_instance.csv']
            elif kind=='analysis':
                script='summarize_schedule.py';argv=['--study',study,'--out',out]
                for ref in cfg['reference_evaluations']:
                    folder=Path(cfg['reference_remote_root'])/'runs'/ref['run_id']
                    inputs.extend([folder/'official_metrics.csv',folder/'targeted_per_instance.csv'])
                for ev in [r for r in cfg['runs'] if r['kind']=='evaluation']:
                    folder=study/'runs'/ev['run_id'];inputs.extend([folder/'official_metrics.csv',folder/'targeted_per_instance.csv'])
                expect=[out/'COMPLETE.json',out/'RESULTS.md',out/'official_contrasts.csv',out/'targeted_contrasts.csv']
            else:raise ValueError(kind)
            args=SimpleNamespace(output=str(out),run_id=spec['run_id'],study=cfg['study_id'],cwd=str(source),
                                 input=list(map(str,inputs)),snapshot=list(map(str,source.glob('*.py')))+[str(source/'protocol.json')],
                                 expect=list(map(str,expect)),scope=json.dumps(dict(kind=kind,arm='reg_scheduled',seed=cfg['seed'],checkpoint=spec.get('checkpoint'))),
                                 metrics=None,command=[sys.executable,'-u',str(source/script),*map(str,argv)])
            print('START',json.dumps(spec),flush=True)
            code=runner.execute(args)
            if code:raise RuntimeError(f'{kind} failed: {spec["run_id"]}, exit {code}')
            if not all(p.exists() for p in expect):raise RuntimeError('Missing expected outputs')
            state['completed'].append(spec['run_id']);state['updated_at']=now();atomic_json(q,state)
        state.update(status='completed',current=None,finished_at=now());atomic_json(q,state)
        print('QUEUE_COMPLETE',state['finished_at'],flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at=now());atomic_json(q,state);traceback.print_exc();raise

if __name__=='__main__':main()
