import argparse,json,os,shutil,sys,traceback
from pathlib import Path
from types import SimpleNamespace
import runner
from recording import atomic_json,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    p.add_argument('--continuation',type=Path);a=p.parse_args();study=a.study.resolve()
    cfg=json.loads((study/'protocol.json').read_text());q=study/('queue_continuation.json' if a.continuation else 'queue.json')
    specs=json.loads(a.continuation.read_text())['runs'] if a.continuation else cfg['runs']
    if q.exists():raise FileExistsError('Queue exists; no implicit restart')
    state=dict(status='running',started_at=now(),pid=os.getpid(),current=None,completed=[]);atomic_json(q,state)
    try:
        for spec in specs:
            out=study/'runs'/spec['run_id'];out.mkdir(parents=True,exist_ok=False);source=out/'execution_source';source.mkdir()
            for f in (study/'scripts').glob('*.py'):shutil.copyfile(f,source/f.name)
            shutil.copyfile(study/'protocol.json',source/'protocol.json')
            state.update(current=spec,updated_at=now());atomic_json(q,state)
            kind=spec['kind'];inputs=[study/'protocol.json'];expect=[];argv=['--study',study,'--out',out]
            if kind=='verification':
                script='verify_saved.py';original=next(x for x in cfg['runs'] if x['kind']=='training' and x['arm']=='baseline')
                inputs.extend([study/'runs'/original['run_id']/'weights/last.pt',study/'runs'/original['run_id']/'results.csv'])
                expect=[out/'TRAINING_COMPLETE.json']
            elif kind=='preparation':script='prepare.py';inputs.extend([cfg['source_data'],cfg['selection']]);expect=[out/'COMPLETE.json']
            elif kind=='training':
                script='train_bridge.py';argv+=['--arm',spec['arm']];inputs.extend([cfg['weight'],study/'data/train.yaml',study/'data/train.txt'])
                expect=[out/'TRAINING_COMPLETE.json',out/'weights/last.pt',out/'auxiliary_epochs.json']
            elif kind=='evaluation':
                train=next(x for x in cfg['runs'] if x['kind']=='training' and x['arm']==spec['arm'])
                weight=study/'runs'/train['run_id']/'weights/last.pt';script='evaluate_bridge.py';argv+=['--arm',spec['arm'],'--weight',weight]
                inputs.extend([weight,study/'data/manifest.json',study/'data/panel.csv',study/'data/evaluate.yaml',Path(cfg['coco_root'])/'annotations/instances_val2017.json'])
                expect=[out/'COMPLETE.json',out/'metrics.json',out/'per_gt.csv']
            elif kind=='analysis':
                script='summarize.py';expect=[out/'COMPLETE.json',out/'RESULTS.md',out/'CONTRASTS.json']
                for r in cfg['runs']:
                    if r['kind']=='evaluation':inputs.extend([study/'runs'/r['run_id']/'metrics.json',study/'runs'/r['run_id']/'per_gt.csv'])
            else:raise ValueError(kind)
            args=SimpleNamespace(output=str(out),run_id=spec['run_id'],study=cfg['study_id'],cwd=str(source),
                input=list(map(str,inputs)),snapshot=list(map(str,source.glob('*.py')))+[str(source/'protocol.json')],expect=list(map(str,expect)),
                scope=json.dumps(dict(kind=kind,arm=spec.get('arm'),seed=cfg['seed'],epochs=cfg['epochs'],train_images=cfg['train_images'])),
                metrics=None,command=[sys.executable,'-u',str(source/script),*map(str,argv)])
            print('START',json.dumps(spec),flush=True);code=runner.execute(args)
            if code:raise RuntimeError(f'{spec} exited {code}')
            assert all(f.exists() for f in expect)
            state['completed'].append(spec['run_id']);atomic_json(q,state)
        state.update(status='completed',current=None,finished_at=now());atomic_json(q,state)
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at=now());atomic_json(q,state);traceback.print_exc();raise
if __name__=='__main__':main()
