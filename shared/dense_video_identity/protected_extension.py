"""Appearance completion with invariant protection of existing high-score motion edges."""
from pathlib import Path
import cv2
from .common import Run,read_json,write_json,setup_paths,load_gt,sha256
from .learn import load_cache,load_identity,embed_array
from .conflict_selector import sequence
from .trajectory_selector import load_trajectory
from .evaluate import official_metrics,metric_data,compact
from .infer import extract


def smoke(root):
    setup_paths(root);cfg=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json');model,ck=load_identity(root)
    with Run(root,'protected_smoke',dict(frames=32,training=0,checks=['exact motion baseline','one-to-one','only empty-base additions','protected high-score edges'])) as run:
        for name in ['pig_4','horse_5']:
            seq=next(s for s in manifest['sequences'] if s['name']==name);records,raw=load_cache(root,seq);z=embed_array(model,ck,raw)
            base,_=sequence('motion',records[:16],z,cfg)
            ref=read_json(Path(read_json(root/'v1_reference.json')['dev'])/'bytetrack'/name/'tracks.json')[:16]
            assert base==ref
            for mode in ['extension','extension_trajectory']:
                out,events=sequence(mode,records[:16],z,cfg,lambda x:.8,.7)
                assert all(not e['base'] for e in events if e['selected_appearance'])
                for f in out:
                    assert len({t['identity'] for t in f['tracks']})==len(f['tracks'])
        write_json(run.path/'SUMMARY.json',dict(status='passed',frames=32,baseline_exact=True))


def prepare(root):
    manifest=read_json(root/'input_manifest.json');names=['zebra_3','zebra_4']
    assert not any(s['name'] in names for s in manifest['sequences'])
    with Run(root,'protected_confirmation_registration',dict(sequences=names,selection='next two preselected complete sequences',new_training=0)) as run:
        write_json(run.path/'input_manifest_before.json',manifest)
        for name in names:
            video=root/'data'/(name+'.mp4');gt=root/'data'/(name+'_gt.txt');cap=cv2.VideoCapture(str(video));assert cap.isOpened()
            seq=dict(name=name,split='protected_confirm',model='horse',classes=[22],video=str(video),gt=str(gt),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),fps=cap.get(cv2.CAP_PROP_FPS),width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),video_sha256=sha256(video),gt_sha256=sha256(gt),gt_kind='official_AnimalTrack_bbox_identity',visibility_labels=False)
            cap.release();manifest['sequences'].append(seq);print('PROTECTED_CONFIRM',seq,flush=True)
        write_json(root/'input_manifest.json',manifest);write_json(run.path/'input_manifest_after.json',manifest)
    extract(root,read_json(root/'config.json')['detector'],only_names=names,pointer_name='protected_confirmation_features_pointer.json')
    ptr=read_json(root/'protected_confirmation_features_pointer.json')
    for seq in manifest['sequences']:
        if seq['name'] in names:seq['cache_path']=str(root/ptr['cache']/seq['name'])
    write_json(root/'input_manifest.json',manifest)


def evaluate(root,split):
    setup_paths(root);cfg=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json');model,ck=load_identity(root);metrics=official_metrics(root)
    modes=['extension','extension_trajectory'] if split!='protected_confirm' else ['motion','all_appearance','extension','extension_trajectory']
    predictor,threshold=load_trajectory(root)
    with Run(root,'protected_evaluate_'+split,dict(split=split,modes=modes,new_training=0,protected='all first-stage motion assignment edges',features=read_json(root/'identity_pointer.json'),ET=read_json(root/'trajectory_selector_pointer.json'),GT='evaluation only')) as run:
        raw={m:{} for m in modes};table=[]
        for seq in [s for s in manifest['sequences'] if s['split']==split]:
            records,rawx=load_cache(root,seq);z=embed_array(model,ck,rawx);gt=load_gt(root,seq) if seq.get('gt') else None
            for mode in modes:
                out,events=sequence(mode,records,z,cfg,predictor if mode=='extension_trajectory' else None,threshold)
                dest=run.path/mode/seq['name'];dest.mkdir(parents=True);write_json(dest/'tracks.json',out);write_json(dest/'decisions.json',events)
                stat=dict(sequence=seq['name'],arm=mode,groups=len(events),interventions=sum(e['selected_appearance'] for e in events),frames=seq['frames'],protected_edge_assertions_passed=mode in {'extension','extension_trajectory'})
                if gt is not None:
                    data=metric_data(gt,out);rs={k:m.eval_sequence(data) for k,m in metrics.items()};raw[mode][seq['name']]=rs;stat.update(compact(rs));write_json(dest/'metric_details.json',rs)
                table.append(stat);print('PROTECTED_EVAL',stat,flush=True)
        combined={m:compact({k:metric.combine_sequences({s:rs[k] for s,rs in raw[m].items()}) for k,metric in metrics.items()}) for m in modes} if split!='demo' else {}
        write_json(run.path/'SUMMARY.json',dict(split=split,by_sequence=table,combined=combined))
        write_json(root/('protected_'+split+'_pointer.json'),dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','prepare','evaluate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--split',default='dev');a=p.parse_args()
    {'smoke':lambda:smoke(a.root),'prepare':lambda:prepare(a.root),'evaluate':lambda:evaluate(a.root,a.split)}[a.stage]()
