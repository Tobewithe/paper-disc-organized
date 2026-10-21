"""Read-only GT attribution of actual protected additions; no labels enter inference."""
from pathlib import Path
from collections import Counter
from .common import Run,read_json,write_json,setup_paths,load_gt
from .learn import load_cache,match_gt


def main(root):
    setup_paths(root);manifest=read_json(root/'input_manifest.json')
    with Run(root,'connection_outcome_audit',dict(arms=['extension','extension_trajectory'],GT='posthoc evaluation only',event='selected empty-base appearance addition',matching='fixed detection-GT Hungarian IoU>=.5',cause_scope='local GT identity continuity, not human physical occlusion')) as run:
        table=[];all_events=[]
        for split in ['dev','test','confirm','transfer','protected_confirm']:
            ptr=read_json(root/('protected_'+split+'_pointer.json'));folder=root/ptr['path']
            for seq in [s for s in manifest['sequences'] if s['split']==split]:
                records,_=load_cache(root,seq);mapping=match_gt(records,load_gt(root,seq))
                for mode in ['extension','extension_trajectory']:
                    tracks=read_json(folder/mode/seq['name']/'tracks.json');events=read_json(folder/mode/seq['name']/'decisions.json');byframe={};last={};counts=Counter()
                    for e in events:
                        if e['selected_appearance']:byframe.setdefault(e['frame'],[]).append(e)
                    for out in tracks:
                        f=out['frame']
                        for e in byframe.get(f,[]):
                            for tid,did in e['alternative']:
                                old=last.get(tid);new=mapping[f].get(did);outcome='unknown' if old is None or new is None else 'same_GT_identity' if old==new else 'different_GT_identity'
                                counts[outcome]+=1;all_events.append(dict(sequence=seq['name'],split=split,arm=mode,frame=f,track_id=tid,det_id=did,prior_gt=old,current_gt=new,outcome=outcome,features=e['features']))
                        for t in out['tracks']:
                            gt=mapping[f].get(t['det_id'])
                            if gt is not None:last[t['identity']]=gt
                    table.append(dict(sequence=seq['name'],split=split,arm=mode,counts=dict(counts)))
        write_json(run.path/'SUMMARY.json',dict(by_sequence=table,events=len(all_events)));write_json(run.path/'events.json',all_events)
        write_json(root/'connection_audit_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
        for x in table:
            if x['sequence'] in {'horse_3','zebra_1','zebra_4'}:print('CONNECTION_AUDIT',x,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
