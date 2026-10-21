from pathlib import Path
from types import SimpleNamespace
from .common import Run,read_json,write_json,setup_paths,load_gt
from .learn import load_cache,load_identity,embed_array
from .tracking import frame_input
from .evaluate import official_metrics,metric_data,compact


def run_sequence(mode,records,z,cfg):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.deep_oc_sort import DeepOCSORT
    tracker={'botsort':BOTSORT,'deepocsort':DeepOCSORT}[mode](SimpleNamespace(**cfg['tracker']));out=[]
    for rec in records:
        boxes,feat=frame_input(rec,z);rows=tracker.update(boxes,img=None,feats=feat)
        ts=[dict(base_id=int(t[4]),det_id=int(t[-1]),box=rec['boxes'][int(t[-1])],score=float(rec['scores'][int(t[-1])]),write=True,identity=int(t[4]),status='tracked') for t in rows]
        out.append(dict(frame=rec['frame'],tracks=ts))
    return out


def main(root,smoke=False):
    setup_paths(root);cfg=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json');model,ck=load_identity(root);metrics=official_metrics(root)
    modes=['botsort','deepocsort'];splits=['dev'] if smoke else ['dev','test','confirm','transfer','protected_confirm']
    with Run(root,'existing_baseline_smoke' if smoke else 'existing_baseline_comparison',dict(config=cfg['tracker'],modes=modes,splits=splits,new_training=0,feature=read_json(root/'identity_pointer.json'),native_encoder='auto',GMC='none')) as run:
        table=[];combined={};parity=[]
        for split in splits:
            raw={m:{} for m in modes}
            for seq in [s for s in manifest['sequences'] if s['split']==split]:
                records,rawx=load_cache(root,seq);records=records[:16] if smoke else records;z=embed_array(model,ck,rawx)
                for mode in modes:
                    out=run_sequence(mode,records,z,cfg)
                    if mode=='deepocsort' and split in {'dev','test'}:
                        ref=read_json(Path(read_json(root/'v1_reference.json')[split])/mode/seq['name']/'tracks.json')
                        assert out==(ref[:16] if smoke else ref),seq['name'];parity.append(seq['name'])
                    for f in out:assert len({t['identity'] for t in f['tracks']})==len(f['tracks'])
                    if not smoke:
                        dest=run.path/mode/seq['name'];dest.mkdir(parents=True);write_json(dest/'tracks.json',out)
                        gt=load_gt(root,seq);data=metric_data(gt,out);rs={k:m.eval_sequence(data) for k,m in metrics.items()};raw[mode][seq['name']]=rs
                        stat=dict(split=split,sequence=seq['name'],arm=mode,frames=seq['frames'],**compact(rs));table.append(stat);write_json(dest/'metric_details.json',rs);print('EXISTING',stat,flush=True)
            if not smoke:combined[split]={m:compact({k:metric.combine_sequences({s:rs[k] for s,rs in raw[m].items()}) for k,metric in metrics.items()}) for m in modes}
        write_json(run.path/'SUMMARY.json',dict(by_sequence=table,combined=combined,exact_legacy_parity=parity,smoke=smoke))
        if not smoke:write_json(root/'existing_baseline_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.root,a.smoke)
