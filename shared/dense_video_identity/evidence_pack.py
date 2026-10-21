"""Assemble measured evidence and verify aggregate denominators without model fitting."""
from pathlib import Path
import csv
import math
from .common import Run,read_json,write_json,sha256

LABELS={'motion':'ByteTrack','all_appearance':'全部外观','margin_rule':'差距规则','net_benefit':'单帧收益 CGS','correctness':'同容量正确性对照','trajectory':'轨迹收益 CGS-T','extension':'保护式补接 PEA','extension_trajectory':'PEA + 学习门','botsort':'BoT-SORT','deepocsort':'Deep OC-SORT'}
SPLITS={'dev':'开发集','test':'历史五视频复核','confirm':'预留马视频','transfer':'斑马1/2确认','protected_confirm':'斑马3/4确认'}


def build(root):
    manifest=read_json(root/'input_manifest.json');rows=[];combined={};sources=[]
    for split in SPLITS:
        combined[split]={}
        for family in ['conflict','trajectory','protected']:
            p=root/(family+'_'+split+'_pointer.json')
            if not p.exists():continue
            ptr=read_json(p);folder=root/ptr['path'];s=read_json(folder/'SUMMARY.json')
            assert read_json(folder/'run.json')['execution_status']=='completed'
            sources.append(dict(run_id=ptr['run_id'],summary_sha256=sha256(folder/'SUMMARY.json')))
            for r in s['by_sequence']:rows.append(dict(r,split=split,run_id=ptr['run_id'],track_path=str(folder/r['arm']/r['sequence']/'tracks.json')))
            combined[split].update(s['combined'])
    ptr=read_json(root/'existing_baseline_pointer.json');folder=root/ptr['path'];s=read_json(folder/'SUMMARY.json')
    for r in s['by_sequence']:rows.append(dict(r,run_id=ptr['run_id'],track_path=str(folder/r['arm']/r['sequence']/'tracks.json')))
    for split,metrics in s['combined'].items():combined[split].update(metrics)
    sources.append(dict(run_id=ptr['run_id'],summary_sha256=sha256(folder/'SUMMARY.json')))
    with Run(root,'evidence_assembly',dict(operation='aggregate recomputation, state/identity invariants and reporting',new_training=0,source_runs=sources)) as run:
        checks=[]
        for split,arms in combined.items():
            for arm,score in arms.items():
                selected=[r for r in rows if r['split']==split and r['arm']==arm]
                idtp=sum(r['IDTP'] for r in selected);idfp=sum(r['IDFP'] for r in selected);idfn=sum(r['IDFN'] for r in selected)
                recalculated=200*idtp/(2*idtp+idfp+idfn)
                assert abs(recalculated-score['IDF1'])<1e-8,(split,arm)
                assert score['FN']==sum(r['FN'] for r in selected)
                checks.append(dict(split=split,arm=arm,recomputed_IDF1=recalculated,passed=True))
        unique_frames=0
        for r in rows:
            tracks=read_json(r['track_path']);seq=next(s for s in manifest['sequences'] if s['name']==r['sequence'])
            assert len(tracks)==seq['frames']
            for f in tracks:
                assert len({t['identity'] for t in f['tracks']})==len(f['tracks'])
                assert len({t['det_id'] for t in f['tracks']})==len(f['tracks'])
            unique_frames+=len(tracks)
        def gate(split,arm):
            c=combined[split];n=c['motion'];a=c['all_appearance'];m=c[arm]
            result=dict(delta_IDF1_native=m['IDF1']-n['IDF1'],delta_IDF1_all_appearance=m['IDF1']-a['IDF1'],delta_HOTA_native=m['HOTA']-n['HOTA'])
            result['passed']=result['delta_IDF1_native']>=1 and result['delta_IDF1_all_appearance']>0 and result['delta_HOTA_native']>=-.5
            return result
        gates={'CGS_T_development':gate('test','trajectory'),'CGS_T_independent_zebra12':gate('transfer','trajectory'),'PEA_development':gate('test','extension'),'PEA_independent_zebra34':gate('protected_confirm','extension')}
        damage=[]
        for r in rows:
            if r['arm']!='extension':continue
            n=next(x for x in rows if x['sequence']==r['sequence'] and x['arm']=='motion')
            delta=r['IDF1']-n['IDF1']
            if delta<-3:damage.append(dict(sequence=r['sequence'],delta_IDF1=delta))
        topology=read_json(root/read_json(root/'trajectory_opportunity_pointer.json')['path']/'examples.json')
        counts={}
        for kind in ['new_connection','replacement']:
            es=[e for e in topology if (not e['base'])==(kind=='new_connection')]
            counts[kind]=dict(n=len(es),positive=sum(e['gain']>0 for e in es),negative=sum(e['gain']<0 for e in es),neutral=sum(e['gain']==0 for e in es),net=sum(e['gain'] for e in es))
        data=dict(labels=LABELS,split_labels=SPLITS,rows=rows,combined=combined,gates=gates,PEA_damage_over3=damage,training_topology=counts,
                  source_runs=sources,scopes={k:dict(sequences=[s['name'] for s in manifest['sequences'] if s['split']==k],frames=sum(s['frames'] for s in manifest['sequences'] if s['split']==k)) for k in SPLITS},
                  annotations=read_json(root/read_json(root/'field_evaluation_pointer.json')['path']/'SUMMARY.json'),
                  decision='Keep reproducible local gains, close current CGS/CGS-T/PEA fixed configurations to further tuning. Prioritize verification and rollback of false reattachment without changing the motion tracker state; no new training launched.')
        write_json(run.path/'SUMMARY.json',dict(gates=gates,rows=len(rows),aggregate_checks=len(checks),checked_track_frames=unique_frames,training_topology=counts))
        write_json(run.path/'VERIFICATION.json',dict(aggregate_checks=checks,unique_identity_and_detection_all_frames=True,checked_track_frames=unique_frames))
        write_json(run.path/'DECISION.json',dict(gates=gates,per_video_damage_over3=damage,primary_claim='Measured local and zebra12 gains; general superiority and new supervision mechanism not established',next_action=data['decision'],new_training_started=False))
        write_json(root/'showcase/results.json',data)
        write_json(root/'evidence_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
        out=root/'showcase/scores.csv'
        fields=['split','sequence','arm','IDF1','HOTA','IDSW','FP','FN','IDTP','IDFP','IDFN','run_id']
        with out.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        print('EVIDENCE_GATES',gates,flush=True)
        for split in ['test','transfer','protected_confirm']:print('COMBINED',split,{k:round(v['IDF1'],4) for k,v in combined[split].items()},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);build(p.parse_args().root)
