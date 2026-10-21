from pathlib import Path
from collections import Counter
import json
import re
import shutil
import zipfile
import numpy as np
import cv2
from .common import Run,read_json,write_json,sha256,now
from .reversible_adjudicate import idf

def main(root):
    ep=read_json(root/'decision_evaluation_pointer.json');rp=Path(ep['path']);s=read_json(rp/'SUMMARY.json');mp=read_json(root/'decision_model_pointer.json');ap=read_json(root/'decision_action_audit_pointer.json');action=read_json(Path(ap['path'])/'SUMMARY.json');dest=root/'showcase';manifest=read_json(root/'input_manifest.json')
    with Run(root,'decision_final_verification',dict(evaluation=ep,training=mp,action_audit=ap,operation='configuration, arithmetic, immutable outputs, media, final report, transfer manifest',new_training_steps=0)) as run:
        modelchecks=[];train=Path(mp['path']);initial=np.load(train/'full/initial_parameters.npy');batches=read_json(train/'full/batch_indices.json')
        for a,v in mp['models'].items():
            assert sha256(v['path'])==v['sha256'];assert np.array_equal(initial,np.load(train/a/'initial_parameters.npy'));assert batches==read_json(train/a/'batch_indices.json');modelchecks.append(dict(arm=a,sha256=v['sha256'],same_initialization=True,same_batches=True))
        ptr=read_json(root/'identity_pointer.json');assert sha256(ptr['checkpoint'])==ptr['sha256']
        for k,p in manifest['models'].items():assert sha256(p)==manifest['model_sha256'][k]
        vendor=read_json(Path(r'D:\thesis-video-v3-20261019')/'RUNTIME_SOURCE_HASHES.json')
        for h in vendor:assert sha256(root/h['path'])==h['sha256'],h['path']
        write_json(root/'RUNTIME_SOURCE_HASHES.json',vendor)
        inputs=[]
        for seq in manifest['sequences']:
            for key in ['video','gt']:assert sha256(seq[key])==seq[key+'_sha256']
            for name in ['detections.json','features.npz']:
                p=Path(seq['cache_path'])/name;inputs.append(dict(sequence=seq['name'],path=str(p),sha256=sha256(p),bytes=p.stat().st_size))
        write_json(root/'FINAL_FEATURE_HASHES.json',inputs)
        equality=[]
        for seq in manifest['sequences']:
            if seq['decision_split']=='fit':continue
            c=read_json(rp/'confirm'/seq['name']/'tracks.json')
            for a in ['full','self','now','full_drop_comp','full_drop_time']:
                v=read_json(rp/a/seq['name']/'tracks.json');assert v==c,(seq['name'],a);equality.append(dict(sequence=seq['name'],arm=a,exact_C_output=True))
        arithmetic=[]
        for split,arms in s['combined'].items():
            for a,v in arms.items():
                rows=[r for r in s['by_sequence'] if r['split']==split and r['arm']==a];actual=idf(rows);assert abs(actual-v['IDF1'])<1e-8;arithmetic.append(dict(split=split,arm=a,IDF1=actual))
        videos=[]
        for v in read_json(dest/'video_catalog.json'):
            p=dest/v['path'];assert sha256(p)==v['sha256'];cap=cv2.VideoCapture(str(p));n=0
            while True:
                ok,img=cap.read()
                if not ok:break
                assert img.shape[:2]==(900,1600);n+=1
            cap.release();assert n==v['frames'];videos.append(dict(path=v['path'],decoded_frames=n,width=1600,height=900))
        html=(dest/'index.html').read_text(encoding='utf-8');links=re.findall(r'(?:src|href|poster)="([^"#]+)"',html)
        for link in links:
            if not link.startswith(('http','data:')):assert (dest/link).exists(),link
        events={y:{(r['sequence'],r['base_id']) for r in action['rows'] if r['y']==y} for y in [-1,0,1]}
        verified=dict(action_counts=action['counts'],unique_events={str(k):len(v) for k,v in events.items()},wrong_event_ids=sorted(events[1]),status='insufficient_distinct_wrong_events_for_population_alignment_gate',point_evidence='2 events / 4 trigger times; no single withdrawal improves IDF1; 2 harmful and 2 neutral actions',source=ap,correction='initial audit DECISION counted actions rather than distinct events; original retained',next_focus='build trajectory-level decision supervision; first verify action utility on independently identifiable events',new_training_authorized_by_this_audit=False)
        write_json(root/'ACTION_AUDIT_VERIFIED.json',verified);write_json(dest/'evidence/ACTION_AUDIT_VERIFIED.json',verified);write_json(dest/'evidence/ACTION_AUDIT.json',action)
        decision_text='**优先聚焦身份决策的监督目标与整段轨迹效用是否一致**，关闭CETD-v1当前配置，不继续调阈值或增训。依据：竞争信息有判别价值，但新视频AUROC仅0.630，校准时所有实际撤销方案都损伤，最终选择不撤销；2个错接事件的4个固定触发时刻，单独撤销也没有一项提高IDF1（2项下降、2项不变）。这证明局部错接标签不充分保证动作有益，尚不能估计总体错位比例。唯一下一步是在独立可判别事件上建立“保留/撤销/改接”的轨迹级动作结果标签，先验证监督是否对齐，再立新模型协议；该新训练尚未启动。'
        for p in [root/'REPORT.md',dest/'REPORT.md']:
            report=p.read_text(encoding='utf-8');report=report.replace('见下方人工核验后的研究裁决；本报告不将S0阳性或C的收益改记为新网络贡献。',decision_text)
            report += '\n\n## 动作监督核验与最终复核\n\n'+decision_text+'\n\n事后核验30个固定校准触发动作：13个局部正确标签（1正、2零、10负），4个局部错误标签（0正、2零、2负），13个未知（0正、9零、4负）。同一轨迹不同触发时刻相关；4个错接动作只来自2个事件，少于冻结的3事件条件，不能声称总体监督错位已经定量成立。原动作审计DECISION计数单位错误保留，以ACTION_AUDIT_VERIFIED.json为最终单位核验。\n\n全部15段评价视频中，三组学习模型及两项干预均与C逐帧输出完全相同，新增撤销次数为0。模型贡献为零，C相对ByteTrack的新确认+2.679不能记作CETD创新。模型/冻结权重/运行依赖哈希、相同初始化/批次、IDF1汇总复算和视频逐帧解码通过。\n'
            p.write_text(report,encoding='utf-8')
        card=root/'MODEL_AND_NOVELTY.md';text=card.read_text(encoding='utf-8');text=text.replace('模型效用和独立机制尚待正式比较。','正式比较已完成：新4视频完整模型与C完全相同、低于BoT-SORT，新增模型效用与独立机制未成立。');card.write_text(text,encoding='utf-8');shutil.copy2(card,dest/card.name)
        for name in ['ACTION_AUDIT_PROTOCOL.md','ENGINEERING_CORRECTIONS.md']:shutil.copy2(root/name,dest/name)
        html=html.replace('<section><h2>结论边界</h2>',f'<section><h2>下一研究重点</h2><p>{decision_text.replace("**","")}</p><p><a href="evidence/ACTION_AUDIT_VERIFIED.json">动作单位核验</a></p></section><section><h2>结论边界</h2>');(dest/'index.html').write_text(html,encoding='utf-8')
        verification=dict(models=modelchecks,frozen_weights_unchanged=True,runtime_hashes=len(vendor),sequence_input_hashes=len(manifest['sequences']),arithmetic=arithmetic,exact_C=equality,videos=videos,html_local_assets_checked=len(links),visual_review='architecture, result chart, video poster inspected via view_image; no browser rendering claimed')
        write_json(run.path/'VERIFICATION.json',verification);write_json(dest/'evidence/VERIFICATION.json',verification);write_json(run.path/'ACTION_AUDIT_VERIFIED.json',verified);write_json(run.path/'SUMMARY.json',dict(status='verified',extra_model_actions=0,model_parameters=12993,training_steps_per_arm=800,evaluation_sequences=15,arms=12))
    study=read_json(root/'study.json');study.update(plan_status='CETD_v1_closed',execution_status='completed',scientific_status='information_supported_model_utility_failed',artifact_status='verified_offline_exhibit',transfer_status='pending',runs=sorted(p.name for p in (root/'runs').iterdir() if (p/'run.json').exists()),formal_training_steps=2400,smoke_training_steps=60,updated_at=now());write_json(root/'study.json',study)
    for p in (root/'runs').iterdir():
        if not (p/'run.json').exists():continue
        r=read_json(p/'run.json');assert r['execution_status']=='completed',p.name
        assert sha256(p/'SOURCE_SNAPSHOT.json')==r['source_snapshot_sha256'];assert sha256(p/'EVAL_CONTRACT.json')==r['contract_sha256']
        for h in read_json(p/'SOURCE_SNAPSHOT.json'):assert sha256(p/'source'/h['path'])==h['sha256']
    include=[]
    for folder in ['code','runs','showcase']:
        for p in (root/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or 'cache' in p.relative_to(root).parts:continue
            include.append(p)
    include += [p for p in root.iterdir() if p.is_file() and p.suffix in ['.json','.md','.log'] and not p.name.startswith(('TRANSFER','DELIVERY_EXPORT'))]
    include=sorted(set(include));receipt=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in include];write_json(root/'TRANSFER_MANIFEST.json',receipt)
    out=root/'delivery_v4.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in include:z.write(p,p.relative_to(root).as_posix())
        z.write(root/'TRANSFER_MANIFEST.json','TRANSFER_MANIFEST.json')
    export=dict(path=str(out),sha256=sha256(out),bytes=out.stat().st_size,verified_files=len(receipt));write_json(root/'DELIVERY_EXPORT.json',export);print('DELIVERY',export,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
