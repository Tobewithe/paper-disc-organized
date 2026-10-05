"""Normal original-image evaluation; frozen candidate set and fixed epoch3."""
import argparse
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from runtime_utils import dump,load,gpu_image,setup
from train_evidence import read_training_image,feature_channels,feed
from training_engine import make_model,resolve_config
import evaluate as metrics

ARMS=('A','N','F','R','RH','R1')
PAIRS=(('R','F'),('R','A'),('R','N'),('F','A'),('N','A'),('R','RH'),('R','R1'))
metrics.ARMS=ARMS
metrics.PAIRS=PAIRS
_coco={};_baseline={}


def identity(row):
    return (row['split'],int(row['image_id']),int(row['annotation_id']),int(row['raw_id']))


def coco_for(cfg):
    key=cfg['annotations_train']
    if key not in _coco: _coco[key]=COCO(key)
    return _coco[key]


def baseline_for(cfg):
    key=cfg['baseline_rows']
    if key not in _baseline:
        rows={}
        with open(key,encoding='utf-8-sig') as stream:
            for line in stream:
                if line.strip():
                    row=json.loads(line);k=identity(row)
                    if k in rows: raise ValueError('Duplicate historical identity')
                    rows[k]=(float(row['iou_A']),int(row['mask75_A']))
        _baseline[key]=rows
    return _baseline[key]


@torch.no_grad()
def bce_values(x,cs):
    proto=F.interpolate(x['proto'][None].float(),(640,640),mode='bilinear',align_corners=False)[0]
    parts=[]
    for lo in range(0,len(cs),8):
        c=cs[lo:lo+8]; box=x['target_boxes'][lo:lo+8]
        gt=(x['masks'][None]==(x['owners'][lo:lo+8]+1)[:,None,None]).float()
        z=torch.einsum('in,nhw->ihw',c.float(),proto)
        area=((box[:,2:]-box[:,:2])/640).prod(1)
        assert (area>0).all()
        loss=ops.crop_mask(F.binary_cross_entropy_with_logits(z,gt,reduction='none'),box).mean((1,2))/area*float(x['segmentation_gain'])
        if not torch.isfinite(loss).all(): raise FloatingPointError('Nonfinite BCE')
        parts.append(loss.cpu())
    return torch.cat(parts) if parts else torch.empty(0)


def decode_smoke(cfg,x,c,mode):
    gx=gpu_image(x)
    decoded=metrics.evaluate_image(gx,{'A':gx['c0'],mode:c.detach()},coco_for(cfg))
    baseline=baseline_for(cfg); maximum=0.; flips=0; mask75_changes=0; iouchange=0.
    for row in decoded:
        old=baseline[identity(row)]
        error=abs(row['iou_A']-old[0]);maximum=max(maximum,error)
        if error>1e-6 or row['mask75_A']!=old[1]: raise AssertionError('Original mask replay differs')
        flips+=row['original_pixel_changes_'+mode+'_vs_A']
        mask75_changes+=row['mask75_'+mode]!=row['mask75_A']
        iouchange=max(iouchange,abs(row['iou_'+mode]-row['iou_A']))
    if iouchange>1e-5 or mask75_changes: raise AssertionError('Initialization decode altered task metrics')
    return dict(candidates=len(decoded),historical_baseline_max_error=maximum,initial_original_pixel_flips=flips,initial_iou_max_change=iouchange,initial_mask75_changes=mask75_changes)


def summarize(rows,out,cfg,expected,audit,timing):
    tables={}
    for split in ('fit','dev'):
        ss=[r for r in rows if r['split']==split]
        groups={'all':ss,'box_good_mask_bad':[r for r in ss if r['box_good_mask_bad']],
                'original_success':[r for r in ss if r['mask75_A']], 'original_failure':[r for r in ss if not r['mask75_A']]}
        if split=='dev':
            groups.update({f'P{level+3}':[r for r in ss if r['pyramid_level']==level] for level in range(3)})
        for name,group in groups.items():
            tables[split+':'+name]=metrics.make_table(group,20261003,cfg['bootstrap'],'exploratory image-paired screening')
    def delta(group,pair): return tables[group]['comparisons'][pair]['iou']['image_macro']['delta']
    rf=delta('dev:all','R_minus_F');ra=delta('dev:all','R_minus_A');rn=delta('dev:all','R_minus_N')
    target=delta('dev:box_good_mask_bad','R_minus_A')
    practical=(ra is not None and ra>=cfg['screen_effect_all']) or (target is not None and target>=cfg['screen_effect_target'])
    point_positive=all(v is not None and v>0 for v in (rf,ra,rn)) and practical
    decision='positive_screen_needs_separate_confirmation' if point_positive else 'stop_current_implementation_no_predeclared_positive_screen'
    result=dict(tables=tables,expected=expected,audit=audit,seconds=timing,decision=decision,
                primary='dev:all R_minus_F image_macro original_mask_iou',single_seed_exploratory=True,automatic_followup=False)
    metrics.write_json(out/'SUMMARY.json',result)
    with (out/'PER_IMAGE.jsonl').open('w',encoding='utf-8') as stream:
        for group in metrics.image_groups(rows):
            rec=dict(split=group[0]['split'],image_id=group[0]['image_id'],n=len(group))
            for arm in ARMS:
                for metric in metrics.METRICS: rec[f'{metric}_{arm}']=metrics.avg([metrics.metric_value(r,metric,arm) for r in group])
            stream.write(json.dumps(metrics.clean(rec),ensure_ascii=False,allow_nan=False)+'\n')
    def fmt(v): return 'NA' if v is None else f'{100*v:+.4f}'
    report=['# 原型响应反馈读取：短程结果','',
            f'判定：{decision}。本轮只有seed0与历史开发图，属于筛选，不是独立确认或完整COCO AP。',
            '','固定epoch3；R使用更新后的响应聚合特征，F始终使用初始响应。N为原生系数分支普通微调；A为原模型。RH是同一R参数改用旧响应；R1是R的第一步中间结果。','',
            '| 分组 | 比较 | 图片macro IoU差(pp) | 95%图片配对区间(pp) | 候选IoU差(pp) | 修复/损伤 |','|---|---|---:|---|---:|---|']
    for group in ['fit:all','dev:all','dev:box_good_mask_bad','dev:original_success','dev:original_failure']:
        for a,b in PAIRS:
            p=tables[group]['comparisons'][f'{a}_minus_{b}']; m=p['iou']['image_macro'];tr=p['mask75_transition']
            report.append(f"|{group}|{a}-{b}|{fmt(m['delta'])}|[{fmt(m['ci95'][0])},{fmt(m['ci95'][1])}]|{fmt(p['iou']['candidate']['delta'])}|{tr['repair']}/{tr['damage']}|")
    report += ['','Coverage、AUC、FPR、BCE以及各尺度逐图/候选均值和配对区间见SUMMARY.json；AUC未定义的候选单独计数，不删除其IoU。',
               '','解释边界：R-F检验本实现的反馈训练效果；R-RH是同参数急性干预，可能含输入分布变化，不能单独代替F/R训练对照。仅fit提升不构成泛化收益。所有计数包含无效算子/空掩码候选；仅无正样本图无实例指标。',
               '','结束：不自动增加步数、门控、loss、数据或训练时长。正信号须另行独立确认；无信号只降低本实现优先级，不宣布所有反馈无效。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoints',required=True)
    args=p.parse_args();root=Path(args.config).parent;out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg=resolve_config(json.loads(Path(args.config).read_text(encoding='utf-8-sig')));setup(cfg['seed'])
    if not torch.cuda.is_available(): raise RuntimeError('GPU host required')
    if (out/'PER_CANDIDATE.jsonl').exists(): raise RuntimeError('Existing evaluation rows; fresh Run required')
    index=json.loads((root/'INDEX.json').read_text(encoding='utf-8-sig'))
    selected={'fit':index['fit'][:cfg['fit_eval_images']],'dev':index['dev']}
    first=next(r for rows in selected.values() for r in rows if r['n'])
    channels=feature_channels(read_training_image(cfg,first['image_id']))
    paths=json.loads(Path(args.checkpoints).read_text(encoding='utf-8-sig'));models={}
    for mode in cfg['arms']:
        ck=load(paths[mode]);assert ck['epoch']==cfg['epochs'] and ck['config']==cfg and ck['mode']==mode
        model=make_model(channels,mode,cfg).eval().requires_grad_(False);model.load_state_dict(ck['state_dict'],strict=True);models[mode]=model
    baseline=baseline_for(cfg);coco=coco_for(cfg);rows=[];visited=set();expected={};started=time.monotonic();base_error=0.
    timing={a:0. for a in ('N','F','R','RH')}
    for split,items in selected.items():
        expected[split]=dict(planned_images=len(items),effective_images=sum(r['n']>0 for r in items),candidates=sum(r['n'] for r in items),no_positive=[r['image_id'] for r in items if not r['n']])
        for position,item in enumerate(items):
            if not item['n']: continue
            x=read_training_image(cfg,int(item['image_id']));assert x['split']==split and len(x['rows'])==item['n']
            fs,sel=feed([x]);cs={}
            with torch.no_grad():
                for mode,model in models.items():
                    torch.cuda.synchronize();t=time.monotonic()
                    if mode=='R':
                        final,stage1,trace,_=model.forward_trace(fs,sel);cs['R']=final[0];cs['R1']=stage1[0]
                    else: cs[mode]=model(fs,sel)[0]
                    torch.cuda.synchronize();timing[mode]+=time.monotonic()-t
                torch.cuda.synchronize();t=time.monotonic();cs['RH']=models['R'].forward_trace(fs,sel,hold=True)[0][0]
                torch.cuda.synchronize();timing['RH']+=time.monotonic()-t
                gx=gpu_image(x);cs['A']=gx['c0'];loss={a:bce_values(gx,c) for a,c in cs.items()}
                decoded=metrics.evaluate_image(gx,cs,coco)
            for j,row in enumerate(decoded):
                key=identity(row)
                if key in visited or key not in baseline: raise ValueError('Identity baseline join failed')
                visited.add(key);old=baseline[key];error=abs(row['iou_A']-old[0]);base_error=max(base_error,error)
                if error>1e-6 or row['mask75_A']!=old[1]: raise AssertionError('Baseline changed')
                for arm in cs: row[f'bce_{arm}']=float(loss[arm][j])
                row['operator_valid']=bool(x['_operator']['valid'][j])
                row['R_context_change_step2']=trace[0]['context_change_step2']
            metrics.append_rows(out/'PER_CANDIDATE.jsonl',decoded);rows.extend(decoded)
            if position%10==0 or position+1==len(items):
                state=dict(stage='evaluation',split=split,images=position+1,planned=len(items),candidates=len(rows),elapsed_s=time.monotonic()-started)
                dump(out/'PROGRESS.json',state);print(json.dumps(state),flush=True)
            del x,gx,fs,sel,cs,decoded,loss
    assert len(rows)==sum(r['candidates'] for r in expected.values())
    audit=dict(baseline_max_error=base_error,all_identities_unique=True,all_candidates_retained=True,checkpoints=paths,normal_decode='unchanged inherited evaluator with original COCO GT',no_val_tuning=True)
    dump(out/'EVALUATION_AUDIT.json',audit)
    result=summarize(rows,out,cfg,expected,audit,timing)
    dump(out/'COMPLETE.json',dict(completed=True,decision=result['decision'],elapsed_s=time.monotonic()-started,automatic_followup=False))


if __name__=='__main__': main()
