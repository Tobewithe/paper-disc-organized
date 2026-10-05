import argparse,csv,json
from pathlib import Path
import numpy as np
from recording import atomic_json,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.study/'protocol.json').read_text());manifest=json.loads((a.study/'data/manifest.json').read_text())
    metrics={};rows={};trains={}
    certified=json.loads((a.study/'certified_receipts.json').read_text()) if (a.study/'certified_receipts.json').exists() else {}
    for spec in cfg['runs']:
        folder=a.study/'runs'/spec['run_id']
        if spec['kind']=='training':trains[spec['arm']]=json.loads(Path(certified.get(spec['arm'],folder/'TRAINING_COMPLETE.json')).read_text())
        if spec['kind']=='evaluation':
            metrics[spec['arm']]=json.loads((folder/'metrics.json').read_text())
            with (folder/'per_gt.csv').open(encoding='utf-8-sig',newline='') as f:
                rows[spec['arm']]={(r['task'],int(r['annotation_id'])):r for r in csv.DictReader(f)}
    with (a.study/'data/panel.csv').open(encoding='utf-8-sig',newline='') as f:panel=list(csv.DictReader(f))
    contrasts=[]
    for new,base in [('teacher_edited','teacher_original'),('teacher_edited','hard_gt'),('teacher_edited','baseline'),('teacher_original','baseline'),('hard_gt','baseline')]:
        assert set(rows[new])==set(rows[base])
        for task in ['bbox','segm']:
            for scope in ['raw_geometry_small','matched_small_control','random_val','random_small']:
                if scope.startswith('random'):
                    ids=[gid for (t,gid),r in rows[base].items() if t==task and int(r['image_id']) in manifest['random_image_ids'] and (scope!='random_small' or float(r['area'])<1024)]
                else:ids=[int(r['annotation_id']) for r in panel if r['cohort']==scope]
                imids=np.array([int(rows[base][task,gid]['image_id']) for gid in ids]);unique=np.unique(imids)
                for key in ['r50','r75']:
                    old=np.array([float(rows[base][task,gid][key]) for gid in ids]);newv=np.array([float(rows[new][task,gid][key]) for gid in ids]);diff=newv-old
                    sums=np.array([diff[imids==i].sum() for i in unique]);counts=np.array([(imids==i).sum() for i in unique])
                    rng=np.random.default_rng(916);samples=rng.integers(len(unique),size=(1000,len(unique)))
                    boot=sums[samples].sum(1)/counts[samples].sum(1);lo,hi=np.quantile(boot,[.025,.975])
                    contrasts.append(dict(method=new,reference=base,task=task,scope=scope,metric=key,n=len(ids),
                        method_mean=float(newv.mean()),reference_mean=float(old.mean()),delta_points=100*float(diff.mean()),
                        ci_low_points=100*float(lo),ci_high_points=100*float(hi),recovered=int((diff>0).sum()),lost=int((diff<0).sum())))
    atomic_json(a.out/'CONTRASTS.json',contrasts);atomic_json(a.out/'ALL_METRICS.json',metrics);atomic_json(a.out/'TRAINING_SUMMARY.json',trains)
    lines=['# 1,000 张、1 轮、seed 0：监督桥接快速筛查','',
        '评价：256 张独立随机 COCO val 图像；既定匹配池随机抽取 64 对失败/对照实例。主评价固定 last、one-to-one；原始 COCOeval 一对一匹配，全图所有 GT 保留。',
        '区间为 1,000 次图像级配对 bootstrap，仅反映该验证样本的抽样不确定性，不代表训练种子稳定性。AP 是小子集描述值。','',
        '| 方法 | 随机子集 Box AP | Mask AP | 小实例 Box R50 | 小实例 Mask R75 | 辅助监督次数 |','|---|---:|---:|---:|---:|---:|']
    for arm in cfg['arms']:
        box=next(x for x in metrics[arm] if x['scope']=='random_val' and x['task']=='bbox');mask=next(x for x in metrics[arm] if x['scope']=='random_val' and x['task']=='segm')
        lines.append(f"| {arm} | {box['ap']*100:.3f} | {mask['ap']*100:.3f} | {box['small_r50']*100:.3f} | {mask['small_r75']*100:.3f} | {trains[arm]['method_stats']['selected_positions']} |")
    lines+=['','## 编辑教师相对原图教师（百分点）','','| 任务/对象 | 指标 | 差值 | 95% CI | 恢复 / 失去 |','|---|---|---:|---|---:|']
    for r in contrasts:
        if r['method']=='teacher_edited' and r['reference']=='teacher_original':
            lines.append(f"| {r['task']} / {r['scope']} | {r['metric']} | {r['delta_points']:+.3f} | [{r['ci_low_points']:+.3f}, {r['ci_high_points']:+.3f}] | {r['recovered']} / {r['lost']} |")
    lines+=['','这是难位置监督筛查，未要求教师恢复门限；不能沿用旧方案的“教师已恢复实例”表述。C/D 共同候选规则、损失、系数一致，但各自训练后的候选与激活次数可以不同。',
        '1 轮且 1,000 张只能识别早期信号；不支持把无差异解读为该路线不可行，也不支持把单个变化实例解读为稳定方法收益。']
    (a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(a.out/'COMPLETE.json',dict(status='complete',at=now(),arms=cfg['arms'],contrasts=len(contrasts)))
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
