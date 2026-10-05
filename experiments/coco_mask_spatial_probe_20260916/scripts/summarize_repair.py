"""Paired image bootstrap of fixed COCO Mask75 matched identities, plus AP table."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    for k in ('source','annotations','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    results=json.loads((a.source/'RESULTS.json').read_text());matched={k:set(v) for k,v in json.loads((a.source/'MATCHED_GT75.json').read_text()).items()}
    raw=json.loads(a.annotations.read_text());ids=json.loads((a.source/'SPLIT.json').read_text())['val_ids'];image_index={v:i for i,v in enumerate(ids)}
    anns=[ann for ann in raw['annotations'] if ann['image_id'] in image_index and not ann.get('iscrowd',0) and not ann.get('ignore',0)]
    rng=np.random.default_rng(20260916)
    draws=rng.multinomial(len(ids),[1/len(ids)]*len(ids),size=2000).astype(np.float32)
    def stratum(ann):return 'small' if ann['area']<1024 else 'medium' if ann['area']<9216 else 'large'
    comparisons=[('safe_full','combo_full'),('combo_selected','combo_full'),('combo_selected','scalar_selected'),('safe_selected','scalar_selected'),('safe_selected','combo_selected')]
    pairs=[(name,'baseline') for name in results if name!='baseline']+comparisons
    contrasts={}
    for left,right in pairs:
        contrast={}
        for group in ('all','small','medium','large'):
            gg=anns if group=='all' else [ann for ann in anns if stratum(ann)==group]
            counts=np.zeros(len(ids),np.float32);delta=np.zeros(len(ids),np.float32)
            for ann in gg:
                i=image_index[ann['image_id']];counts[i]+=1;delta[i]+=int(ann['id'] in matched[left])-int(ann['id'] in matched[right])
            ratio=draws@delta/(draws@counts)*100
            contrast[group]=dict(gt=int(counts.sum()),net_gt=int(delta.sum()),delta_recall_points=float(delta.sum()/counts.sum()*100),
                ci95=np.quantile(ratio,[.025,.975]).tolist())
        contrasts[f'{left}_minus_{right}']=contrast
    out=dict(contrasts=contrasts,interpretation='Image-clustered bootstrap of fixed official COCO GT matches. Not AP confidence intervals; one seed, exploratory sameval.')
    (a.out/'CONTRASTS.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# 误伤修复对照的完整结果','','| 组 | Mask AP | ΔAP点 | AP75 | AP小 | AP中 | AP大 | 修复 | 误伤 | 净增 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,row in results.items():
        m=row['metrics'];lines.append(f"| {name} | {100*m['AP']:.3f} | {100*(m['AP']-results['baseline']['metrics']['AP']):+.3f} | {100*m['AP75']:.3f} | {100*m['APS']:.3f} | {100*m['APM']:.3f} | {100*m['APL']:.3f} | {row['repaired75']} | {row['damaged75']} | {row['repaired75']-row['damaged75']} |")
    lines+=['','## 预定的机制对照','','| 对照 | Mask75召回差，点 | 图片配对95%区间 |','|---|---:|---|']
    for left,right in comparisons:
        v=contrasts[f'{left}_minus_{right}']['all'];lo,hi=v['ci95'];lines.append(f"| {left} − {right} | {v['delta_recall_points']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines+=['','表中AP没有计算置信区间。召回区间来自2000次按图片配对重采样，只反映此验证集的抽样不确定性，不证明跨训练seed稳定。所有模型使用同一800图拟合/200图选择集，强度在train2017选择。','']
    (a.out/'TABLE.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
