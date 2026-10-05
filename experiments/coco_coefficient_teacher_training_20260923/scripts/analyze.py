import argparse, gzip, json, os
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root;out=Path(os.environ['RESEARCH_RUN_DIRECTORY']);ids=json.loads((root/'deployment_ids.json').read_text())['run_ids']
    cfg=json.loads((root/'PROTOCOL.json').read_text());source=json.loads((Path(cfg['data'])/'annotations/instances_val2017.json').read_text())
    anns={v['id']:v for v in source['annotations'] if not v.get('iscrowd',0) and not v.get('ignore',0)}
    image_ids=sorted(v['id'] for v in source['images']);image_index={v:i for i,v in enumerate(image_ids)}
    matched={};metrics={};training={}
    for mode in 'RABC':
        r=root/'runs'/ids['eval_'+mode]
        metrics[mode]=json.loads((r/'RESULTS.json').read_text());matched[mode]=set(json.loads((r/'MATCHED_GT75.json').read_text())['mask'])
        if mode!='R':
            tr=root/'runs'/ids['train_'+mode]
            training[mode]={'complete':json.loads((tr/'COMPLETE.json').read_text()),'run':json.loads((tr/'run.json').read_text())['started_at']}
    assert len({v['detection_identity_sha256'] for v in metrics.values()})==1,'Frozen detector prediction identity changed'
    order={m:(root/'runs'/ids['train_'+m]/'batch_identity.jsonl').read_text() for m in 'ABC'}
    assert order['A']==order['B']==order['C'],'Paired input/augmentation sequence differs'
    groups={'all':set(anns),'small':{i for i,v in anns.items() if v['area']<1024},
            'low_fill':{i for i,v in anns.items() if v['area']/max(v['bbox'][2]*v['bbox'][3],1e-9)<.25}}
    results={};rng=np.random.default_rng(923)
    for name,group in groups.items():
        denominator=np.zeros(len(image_ids))
        for i in group:denominator[image_index[anns[i]['image_id']]]+=1
        results[name]={}
        for mode in 'RABC':
            good=matched[mode]&group
            results[name][mode]={'gt':len(group),'matched':len(good),'recall75':100*len(good)/max(len(group),1),
             'repairs_vs_R':len((good-matched['R'])&group),'damages_vs_R':len((matched['R']-good)&group)}
        contrasts={}
        for high,low in [('A','R'),('B','A'),('C','A'),('C','B')]:
            delta=np.zeros(len(image_ids))
            for i in (matched[high]-matched[low])&group:delta[image_index[anns[i]['image_id']]]+=1
            for i in (matched[low]-matched[high])&group:delta[image_index[anns[i]['image_id']]]-=1
            samples=[]
            for _ in range(2000):
                ix=rng.integers(0,len(image_ids),len(image_ids));samples.append(100*delta[ix].sum()/max(denominator[ix].sum(),1))
            contrasts[high+'-'+low]={'delta_recall75':100*delta.sum()/max(denominator.sum(),1),
                                   'image_bootstrap95':np.quantile(samples,[.025,.975]).tolist()}
        results[name]['contrasts']=contrasts
    # Continuous same-output candidate panel, selected solely from baseline box geometry.
    with gzip.open(root/'runs'/ids['eval_R']/'fixed_output_geometry.json.gz','rt') as f:reference=json.load(f)
    panel={}
    for mode in 'RABC':
        with gzip.open(root/'runs'/ids['eval_'+mode]/'fixed_output_geometry.json.gz','rt') as f:geom=json.load(f)
        values=[]
        for iid,r in reference.items():
            g=geom[iid];assert g['gt_ids']==r['gt_ids'] and g['raw_ids']==r['raw_ids'] and g['classes']==r['classes']
            if not r['gt_ids'] or not r['raw_ids']:continue
            bi=np.asarray(r['box_iou']);mi=np.asarray(g['mask_iou']);base=np.asarray(r['mask_iou'])
            for j,ann_id in enumerate(r['gt_ids']):
                k=int(bi[:,j].argmax())
                if bi[k,j]<.5:continue
                iou=float(mi[k,j]);pg=float(g['pred_areas'][k]);gg=float(g['gt_areas'][j]);inter=iou*(pg+gg)/(1+iou)
                values.append({'annotation_id':ann_id,'image_id':int(iid),'raw_id':r['raw_ids'][k],
                               'box_iou':float(bi[k,j]),'mask_iou':iou,'base_mask_iou':float(base[k,j]),
                               'coverage':inter/max(gg,1),'purity':inter/pg if pg else None})
        with gzip.open(out/f'fixed_output_panel_{mode}.jsonl.gz','wt') as f:
            for v in values:f.write(json.dumps(v)+'\n')
        panel[mode]={'n':len(values),'mean_mask_iou':float(np.mean([v['mask_iou'] for v in values])),
                     'mean_coverage':float(np.mean([v['coverage'] for v in values])),
                     'mean_purity_nonempty':float(np.mean([v['purity'] for v in values if v['purity'] is not None]))}
        del geom
    result={'metrics':metrics,'recall_groups':results,'fixed_output_panel':panel,
            'input_order_equal':True,'detector_outputs_equal':True,
            'limitations':['seed0 only','image intervals conditional on fixed weights, not AP intervals','fixed-output panel is not full raw capability','not end-to-end training']}
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2))
    lines=['# 原生系数头在线训练：首轮结果','', '固定Ultralytics 8.4.100，one-to-many + NMS；2000图训练，末轮EMA cv4，完整5000图官方COCOeval。', '',
           '|组|Mask AP|AP75|AP small|Box AP|修复@75|误伤@75|','|---|---:|---:|---:|---:|---:|---:|']
    for mode in 'RABC':
        m=metrics[mode];g=results['all'][mode]
        lines.append(f"|{mode}|{m['mask']['AP']:.4f}|{m['mask']['AP75']:.4f}|{m['mask']['APS']:.4f}|{m['box']['AP']:.4f}|{g['repairs_vs_R']}|{g['damages_vs_R']}|")
    lines+=['','A普通继续训练；B直接纠错监督；C训练期系数求解目标。检测输出与成对训练输入一致。', '',
            '方法判断重点为C对B及A，而非仅对R。单种子结果不能直接证明稳定性；本轮没有解冻原型/骨干。']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps({'status':'completed','summary':'RESULTS.md'}));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
