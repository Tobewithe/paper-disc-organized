"""Local deterministic report and arithmetic integrity checks for S015."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'diagnostics/no_candidate317_20260912'
REPORT=ROOT.parents[1]/'refine-logs/coco-structure'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    rec=json.loads((OUT/'COMPLETE.json').read_text())
    for name,digest in rec['hashes'].items():assert sha(OUT/name)==digest,name
    config=json.loads((OUT/'protocol.json').read_text());assert sha(ROOT/'no_candidate_readout_probe.py')==config['script_sha256']
    ac=json.loads((OUT/'ANALYSIS_COMPLETE.json').read_text());assert sha(OUT/'ANALYSIS.json')==ac['analysis_sha256']
    assert sha(ROOT/'summarize_no_candidate_readout.py')==ac['script_sha256']
    analysis=json.loads((OUT/'ANALYSIS.json').read_text());targets=read(OUT/'targets.csv');metrics=read(OUT/'metrics.csv')
    domain=json.loads((OUT/'ORIGINAL_DOMAIN_ANALYSIS.json').read_text())
    for name,digest in domain['hashes'].items():assert sha(OUT/name)==digest,name
    assert sha(ROOT/'verify_no_candidate_readout.py')==domain['script_sha256']
    final_count=0
    if (OUT/'FINAL_SHA256SUMS.txt').exists():
        for line in (OUT/'FINAL_SHA256SUMS.txt').read_text().splitlines():
            digest,name=line.split('  ',1);assert sha(OUT/name)==digest,name;final_count+=1
    assert len(targets)==317 and len({r['annotation_id'] for r in targets})==317
    assert sum(analysis['stages'].values())==317
    error=0.
    for r in metrics:
        a=int(r['gt_valid_pixels']);tp=int(r['tp']);fp=sum(int(r[k]) for k in ['same_neighbor_pixels','background_pixels','other_pixels'])
        error=max(error,abs(float(r['mask_iou'])-tp/(a+fp)))
        assert tp+int(r['fn'])==a
        assert float(r['coverage'])<=float(r['crop_coverage'])+1e-12
    assert error<1e-12
    ok={r['annotation_id'] for r in targets if r['fit_status']=='ok'}
    for arm in ['original','threshold_oracle','free_coefficient','free_coefficient_bias']:
        rr=[r for r in metrics if r['arm']==arm and r['annotation_id'] in ok]
        s=next(r for r in analysis['groups'] if r['group']=='all_eligible' and r['arm']==arm)
        assert len(rr)==s['targets']
        assert sum(float(r['mask_iou'])>=.75 for r in rr)==s['recovered75']
        assert abs(100*np.mean([float(r['mask_iou']) for r in rr])-s['mask_iou'])<1e-10
    verification=dict(status='PASS',files_verified=len(rec['hashes'])+1,targets=317,
        fitted_targets=len(ok),pixel_iou_max_arithmetic_error=error,
        original_domain_additional_files_verified=len(domain['hashes']),
        final_remote_manifest_files_verified=final_count,
        scope='Hash, denominator, exact pixel arithmetic and summary checks; not independent solver execution or full official AP.')
    (OUT/'LOCAL_VERIFICATION.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    names={'no_raw_box50':'无 IoU≥0.5 原始框','no_correct_argmax_box50':'框可用，类别未通过','no_postconf_correct_box50':'框/类别可用，分数未通过',
        'postconf_box50_but_unmatched_final':'有阈值后框，最终未匹配','final_bbox50_matched_mask_failure':'最终框匹配，掩码失败'}
    vals=list(analysis['stages'].values());labs=[names[k] for k in analysis['stages']]
    axes[0].barh(np.arange(len(vals)),vals,color='#667e92')
    axes[0].set_yticks(np.arange(len(vals)),labs);axes[0].invert_yaxis();axes[0].set_xlim(0,max(vals)*1.17)
    for i,v in enumerate(vals):axes[0].text(v+2,i,str(v),va='center')
    axes[0].set_title('317 个无合格候选目标：失败阶段')
    arms=['original','threshold_oracle','free_coefficient','free_coefficient_bias']
    rr=[next(r for r in domain['groups'] if r['group']=='all_eligible' and r['arm']==arm) for arm in arms]
    ys=[r['coco_iou75'] for r in rr];labels=['原系数','只调阈值','32维自由系数','系数＋偏置']
    axes[1].bar(np.arange(4),ys,color=['#667e92','#d79937','#008883','#7853a1'])
    axes[1].set_xticks(np.arange(4),labels,rotation=15);axes[1].set_ylim(0,max(ys)*1.2)
    for i,v in enumerate(ys):axes[1].text(i,v+max(ys)*.025,str(v),ha='center')
    axes[1].set_title(f'固定原型与预测框：{len(ok)} 个可拟合目标\n原始COCO标注域 IoU≥0.75 的数量')
    fig.suptitle('已有候选画不对以后：临时改变读出能恢复多少？',fontsize=15,y=1.02)
    fig.text(.5,-.055,'同图GT辅助拟合，非泛化成绩；优化未修好的目标不能据此判定原型无表达能力。',ha='center')
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(REPORT/f'NO_CANDIDATE_READOUT_20260912.{ext}',bbox_inches='tight',dpi=180)
    print(json.dumps(verification))

if __name__=='__main__':main()
