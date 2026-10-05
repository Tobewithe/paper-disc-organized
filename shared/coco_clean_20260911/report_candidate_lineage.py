"""Local artifact verification, deterministic plots, and evidence tables."""
import csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

BASE=Path(__file__).resolve().parent
OUT=BASE/'diagnostics/candidate_lineage300_20260912'
REPORT=BASE.parents[1]/'refine-logs/coco-structure'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    manifest=json.loads((OUT/'ANALYSIS_COMPLETE.json').read_text())
    for name,digest in {**manifest['hashes'],**manifest['prediction_hashes']}.items():
        assert sha(OUT/name)==digest,name
    assert manifest['script_sha256']==sha(BASE/'summarize_candidate_lineage.py')
    final_manifest=OUT/'FINAL_ARTIFACT_SHA256SUMS.txt'
    checked=0
    if final_manifest.exists():
        for line in final_manifest.read_text().splitlines():
            digest,name=line.split('  ',1)
            assert sha(OUT/name)==digest,name
            checked+=1
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['script_sha256']==sha(BASE/'candidate_lineage_probe.py')
    analysis=json.loads((OUT/'ANALYSIS.json').read_text())
    stats=read(OUT/'task_summary.csv');sp=read(OUT/'spatial.csv');arms=[r['arm'] for r in stats]
    assert len(protocol['images'])==300
    for arm in arms:
        rr=[r for r in sp if r['arm']==arm]
        assert len(rr)==3229
        for r in rr:
            assert all(np.isfinite(float(r[k])) for k in ['mask_iou','coverage','same_neighbor','background'])
    fp=read(OUT/'official_false_positives.csv')
    preds=[]
    for arm in arms:
        rr=[r for r in fp if r['arm']==arm]
        preds.append(dict(arm=arm,tp75=sum(int(r['tp75']) for r in rr),fp75=sum(int(r['fp75']) for r in rr),
            duplicate_like_fp75=sum(int(r['duplicate_like_fp75']) for r in rr)))
    verified=dict(status='PASS',verification='Downloaded file hashes, local source hashes, finite per-target metrics and group counts; not independent model inference or official AP reproduction.',
        files_verified=len(manifest['hashes'])+len(manifest['prediction_hashes']),final_manifest_files_verified=checked,images=300,
        per_arm_spatial_targets=3229,predictions=preds,tar_sha256=sha(OUT.with_suffix('.tar.gz')))
    (OUT/'LOCAL_VERIFICATION.json').write_text(json.dumps(verified,indent=2),encoding='utf-8')
    chinese=Path('C:/Windows/Fonts/msyh.ttc')
    font=FontProperties(fname=str(chinese))
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),gridspec_kw={'width_ratios':[1,1,1.2]})
    labels=['原预测','只换系数','只换框','两者都换'];colors=['#667788','#007f83','#d89436','#7556a3']
    for k,field in enumerate(['mask_ap','r75_high']):
        vals=np.array([float(r[field])*100 for r in stats]);x=np.arange(4)
        ax[k].bar(x,vals,color=colors,width=.67)
        for j,v in enumerate(vals):ax[k].text(j,v+.3,f'{v:.2f}',ha='center',fontsize=10)
        ax[k].set_xticks(x,labels,rotation=18);ax[k].set_ylim(0,max(vals)*1.14)
        ax[k].set_title(['整图 Mask AP','高拥挤实例 R75（%）'][k]);ax[k].grid(axis='y',alpha=.2);ax[k].set_axisbelow(True)
    fields=['mask_iou','coverage','same_neighbor','background'];names=['掩码 IoU','自身覆盖','同类邻居错误','背景错误']
    rr=[next(r for r in analysis['paired_differences'] if r['arm']=='coefficient_only' and r['group']=='high' and r['metric']==field) for field in fields]
    vals=np.array([r['difference_pp'] for r in rr]);ci=np.array([r['ci95_pp'] for r in rr])
    ax[2].barh(np.arange(4),vals,color=['#007f83']*4,xerr=np.array([vals-ci[:,0],ci[:,1]-vals]),capsize=3)
    ax[2].set_yticks(np.arange(4),names);ax[2].invert_yaxis();ax[2].axvline(0,color='gray',lw=1)
    ax[2].set_title('只换系数：高组像素变化');ax[2].set_xlabel('相对原预测变化（百分点），95% 图级配对区间')
    for j,v in enumerate(vals):ax[2].text((ci[j,1]+.2) if v>0 else (ci[j,0]-.2),j,f'{v:+.2f}',ha='left' if v>0 else 'right',va='center')
    ax[2].set_xlim(-8,5.2)
    fig.suptitle('COCO 300 图候选诊断：已有候选包含更好的读出，但当前保留结果未充分利用',fontsize=14,y=1.015)
    fig.text(.5,-.05,'GT 辅助选候选，仅为诊断机会，不是可部署方法成绩。模型、原型、预测槽位和分数冻结；无训练。',ha='center',fontsize=10)
    fig.tight_layout()
    for ext in ['png','svg','pdf']:
        fig.savefig(REPORT/f'CANDIDATE_LINEAGE_20260912.{ext}',dpi=190,bbox_inches='tight')
    print(json.dumps(verified,ensure_ascii=False))


if __name__=='__main__':main()
