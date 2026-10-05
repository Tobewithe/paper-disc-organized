"""Build manuscript tables/figures from completed runs; no model inference."""
from pathlib import Path
import json
import csv
import hashlib
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[2]
RUNS = ROOT / 'experiments/mask_boundary_route_20260914/runs'
EVIDENCE = {}
def read(run, name='SUMMARY.json'):
    original = RUNS / run / name
    archived = PAPER / 'reproducibility/evidence' / run / name
    path = archived if archived.exists() else original
    EVIDENCE[run + '/' + name] = {'path': original.relative_to(ROOT).as_posix(),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return json.loads(path.read_text(encoding='utf-8-sig'))

m=read('RUN_e0e0d46defb74e3d9a90094f49ce26fe')
s=read('RUN_f9df48632ac742248343184263262429')
mg=read('RUN_f0836430494141479d6a36da31db7df4')
groups=read('RUN_44ab365e6ae64cb9a638557d60126b01')['models']
speed=read('RUN_e7df55bcc56541ca8c449e5065c885a4')['stats']
fit=read('RUN_d01ffdef9f3c498fa3529d96cea875eb')
factor=read('RUN_69cb2c16410147d6b6c683d751383f7c')
decisions=read('RUN_b0591a9bbbca4663995576db1448e1bd')
examples=read('RUN_8ecb45b277cc42c5a0d37217e010f5ab')
for ext in ('pdf','png'):
    source = RUNS/'RUN_8ecb45b277cc42c5a0d37217e010f5ab'/f'repair_damage_examples.{ext}'
    target = PAPER/'figures'/f'development_examples.{ext}'
    if source.exists():
        shutil.copy2(source, target)
    elif not target.exists():
        raise FileNotFoundError(f'Archived development figure missing: {target}')
metrics={'m':m['metrics'] | {'selected_global':mg['metrics']},'s':s['metrics']}
methods=[('official_zero','Baseline'),('selected_global','Global threshold'),
         ('smooth_gated','Fixed smooth'),('boundary_only','Boundary only'),
         ('risk_area','Area gate'),('risk_shape','Shape gate'),('risk_response','Response gate (ours)')]

def save_table(name,header,rows):
    with (PAPER/'tables'/f'{name}.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

rows=[];tex=[]
for size in ('m','s'):
    tex.append(r'\textit{YOLO26'+size+r'-seg'+(' (transfer)' if size=='s' else '')+r'} & & & & & & \\')
    for key,label in methods:
        vals=metrics[size][key]
        indices=[0,2,3,4,5,9]
        rows.append([size,label]+[100*v for v in vals])
        line=label+' & '+' & '.join(f'{100*vals[i]:.2f}' for i in indices)+r' \\'
        if key=='risk_response':
            line=r'\textbf{Response gate (ours)} & '+' & '.join(r'\textbf{'+f'{100*vals[i]:.2f}'+'}' for i in indices)+r' \\'
        tex.append(line)
    tex.append(r'\midrule' if size=='m' else r'\bottomrule')
(PAPER/'tables/main_results.tex').write_text('\n'.join(tex),encoding='utf8')
save_table('all_coco_metrics',['model','method','AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL'],rows)
for metric_kind,offset in [('ap',0),('ar',6)]:
    lines=[]
    for size in ('m','s'):
        for key,label in methods:
            vals=metrics[size][key][offset:offset+6]
            lines.append(size+' & '+label+' & '+' & '.join(f'{100*v:.3f}' for v in vals)+r' \\')
        if size=='m':lines.append(r'\midrule')
    (PAPER/'tables'/f'full_{metric_kind}.tex').write_text('\n'.join(lines),encoding='utf8')

lines=[]
for key,label in [('official_zero','Baseline'),('filter_only','Filtering only'),('boundary_only','Boundary only'),('smooth_gated','Boundary + filtering')]:
    ap=100*factor['metrics'][key][0]
    lines.append(label+f' & {ap:.6f} & {ap-100*factor["metrics"]["official_zero"][0]:+.6f}'+r' \\')
(PAPER/'tables/factorial.tex').write_text('\n'.join(lines),encoding='utf8')

lines=[]
for key,label in methods:
    fitkey='global_0.25' if key=='selected_global' else key
    lines.append(label+f' & {100*fit["metrics"][fitkey][0]:.5f}'+r' \\')
(PAPER/'tables/selection.tex').write_text('\n'.join(lines),encoding='utf8')
lines=[]
for key,label in [('area','Area'),('shape','Shape'),('response','Response')]:
    q=decisions['decision_groups'][key]
    lines.append(label+' & '+' & '.join(f'{100*q[g]["actual_smooth_delta_mean"]:+.3f}' for g in ('accepted','rejected'))+r' \\')
(PAPER/'tables/decision_groups.tex').write_text('\n'.join(lines),encoding='utf8')

tex=[]
for size,summary in [('m',m),('s',s)]:
    total=summary['ordinary_gt']; grp=groups[size]['groups']['all_matched']
    for key,label in [('fixed','Fixed smooth'),('area','Area gate'),('shape','Shape gate'),('response','Response gate')]:
        q=grp[key] if key in ('fixed','response') else summary['fixed_slot_outcomes'][key]
        net=q['repaired']-q['damaged']
        tex.append(f'{size} & {label} & {q["repaired"]} & {q["damaged"]} & {net} & {100*net/total:.3f}'+r' \\')
    if size=='m':tex.append(r'\midrule')
(PAPER/'tables/transitions.tex').write_text('\n'.join(tex),encoding='utf8')

tex=[]
for size in ('m','s'):
    q=groups[size]['groups']['predefined_overflow_failure']
    for key,label in [('baseline','Baseline'),('fixed','Fixed smooth'),('response','Response gate')]:
        z=q[key]
        tex.append(f'{size} & {label} & {z["mean_iou"]:.4f} & {100*z["mean_purity"]:.2f} & {100*z["mean_coverage"]:.2f} & {z["repaired"]} ({z["repair_rate_pct"]:.2f})'+r' \\')
    if size=='m':tex.append(r'\midrule')
(PAPER/'tables/failure_group.tex').write_text('\n'.join(tex),encoding='utf8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(7.1,2.45),layout='constrained')
for a,size in zip(ax,('m','s')):
    group=groups[size]['groups']['all_matched']
    data=[group['fixed'],group['response']]
    x=np.arange(2)
    a.bar(x-.18,[z['repaired'] for z in data],.34,label='Repaired',color='#147c86')
    a.bar(x+.18,[z['damaged'] for z in data],.34,label='Damaged',color='#c46b46')
    for i,z in enumerate(data):
        for dx,key in [(-.18,'repaired'),(.18,'damaged')]:a.text(i+dx,z[key]+15,str(z[key]),ha='center',fontsize=8)
    a.set_xticks(x,['Fixed smooth','Response gate']);a.set_ylim(0,1100)
    a.set_title('YOLO26'+size+'-seg'+(' · frozen transfer' if size=='s' else ''),fontsize=10)
    a.set_ylabel('Instances crossing IoU = 0.75');a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
ax[0].legend(frameon=False,ncols=2,loc='upper right',fontsize=8)
for ext in ['pdf','png','svg']:fig.savefig(PAPER/'figures'/f'repair_damage.{ext}',dpi=300)
plt.close(fig)

fig,ax=plt.subplots(figsize=(7.1,2.35));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
def box(x,y,w,h,text,color='#e9f3f4'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07,rounding_size=0.08',facecolor=color,edgecolor='#45626a',lw=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.5)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=.9,color='#45626a'))
box(.1,.8,1.8,1.4,'Frozen segmenter\nPrototypes + coefficients\nBoxes + scores')
box(2.45,1.55,1.9,.7,'Baseline mask $M_0$')
box(2.45,.3,1.9,.7,'Trial tightening $M_1$')
arrow((1.97,1.65),(2.36,1.88));arrow((1.97,1.15),(2.36,.65))
box(4.95,.8,2.1,1.4,'Geometry + response\nArea removed by trial\nPredict IoU gain')
arrow((4.42,1.9),(4.87,1.75));arrow((4.42,.65),(4.87,1.15))
box(7.65,1.55,2.15,.7,'Gain > 0: use $M_1$')
box(7.65,.3,2.15,.7,'Otherwise: retain $M_0$')
arrow((7.12,1.75),(7.57,1.89));arrow((7.12,1.14),(7.57,.65))
ax.text(5,2.7,'Training target: IoU($M_1$, GT) − IoU($M_0$, GT)',ha='center',fontsize=9,color='#147c86')
ax.text(5,.03,'Inference uses predictions only; newly empty masks revert to baseline.',ha='center',fontsize=8)
fig.tight_layout(pad=.1)
for ext in ['pdf','png','svg']:fig.savefig(PAPER/'figures'/f'method.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)

tex=[]
for key,label in [('official','Official predictor'),('chunked_zero','Chunked baseline'),('fixed_smooth','Fixed smooth'),('risk_response','Response gate')]:
    z=speed[key];tex.append(label+' & '+ ' & '.join(f'{z[k]:.2f}' for k in ['mean_ms','median_ms','p95_ms'])+r' \\')
(PAPER/'tables/latency.tex').write_text('\n'.join(tex),encoding='utf8')

bundle={'models':metrics,'groups':groups,'fit':fit,'factorial':factor,'speed':speed,'m':m,'s':s,'decisions':decisions,'development_examples':examples}
(PAPER/'tables/source_values.json').write_text(json.dumps(bundle,indent=2),encoding='utf8')
(PAPER/'EVIDENCE_SOURCES.json').write_text(json.dumps(EVIDENCE,indent=2),encoding='utf8')
print('Built tables, vector figures and evidence map from',len(EVIDENCE),'completed result files.')
