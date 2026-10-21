"""Extract existing Run results and draw presentation figures; no model execution."""
from pathlib import Path
import csv
import hashlib
import json
import sys

sys.path.insert(0, str(Path.home() / '.cache/codex-plot-libs'))
BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
OUT = BASE / 'experiment_figures_20261015'
OUT.mkdir(exist_ok=True)
RUNS = {
    'E0': 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170',
    'E1': 'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949',
    'E2': 'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_EVAL_a24a564568c943eab362784abfbe2c54',
    'E3': 'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0',
    'E4': 'experiments/mask_readout_stratified_audit_20261015/runs/RUN_STRATIFIED_FULL_c1066461caf64b9cb18d2563df4ca643',
}
sources = []
def read(rel):
    f = ROOT / rel
    data = f.read_bytes()
    sources.append({'path': rel, 'sha256': hashlib.sha256(data).hexdigest()})
    return json.loads(data.decode('utf-8-sig'))

summaries = {}
for key, rel in RUNS.items():
    run = read(rel + '/run.json')
    assert run['execution_status'] == 'completed' and run['exit_code'] == 0, key
    summaries[key] = read(rel + '/SUMMARY.json')
diagnostic = read('experiments/official_gt_failure_rebaseline_20261013/runs/RUN_HISTORY_RECONCILIATION_ea22c6c2ae3c49eda6899b1ea98f40f4/NATIVE_LEDGER_DIAGNOSTICS.json')
for name in ['CONFIG_AUDIT.json', 'LOCAL_INDEPENDENT_RECHECK.json']:
    assert read(RUNS['E4'] + '/' + name)['status'] == 'passed'

pop = diagnostic['populations']
failure_n = summaries['E0']['ordinary_gt'] - summaries['E0']['mask75_success']
assert pop['F_without_box75'] + pop['T'] + pop['U'] == failure_n == 13378
baseline = [
    {'label': '无合格框配对', 'n': pop['F_without_box75']},
    {'label': '框合格、轮廓不合格 T', 'n': pop['T']},
    {'label': '匹配竞争案例', 'n': pop['U']},
]
for row in baseline:
    row['denominator'] = failure_n
    row['percent'] = row['n'] / failure_n * 100

s1 = summaries['E1']
oracle = []
for key, label in [('small','小目标'),('medium','中目标'),('large','大目标')]:
    v = s1['by_gt_area'][key]
    oracle.append({'label':label, 'n':v['found'], 'denominator':v['denominator'], 'percent':100*v['found']/v['denominator']})
assert sum(r['n'] for r in oracle) == s1['found_same_detection_mask75'] == 122
assert sum(r['denominator'] for r in oracle) == s1['selected_T'] == 256

pilot = []
for key, arm, label in [('E2','C','空间打乱（BCE）'),('E2','B','保留空间（BCE）'),('E3','B','保留空间（soft-IoU）')]:
    v = summaries[key][arm]['target']
    pilot.append({'label':label, 'mean_pp':v['mean']*100, 'low_pp':v['ci95'][0]*100, 'high_pp':v['ci95'][1]*100, 'GT':655, 'images':v['images'], 'source':key})

strata = []
for area, label in [('S','小掩码'),('M','中掩码'),('L','大掩码')]:
    rows = [r for r in summaries['E4']['tables'] if r['area_definition']=='native_area_group' and r['crowd_definition']=='all_crowd_group' and r['area']==area and r['crowd']=='All']
    success = next(r for r in rows if r['cohort']=='success_panel')
    target = next(r for r in rows if r['cohort']=='target_T')
    damage = success['task_damage']
    strata.append({'label':label,'group':area,'n':damage['numerator'],'denominator':damage['denominator'],'percent':100*damage['rate'],'low_percent':100*damage['cluster95']['interval'][0],'high_percent':100*damage['cluster95']['interval'][1], 'T_n':target['n'],'T_delta_pp':100*target['stages']['binary_original']['image_macro_mean']})
assert sum(r['n'] for r in strata)==53 and sum(r['denominator'] for r in strata)==1024

contrasts = []
for factor,label in [('area','小掩码 − 中大掩码'),('crowd','高重叠 − 低重叠')]:
    v=next(r for r in summaries['E4']['contrasts'] if r['area_definition']=='native_area_group' and r['factor']==factor and r['endpoint']=='damage')
    contrasts.append({'label':label,'factor':factor,'mean_pp':100*v['estimate'],'low_pp':100*v['familywise95']['interval'][0],'high_pp':100*v['familywise95']['interval'][1], 'stable':v['stable_exploratory_heterogeneity']})

data = {'baseline':baseline,'oracle':oracle,'pilot':pilot,'damage_by_native_area':strata,'adjusted_contrasts':contrasts,
        'soft_guardrails':summaries['E3']['B']['damage'] | {'all_bbox75_delta':summaries['E3']['B']['all_bbox75']['mean']},
        'oracle_scope':'GT-assisted finite coefficient optimization on 256 selected T; same-detection MaskIoU>=.75, not deployment recovery or AP',
        'run_sources':RUNS,'source_hashes':sources}
(OUT/'chart_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name, rows in [(k,data[k]) for k in ['baseline','oracle','pilot','damage_by_native_area','adjusted_contrasts']]:
    with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in data.items() if k not in ['source_hashes','run_sources','soft_guardrails']},ensure_ascii=False,indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':13,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#D4DEE7','text.color':'#172B46','axes.labelcolor':'#25374C','xtick.color':'#596B80','ytick.color':'#25374C'})
BLUE, TEAL, ORANGE, GRAY = '#315E9D','#007F82','#C36D37','#596B80'
def save(fig,name):
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/(name+'.'+ext),dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(10,4.3)); vals=[r['n'] for r in baseline]
ax.barh(range(3),vals,color=[GRAY,TEAL,ORANGE],height=.5)
ax.set_yticks(range(3),[r['label'] for r in baseline]); ax.invert_yaxis();ax.set_xlim(0,8000);ax.set_xlabel('分割失败实例数')
for i,r in enumerate(baseline):ax.text(r['n']+100,i,f"{r['n']:,}（{r['percent']:.2f}%）",va='center',fontsize=12)
ax.set_title('分割失败实例的进一步划分',loc='left',weight='bold',pad=18)
fig.text(.02,-.04,'E0；COCO val2017 5,000 图；分割失败 GT=13,378；不等同于 AP。',fontsize=10,color=GRAY)
save(fig,'01_失败实例划分')

fig,ax=plt.subplots(figsize=(9,4.3)); ax.bar(range(3),[r['percent'] for r in oracle],color=TEAL,width=.5)
ax.set_ylim(0,106);ax.set_ylabel('同一检测达到 MaskIoU≥0.75（%）');ax.set_xticks(range(3),[r['label'] for r in oracle])
for i,r in enumerate(oracle):ax.text(i,r['percent']+2,f"{r['percent']:.2f}%\n{r['n']}/{r['denominator']}",ha='center',fontsize=12)
ax.set_title('有限系数优化找到的恢复机会',loc='left',weight='bold',pad=18)
fig.text(.02,-.04,'E1；256 个固定目标，总计 122/256；按 GT 面积分组；标注辅助实验，非部署效果。',fontsize=10,color=GRAY)
save(fig,'02_有限修正空间')

fig,ax=plt.subplots(figsize=(10,4.3));ax.axvline(0,color=GRAY,ls='--',lw=1)
for i,r in enumerate(pilot):
    ax.errorbar(r['mean_pp'],i,xerr=[[r['mean_pp']-r['low_pp']],[r['high_pp']-r['mean_pp']]],fmt='o',color=[GRAY,BLUE,TEAL][i],capsize=5,lw=2)
    ax.annotate(f"{r['mean_pp']:+.3f}",(r['mean_pp'],i),xytext=(0,13),textcoords='offset points',ha='center')
ax.set_yticks(range(3),[r['label'] for r in pilot]);ax.set_ylim(2.5,-.6);ax.set_xlim(-4,2);ax.set_xlabel('目标组图宏平均 ΔMaskIoU（百分点）')
ax.set_title('自动修正探索的目标组结果',loc='left',weight='bold',pad=18)
fig.text(.02,-.04,'E2/E3；同 512 张开发图，T=655；误差线为原协议 95% 区间；各配置均未通过全部验收。',fontsize=10,color=GRAY)
save(fig,'03_自动修正探索')

fig,ax=plt.subplots(figsize=(9,4.3));ax.bar(range(3),[r['percent'] for r in strata],width=.5,color=[ORANGE,BLUE,TEAL])
ax.set_ylim(0,18);ax.set_ylabel('原成功实例的损伤率（%）');ax.set_xticks(range(3),[r['label'] for r in strata])
for i,r in enumerate(strata):ax.text(i,r['percent']+.4,f"{r['percent']:.2f}%\n{r['n']}/{r['denominator']}",ha='center')
ax.set_title('按原生预测掩码面积分组的损伤',loc='left',weight='bold',pad=18)
fig.text(.02,-.04,'E4；原成功面板 n=1,024；S<32²，M∈[32²,96²)，L≥96²（原图像素）；描述性结果。',fontsize=10,color=GRAY)
save(fig,'04_面积与损伤')

fig,ax=plt.subplots(figsize=(10,4.3));ax.axvline(0,color=GRAY,ls='--',lw=1)
for i,r in enumerate(contrasts):
    ax.errorbar(r['mean_pp'],i,xerr=[[r['mean_pp']-r['low_pp']],[r['high_pp']-r['mean_pp']]],fmt='o',color=[ORANGE,BLUE][i],capsize=5,lw=2)
    ax.annotate(f"{r['mean_pp']:+.3f}",(r['mean_pp'],i),xytext=(0,13),textcoords='offset points',ha='center')
ax.set_yticks(range(2),[r['label'] for r in contrasts]);ax.set_ylim(1.5,-.6);ax.set_xlim(-5,20);ax.set_xlabel('控制另一因素后的损伤率差（百分点）')
ax.set_title('面积与邻框重叠的分层对比',loc='left',weight='bold',pad=18)
fig.text(.02,-.04,'E4；n=1,024；区间已校正 8 项比较；高重叠≥0.5，低重叠<0.1；探索性关联。',fontsize=10,color=GRAY)
save(fig,'05_面积与邻框重叠对比')
