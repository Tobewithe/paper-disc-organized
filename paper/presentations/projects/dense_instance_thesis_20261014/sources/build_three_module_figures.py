"""Extend existing figures with E5 source data; no model execution or new evaluation."""
from pathlib import Path
import csv, hashlib, json, sys
sys.path.insert(0,str(Path.home()/'.cache/codex-plot-libs'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
base=Path(__file__).resolve().parents[1]
root=base.parents[3]
out=base/'three_module_figures_20261015';out.mkdir(exist_ok=True)
data=json.loads((base/'experiment_figures_20261015/chart_data.json').read_text(encoding='utf-8'))
rel='experiments/mask_readout_large_mask_gate_20261015/runs/RUN_LARGE_MASK_FULL_6a2709ae1cd24429b29a3c6d6f1600b8'
def read(p):
    b=(root/p).read_bytes();data['source_hashes'].append({'path':p,'sha256':hashlib.sha256(b).hexdigest()})
    return json.loads(b.decode('utf-8-sig'))
r=read(rel+'/run.json');assert r['execution_status']=='completed' and r['exit_code']==0
s=read(rel+'/SUMMARY.json')
assert all(s['gates'].values())
e3=read(data['run_sources']['E3']+'/SUMMARY.json')['B']
data['run_sources']['E5']=rel
effects=[]
for name,summary in [('无条件修正',e3),('面积选择',s)]:
    for cohort,label in [('target','目标组'),('all_bbox75','整体配对')]:
        d=summary[cohort];effects.append(dict(method=name,cohort=cohort,label=label,mean_pp=d['mean']*100,low_pp=d['ci95'][0]*100,high_pp=d['ci95'][1]*100))
damage=[]
for name,summary in [('无条件修正',e3),('面积选择',s)]:
    d=summary['damage'];damage.append(dict(method=name,n=d['damaged'],denominator=d['sample_size'],percent=d['point_rate']*100,upper_percent=d['upper_rate']*100))
data['selective_effects']=effects;data['selective_damage']=damage;data['selective_coverage']=s['coverage']
data['selective_scope']='Outcome-exposed development panel; nominal original intervals; no fresh independent 95% coverage guarantee; no full 5000-image AP gain.'
(out/'chart_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for key in ['selective_effects','selective_damage']:
    with (out/(key+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[key][0]));w.writeheader();w.writerows(data[key])
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':12,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#172B46'})
colors=['#C36D37','#007F82']
fig,axes=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1.6,1]})
ax=axes[0];ax.axvline(0,color='#596B80',ls='--',lw=1)
ordered=[effects[0],effects[2],effects[1],effects[3]]
for i,d in enumerate(ordered):
    ax.errorbar(d['mean_pp'],i,xerr=[[d['mean_pp']-d['low_pp']],[d['high_pp']-d['mean_pp']]],fmt='o',color=colors[i%2],capsize=5,lw=2)
    ax.annotate(f"{d['mean_pp']:+.3f}",(d['mean_pp'],i),xytext=(0,12),textcoords='offset points',ha='center')
ax.set_yticks(range(4),[d['label']+' · '+d['method'] for d in ordered]);ax.set_ylim(3.6,-.7);ax.set_xlim(-.5,2);ax.set_xlabel('图宏平均 ΔMaskIoU（百分点）');ax.set_title('轮廓质量变化')
ax=axes[1];ax.bar(range(2),[d['percent'] for d in damage],color=colors,width=.5)
ax.set_ylim(0,6.3);ax.set_xticks(range(2),[d['method'] for d in damage]);ax.set_ylabel('原成功任务损伤率（%）');ax.set_title('同一成功面板的损伤')
for i,d in enumerate(damage):ax.text(i,d['percent']+.15,f"{d['percent']:.2f}%\n{d['n']}/{d['denominator']}",ha='center')
fig.tight_layout();fig.text(.02,-.06,'E3/E5；同 checkpoint、同开发集和面板；原生面积≥96²；复用已观察开发数据，区间无新的独立覆盖保证。',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(out/('06_选择性修正开发结果.'+ext),dpi=220,bbox_inches='tight',facecolor='white')
plt.close(fig)
print(json.dumps({'selective_effects':effects,'selective_damage':damage},ensure_ascii=False,indent=2))
