"""Create static evidence plots for frozen train-calibrated gates."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--fit',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();source=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 s=json.loads((source/'SUMMARY.json').read_text());fit=json.loads((Path(a.fit)/'SUMMARY.json').read_text())
 methods=['smooth_gated','risk_area','risk_shape','risk_response'];labels=['Fixed smooth','Learned area','+ Shape','+ Response']
 gains=[s['delta_ap_points'][k] for k in methods]
 repair=[907]+[s['fixed_slot_outcomes'][k]['repaired'] for k in ('area','shape','response')]
 harm=[576]+[s['fixed_slot_outcomes'][k]['damaged'] for k in ('area','shape','response')]
 fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
 axes[0].bar(labels,gains,color=['#94a3b8','#7296ad','#4f879b','#087f8c'])
 axes[0].set_ylabel('Mask AP improvement (points)');axes[0].set_title('4500 COCO validation images')
 for i,x in enumerate(gains):axes[0].text(i,x+.012,f'+{x:.3f}',ha='center')
 axes[0].set_ylim(0,max(gains)*1.2)
 y=np.arange(4);axes[1].bar(y-.18,repair,.36,label='Repaired',color='#087f8c');axes[1].bar(y+.18,harm,.36,label='Damaged',color='#bd6656')
 axes[1].set_xticks(y,labels);axes[1].set_ylabel('Instances at Mask IoU 0.75');axes[1].set_title('Repair and damage together');axes[1].legend()
 for ext in ('png','pdf','svg'):fig.savefig(out/f'risk_calibration.{ext}',dpi=180)
 plt.close(fig)
 rows=list(csv.DictReader((source/'decisions.csv').open(newline='',encoding='utf-8')))
 decision={}
 for mode in ('area','shape','response'):
  rr=[r for r in rows if r['mode']==mode]
  accepted=[r for r in rr if int(r['calibrated'])]
  rejected=[r for r in rr if not int(r['calibrated'])]
  def stats(items):
   if not items:return {'n':0}
   d=np.array([float(r['fixed_smooth_iou'])-float(r['baseline_iou']) for r in items])
   return dict(n=len(items),actual_smooth_delta_mean=float(d.mean()),positive_fraction=float((d>0).mean()),negative_fraction=float((d<0).mean()))
  decision[mode]=dict(accepted=stats(accepted),rejected=stats(rejected))
 result=dict(run_id=os.environ.get('RESEARCH_RUN_ID'),source_run=source.name,fit_run=Path(a.fit).name,
    response_vs_fixed_ap_points=s['delta_ap_points']['risk_response']-s['delta_ap_points']['smooth_gated'],
    response_vs_baseline_ap_points=s['delta_ap_points']['risk_response'],avoided_damage=576-harm[-1],lost_repairs=907-repair[-1],
    decision_groups=decision,limitations=['Fixed smooth repair/damage reproduced in prior factorial run','Illustration is point estimates, no AP uncertainty interval'])
 tmp=out/'SUMMARY.json.tmp';tmp.write_text(json.dumps(result,indent=2),encoding='utf-8');os.replace(tmp,out/'SUMMARY.json')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
