"""Summarize completed frozen transfer without selecting or refitting any rule."""
import argparse,hashlib,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.project);study=root/'experiments/mask_boundary_route_20260914';runs=study/'runs';out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
protocol=json.loads((study/'PROTOCOL_TRANSFER_V5.json').read_text(encoding='utf-8'))
source=runs/protocol['run_ids']['transfer_eval'];data=json.loads((source/'SUMMARY.json').read_text())
assert data['images']==4500 and data['selected_global_evaluated']
labels={'official_zero':'原基线','smooth_gated':'固定连续规则','boundary_only':'固定连续规则，保留候选',
    'risk_area':'冻结面积门控','risk_shape':'冻结面积+形状门控','risk_response':'冻结面积+形状+响应门控','selected_global':'冻结全局阈值0.25'}
lines=['# YOLO26m → YOLO26s 冻结迁移结果','','校准器仅在m权重的train2017预测上拟合；s权重未重新拟合或调参。评价为此前4500张COCO val2017。',
    '','| 方法 | Mask AP | 相对s基线AP点 | AP75 | small AP | medium AP | large AP |','|---|---:|---:|---:|---:|---:|---:|']
for key,label in labels.items():
    m=data['metrics'][key]
    lines.append(f'| {label} | {m[0]*100:.4f} | {data["delta_ap_points"][key]:+.4f} | {m[2]*100:.4f} | {m[3]*100:.4f} | {m[4]*100:.4f} | {m[5]*100:.4f} |')
lines+=['','| 冻结门控 | 修复 | 损伤 | 净修复 | R75差值百分点 |','|---|---:|---:|---:|---:|']
for mode in ('area','shape','response'):
    v=data['fixed_slot_outcomes'][mode]
    lines.append(f'| {mode} | {v["repaired"]} | {v["damaged"]} | {v["net"]} | {v["r75_delta_pp"]:+.4f} |')
risk=data['metrics']['risk_response'][0]*100
summary=dict(source_run=source.name,images=data['images'],ordinary_gt=data['ordinary_gt'],
    response_vs_baseline_ap_pp=data['delta_ap_points']['risk_response'],
    response_vs_smooth_ap_pp=risk-data['metrics']['smooth_gated'][0]*100,
    response_vs_selected_global_ap_pp=risk-data['metrics']['selected_global'][0]*100,
    paired_r75=data['paired_r75_comparisons'],
    limitations=['same previously explored COCO images, not new blind data','one transfer checkpoint','no AP uncertainty estimate','cost measured separately on m, not s'])
lines+=['',f'响应门控比固定连续规则 {summary["response_vs_smooth_ap_pp"]:+.4f} AP点，比冻结全局0.25 {summary["response_vs_selected_global_ap_pp"]:+.4f} AP点。']
for key,v in data['paired_r75_comparisons'].items():
    lo,hi=v['paired_image_bootstrap_95ci_pp'];lines.append(f'- {key}: R75 {v["delta_r75_pp"]:+.4f} 点，图片配对95%区间 [{lo:.4f}, {hi:.4f}]。')
lines+=['','这些数字检验冻结规则能否跨同系列权重迁移，不能证明网络训练失败根因。不能把实例R75区间当作AP显著性检验。',
    '额外计算代价以独立m权重速度Run为准；不将其外推为s权重速度。',
    f'来源：{source.as_posix()}/SUMMARY.json']
for name,value in [('SUMMARY.json',json.dumps(summary,indent=2)),('RESULTS.md','\n'.join(lines)+'\n')]:
    temp=out/(name+'.tmp');temp.write_text(value,encoding='utf-8');os.replace(temp,out/name)
print(json.dumps(summary),flush=True)
