"""Review completed official matches; no inference or training."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np

def read(path):
    with path.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument("--study",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.study/"protocol.json").read_text(encoding="utf-8"))
    source=a.study/"runs"/cfg["run_id"]
    assert json.loads((source/"COMPLETE.json").read_text(encoding="utf-8"))["status"]=="complete"
    a.out.mkdir(parents=True,exist_ok=True)
    contrasts=read(source/"recall_contrasts.csv")
    thresholds=read(source/"threshold_transitions.csv")
    cats=read(source/"category_contributions.csv")
    mechanisms=read(source/"opportunity_transitions.csv")
    lines=["# 小目标召回的逐实例归因","","全部 5,000 张 COCO val2017，15,264 个 small 普通 GT，80 类。复用第 3 轮 last 预测，不训练或推理。","",
           "## 官方 small Mask AR 与实例平均的差异","",
           "| 分支 | 对照 | 类别平均增量 [95% CI] | 实例平均增量 |","|---|---|---|---:|"]
    for r in contrasts:
        if r["task"]=="segm":
            lines.append(f"| {r['branch']} | {r['contrast']} | {float(r['delta_points']):+.3f} [{float(r['ci_low_points']):+.3f}, {float(r['ci_high_points']):+.3f}] | {float(r['micro_delta_points']):+.3f} |")
    lines+=["","数值为点。2,000 次按图片配对 bootstrap；每次按类别平均，抽样后无 GT 的类别省略。单训练 seed，未做多指标校正，不包含训练随机性。","",
            "## IoU 阈值的新增与丢失","","| 分支 | 对照 | 阈值 | 新增 | 丢失 | 净变化 | 类别平均增量 |","|---|---|---:|---:|---:|---:|---:|"]
    for r in thresholds:
        if r["task"]=="segm":
            lines.append(f"| {r['branch']} | {r['contrast']} | {r['threshold']} | {r['gained']} | {r['lost']} | {r['net']} | {float(r['macro_delta_points']):+.3f} |")
    names={"qualifying_mask_already_present_but_unmatched":"已有合格 mask，但未获唯一匹配",
           "mask_opportunity_added_with_existing_box_support":"已有合格 box，缺少合格 mask",
           "mask_opportunity_added_without_previous_same_threshold_box":"同阈值 box 与 mask 均不合格"}
    grouped=defaultdict(lambda:dict(gained=0,lost=0,contribution=0.))
    for r in mechanisms:
        group=grouped[(r["branch"],r["contrast"],r["opportunity_state"])]
        group[r["direction"]]+=int(r["count"]);group["contribution"]+=float(r["ar_small_contribution_points"])
    lines+=["","## 未匹配一侧的候选条件","","| 分支 | 对照 | 候选条件 | 新增事件 | 丢失事件 | 净 AR 贡献 |","|---|---|---|---:|---:|---:|"]
    grouped_rows=[]
    for (b,c,m),r in grouped.items():
        lines.append(f"| {b} | {c} | {names[m]} | {r['gained']} | {r['lost']} | {r['contribution']:+.3f} |")
        grouped_rows.append(dict(branch=b,contrast=c,opportunity=m,**r))
    write(a.out/"opportunity_summary.csv",grouped_rows)
    lines+=["","事件为 GT × IoU 阈值，不是独立实例数。检查的是导出后同类前 100 个候选，机会值不是唯一匹配召回，也不是 NMS 前候选；跨模型最佳候选没有固定身份，因此此表不是因果机制分解。","",
            "## 类别构成与样本量敏感性","",
            "30/100 个 GT 下限仅作事后描述，不取代官方评价或成为筛选规则。"]
    sensitivity=[]
    for b in cfg["branches"]:
        for ref in ["baseline","reg_only"]:
            c="reg_scheduled minus "+ref
            rr=[r for r in cats if r["task"]=="segm" and r["branch"]==b and r["contrast"]==c]
            lines+=["",f"### {b} 相对 {ref}","",
                    f"上升 {sum(float(r['delta_points'])>1e-10 for r in rr)} 类；下降 {sum(float(r['delta_points'])< -1e-10 for r in rr)} 类；其余近似持平。","",
                    "| 每类 small GT 下限 | 类别数 | 保留类别平均 AR 增量 | 对原官方 AR 的贡献 |","|---|---:|---:|---:|"]
            for n in [0,30,100]:
                sel=[r for r in rr if int(r["small_gt"])>=n]
                mean=sum(float(r["delta_points"]) for r in sel)/len(sel)
                contribution=sum(float(r["macro_contribution_points"]) for r in sel)
                sensitivity.append(dict(branch=b,contrast=c,minimum_small_gt=n,categories=len(sel),mean_delta_points=mean,official_contribution_points=contribution))
                lines.append(f"| {n} | {len(sel)} | {mean:+.3f} | {contribution:+.3f} |")
            lines+=["","| 类别 | small GT | 类内 AR 增量 | 对官方 AR 的贡献 |","|---|---:|---:|---:|"]
            ordered=sorted(rr,key=lambda r:float(r["macro_contribution_points"]),reverse=True)
            for r in ordered[:8]+ordered[-5:]:
                lines.append(f"| {r['category_name']} | {r['small_gt']} | {float(r['delta_points']):+.3f} | {float(r['macro_contribution_points']):+.3f} |")
    write(a.out/"category_count_sensitivity.csv",sensitivity)
    # Exact contribution of each GT and leverage of individual images.
    index=dict(np.load(source/"annotations.npz"))
    ids=index["annotation_ids"];img=index["image_ids"];cat=index["category"]
    gt_rows,image_rows,distribution=[],[],[]
    for b in cfg["branches"]:
        method=dict(np.load(source/(b+"__reg_scheduled.npz")))
        for ref in ["baseline","reg_only"]:
            old=dict(np.load(source/(b+"__"+ref+".npz")))
            valid=old["segm_small_valid"];cats_present=sorted(set(cat[valid]))
            n={c:int((valid & (cat==c)).sum()) for c in cats_present}
            diff=(method["segm_small_matches"]>0).mean(1)-(old["segm_small_matches"]>0).mean(1)
            hit_old=old["segm_small_matches"]>0;hit_new=method["segm_small_matches"]>0
            image_sums=defaultdict(float)
            distribution.append(dict(branch=b,reference=ref,improved=int((valid & (diff>0)).sum()),
                                     worsened=int((valid & (diff<0)).sum()),unchanged=int((valid & (abs(diff)<1e-12)).sum()),
                                     gain_any_threshold=int((valid & (hit_new & ~hit_old).any(1)).sum()),
                                     loss_any_threshold=int((valid & (hit_old & ~hit_new).any(1)).sum())))
            for i in np.flatnonzero(valid):
                contribution=100*diff[i]/(len(cats_present)*n[cat[i]])
                image_sums[int(img[i])]+=contribution
                if abs(contribution)>1e-12:
                    gt_rows.append(dict(branch=b,reference=ref,annotation_id=int(ids[i]),image_id=int(img[i]),
                                        category_id=int(cat[i]),category_small_gt=n[cat[i]],mean_recall_change=float(diff[i]),
                                        macro_contribution_points=float(contribution)))
            for image_id,total in image_sums.items():
                image_rows.append(dict(branch=b,reference=ref,image_id=image_id,macro_contribution_points=total))
    write(a.out/"per_gt_ar_contribution.csv",gt_rows);write(a.out/"per_image_ar_contribution.csv",image_rows)
    write(a.out/"per_gt_change_distribution.csv",distribution)
    lines+=["","## 最大的单实例贡献","","| 分支 | 对照 | annotation ID | image ID | 类别 ID | 该类 small GT | 对 AR 的贡献 |","|---|---|---:|---:|---:|---:|---:|"]
    for b in cfg["branches"]:
        selected=sorted([r for r in gt_rows if r["branch"]==b and r["reference"]=="baseline"],key=lambda r:r["macro_contribution_points"],reverse=True)[:10]
        for r in selected:
            lines.append(f"| {b} | baseline | {r['annotation_id']} | {r['image_id']} | {r['category_id']} | {r['category_small_gt']} | {r['macro_contribution_points']:+.3f} |")
    (a.out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (a.out/"COMPLETE.json").write_text(json.dumps(dict(status="complete",source_run=cfg["run_id"],training=False,inference=False,
                                                     scope="Post hoc category count and individual contribution analysis; no new evaluation"),indent=2),encoding="utf-8")
    print(json.dumps(dict(sensitivity=sensitivity,distribution=distribution),indent=2))

if __name__=="__main__":main()
