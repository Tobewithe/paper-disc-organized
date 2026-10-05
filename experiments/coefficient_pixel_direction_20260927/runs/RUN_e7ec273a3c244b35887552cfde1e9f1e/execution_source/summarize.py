"""Image-clustered paired summaries; reused exploratory test, two training seeds."""
import argparse,json
import numpy as np
from common import *

def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    d=np.load(a.evaluation/"METRICS.npz")
    values=d["values"].astype(float);names=d["names"].tolist();cols=d["metrics"].tolist()
    keys=d["keys"];iid=keys[:,0];unique,inv=np.unique(iid,return_inverse=True)
    rng=np.random.default_rng(20260927)
    weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000).astype(float)
    base=values[:,names.index("original")]
    m={k:cols.index(k) for k in cols}
    groups=dict(all=np.ones(len(iid),bool),failure=base[:,m["iou"]]<.75,
        good_box_failure=(base[:,m["iou"]]<.75)&(d["box_iou"]>=.75),
        original_success=base[:,m["iou"]]>=.75)
    def estimate(v,sel):
        valid=sel&np.isfinite(v)
        if not valid.any():return dict(mean=None,ci95=[None,None],n=0,images=0)
        sums=np.bincount(inv[valid],weights=v[valid],minlength=len(unique))
        counts=np.bincount(inv[valid],minlength=len(unique))
        denominator=weights@counts
        boot=(weights@sums)/np.maximum(denominator,1)
        boot=boot[denominator>0]
        p=2*min((np.sum(boot<=0)+1)/(len(boot)+1),(np.sum(boot>=0)+1)/(len(boot)+1))
        return dict(mean=float(v[valid].mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
            p_bootstrap=min(p,1.),n=int(valid.sum()),images=int(np.unique(iid[valid]).size))
    # Pair seed results first; image CI is conditional on these two fitted seeds.
    arms={name:values[:,j] for j,name in enumerate(names[:4])}
    seed_members={}
    for mode in MODES:
        for kind in KINDS:
            key=f"{mode}_{kind}"
            members=[names.index(f"{key}_s{seed}") for seed in (0,1)]
            arms[key]=values[:,members].mean(1);seed_members[key]=members
            if kind=="spatial":
                members=[names.index(f"{key}_s{seed}_projected") for seed in (0,1)]
                arms[key+"_projected"]=values[:,members].mean(1);seed_members[key+"_projected"]=members
    output=dict(images=len(unique),candidates=len(iid),ci="2000 image-clustered paired bootstrap; conditional on two fitted seeds",
        projection_rank=dict(min=int(d["rank"].min()),median=float(np.median(d["rank"])),max=int(d["rank"].max())),
        results={},primary_contrasts={})
    for group,sel in groups.items():
        result={}
        for arm,v in arms.items():
            member=seed_members[arm] if arm in seed_members else [names.index(arm)]
            actual=values[:,member,m["iou"]]
            before=base[:,m["iou"]]
            net=((actual>=.75).astype(float)-(before[:,None]>=.75)).mean(1)
            result[arm]=dict(effect_cos=estimate(v[:,m["effect_cos"]],sel),
                delta_iou=estimate(v[:,m["iou"]]-before,sel),
                delta_auc=estimate(v[:,m["auc"]]-base[:,m["auc"]],sel),
                delta_coverage=estimate(v[:,m["coverage"]]-base[:,m["coverage"]],sel),
                delta_fpr=estimate(v[:,m["fpr_native"]]-base[:,m["fpr_native"]],sel),
                net_mask75=estimate(net,sel),
                coefficient_norm_median=float(np.nanmedian(v[sel,m["coeff_norm"]])) if np.isfinite(v[sel,m["coeff_norm"]]).any() else None,
                projection_residual_median=float(np.nanmedian(v[sel,m["projection_residual"]])) if np.isfinite(v[sel,m["projection_residual"]]).any() else None,
                repairs_per_seed=[int(((before<.75)&(actual[:,k]>=.75)&sel).sum()) for k in range(len(member))],
                damage_per_seed=[int(((before>=.75)&(actual[:,k]<.75)&sel).sum()) for k in range(len(member))],
                delta_iou_per_seed=[float((actual[:,k]-before)[sel].mean()) for k in range(len(member))])
        output["results"][group]=result
    pairs=[("h_spatial","h_coefficient"),("p_spatial","p_coefficient"),
           ("p_spatial","h_spatial"),("p_spatial","wrong_instance_spatial"),
           ("p_spatial","wrong_image_spatial"),
           ("p_spatial_projected","p_coefficient"),("h_spatial_projected","h_coefficient")]
    for group in ("all","good_box_failure"):
        for left,right in pairs:
            key=f"{group}:{left}-{right}"
            output["primary_contrasts"][key]={}
            for metric in ("effect_cos","iou","auc"):
                output["primary_contrasts"][key][metric]=estimate(
                    arms[left][:,m[metric]]-arms[right][:,m[metric]],groups[group])
    # All preregistered contrast/metric/group combinations belong to one family.
    tests=[r for contrast in output["primary_contrasts"].values() for r in contrast.values()]
    order=np.argsort([r["p_bootstrap"] for r in tests])
    floor=0.
    for rank,index in enumerate(order):
        floor=max(floor,min(1.,tests[index]["p_bootstrap"]*(len(tests)-rank)))
        tests[index]["p_holm"]=floor
    write(a.out/"RESULTS.json",output)
    lines=["# 7Q：像素修正预测与系数预测的配对结果","",
        "冻结 YOLO26m-seg / Ultralytics 8.4.100；10,000 fit、200 dev、2,000 复用 val 图。候选由官方 one-to-one GT 条件分配产生；不是完整推理 AP。",
        "测试图未用于本次 probe 训练或选模，但已被前序诊断查看。表内为两个 seed 的平均；置信区间按图像配对重采样，seed 差异单独保存在 RESULTS.json。", ""]
    def fmt(r,scale=1):
        if r["mean"] is None:return "NA"
        lo,hi=r["ci95"]
        return f'{scale*r["mean"]:+.3f} [{scale*lo:+.3f}, {scale*hi:+.3f}]'
    for group in groups:
        lines += [f"## {group}","",
            "| arm | functional cosine | ΔIoU (pp), 95% CI | ΔAUC (pp) | net Mask75 (pp) | repairs / damage (seed 0,1) |",
            "|---|---:|---:|---:|---:|---|"]
        for arm,r in output["results"][group].items():
            if arm=="original":continue
            lines.append(f'| {arm} | {fmt(r["effect_cos"])} | {fmt(r["delta_iou"],100)} | {100*r["delta_auc"]["mean"]:+.3f} | {fmt(r["net_mask75"],100)} | {r["repairs_per_seed"]} / {r["damage_per_seed"]} |')
        lines.append("")
    lines += ["## 口径与限制","",
        "- IoU/coverage：原图 COCO annToMask；AUC/FPR：官方 overlap ownership 图最近邻缩至160网格、固定预测框支持。",
        "- 功能方向在原生 prototype 像素上计算；直接空间输出16×16还原到原生ROI。projected 是不读取GT的最小范数原型空间投影。",
        "- oracle_grid 与 oracle_grid_projected 是读取GT的分辨率/投影上限，不能当成可部署方法。",
        "- 阴性结果不能推出信息不存在；阳性结果不证明缺失信息唯一来自某个网络层。",
        "- 这是复用验证集的探索性诊断，确认性论文结论还需要冻结方法后的新评价。",
        ""]
    (a.out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    write(a.out/"COMPLETE.json",dict(images=len(unique),candidates=len(iid),results="RESULTS.json",report="RESULTS.md"))
    print(json.dumps(dict(images=len(unique),candidates=len(iid),complete=True)),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--evaluation",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);main(p.parse_args())

