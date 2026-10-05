"""Paired image-cluster statistics for 7R; no model training/selection."""
import argparse,json,uuid
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
torch.set_num_threads(6)
ROOT=Path(__file__).resolve().parents[1]
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=True),encoding="utf-8")
def cleanmean(x,axis=0):
    count=np.isfinite(x).sum(axis);total=np.nansum(x,axis)
    return np.divide(total,count,out=np.full(np.shape(total),np.nan),where=count>0)
def main(a):
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    write(out/"run.json",dict(run_id=out.name,status="running",started_at=datetime.now(timezone.utc).isoformat(),source=str(a.source)))
    d=np.load(a.source/"DECOMPOSITION.npz")
    regions=d["regions"].tolist();names=d["names"].tolist();rmet=d["regional_metrics"].tolist();gmet=d["geometry_metrics"].tolist()
    N=len(d["keys"]);rm=d["regional"];gm=d["geometry"];hy=d["hybrid_iou"];io=d["iou640"]
    original=d["original_iou"];box=d["box_iou"];groupnames=("all","failure","good_box_failure","success")
    groups=[np.ones(N,bool),original<.75,(original<.75)&(box>=.75),original>=.75]
    modes=("h_coefficient","p_coefficient","p_spatial","wrong_instance_spatial","wrong_image_spatial")
    modeids={m:[names.index(m+"_s0"),names.index(m+"_s1")] for m in modes}
    images,inverse=np.unique(d["keys"][:,0],return_inverse=True);G=len(images)
    rng=np.random.default_rng(72727);W=np.zeros((2000,G),np.float32)
    for b in range(len(W)):W[b]=np.bincount(rng.integers(G,size=G),minlength=G)
    results={}
    for group,sel in zip(groupnames,groups):
        vectors={};meta={}
        def add(name,x):
            vectors[name]=np.asarray(x,dtype=np.float64);assert vectors[name].shape==(N,),name
        add("baseline_iou640",io[:,0]);add("oracle_gain",io[:,1]-io[:,0])
        for mode,ix in modeids.items():
            add(mode+"/gain",cleanmean(io[:,np.array(ix)+2],1)-io[:,0])
            add(mode+"/mask75_net",cleanmean((io[:,np.array(ix)+2]>=.75).astype(float),1)-(io[:,0]>=.75))
        for ri,radius in enumerate(d["radii"]):
            energy=np.nansum(gm[:,ri,:,gmet.index("target_energy")],1)
            for r,region in enumerate(regions):
                prefix=f"r{radius}/{region}"
                geometry=gm[:,ri,r]
                pix=geometry[:,gmet.index("pixels")]
                e=geometry[:,gmet.index("target_energy")]
                add(prefix+"/area_share",np.nan_to_num(pix)/np.maximum(d["supportarea"],1))
                add(prefix+"/energy_share",np.nan_to_num(e)/np.maximum(energy,1e-12))
                add(prefix+"/oracle_only_gain",d["oracle_only_iou"][:,ri,r]-io[:,0])
                add(prefix+"/oracle_leaveout_loss",io[:,1]-d["oracle_leaveout_iou"][:,ri,r])
                for metric in ("oracle_repair_fn","oracle_repair_fp","oracle_damage_tp","oracle_damage_tn"):
                    add(prefix+"/"+metric+"_per_gt",np.nan_to_num(geometry[:,gmet.index(metric)])/np.maximum(d["gtarea"],1))
                for mode,ix in modeids.items():
                    hybrid_gain=cleanmean(hy[:,ri,r][:,ix],1)-cleanmean(io[:,np.array(ix)+2],1)
                    add(prefix+"/"+mode+"/hybrid_gain",hybrid_gain)
                    if radius==3:
                        add(prefix+"/"+mode+"/hybrid_gain_at_least_1pp",(hybrid_gain>=.01).astype(float))
                        add(prefix+"/"+mode+"/hybrid_damage_at_least_1pp",(hybrid_gain<=-.01).astype(float))
                    for metric in ("cos","amplitude","sign_agree_energy","mean_delta_positive","mean_delta_negative"):
                        add(prefix+"/"+mode+"/"+metric,cleanmean(rm[:,ri,r][:,ix,rmet.index(metric)],1))
                    for metric in ("repair_fn","repair_fp","damage_tp","damage_tn"):
                        add(prefix+"/"+mode+"/"+metric+"_per_gt",
                            cleanmean(np.nan_to_num(rm[:,ri,r][:,ix,rmet.index(metric)]),1)/np.maximum(d["gtarea"],1))
                    # Each seed's counterfactual effect remains visible.
                    if radius==3:
                        for seed,k in enumerate(ix):
                            add(prefix+"/"+mode+f"/hybrid_gain_s{seed}",hy[:,ri,r,k]-io[:,k+2])
                meta[prefix]=dict(nonempty_instances=int((sel&(np.nan_to_num(pix)>0)).sum()),
                                  nonzero_target_instances=int((sel&(np.nan_to_num(e)>1e-10)).sum()))
        # Compare regions on the SAME candidate subset, avoiding empty-region composition effects.
        for mode,ix in modeids.items():
            rc=[cleanmean(rm[:,1,r][:,ix,rmet.index("cos")],1) for r in range(4)]
            for ra,rb in ((0,1),(0,3),(2,3)):
                add("paired_cos/"+mode+"/"+regions[ra]+"_minus_"+regions[rb],rc[ra]-rc[rb])
        for k,region in enumerate(d["subregions"]):
            prefix="sub/"+region
            add(prefix+"/pixels",d["subgeometry"][:,k,gmet.index("pixels")])
            for mode,ix in modeids.items():
                for metric in ("cos","amplitude","mean_delta_positive","mean_delta_negative"):
                    add(prefix+"/"+mode+"/"+metric,cleanmean(d["subregional"][:,k][:,ix,rmet.index(metric)],1))
        vkeys=list(vectors);x=np.stack([vectors[k] for k in vkeys],1)
        x[~sel]=np.nan
        sums=np.zeros((G,x.shape[1]),np.float64);counts=np.zeros_like(sums)
        np.add.at(sums,inverse,np.nan_to_num(x,nan=0,posinf=0,neginf=0))
        np.add.at(counts,inverse,np.isfinite(x).astype(float))
        wb=torch.from_numpy(W).double()
        numerator=(wb@torch.from_numpy(sums)).numpy();denominator=(wb@torch.from_numpy(counts)).numpy()
        boot=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
        stats={}
        for j,key in enumerate(vkeys):
            means=boot[:,j];means=means[np.isfinite(means)]
            est=float(np.nansum(x[:,j])/max(np.isfinite(x[:,j]).sum(),1)) if means.size else float("nan")
            ci=np.quantile(means,[.025,.975]).tolist() if means.size else [float("nan")]*2
            p=min(1.,2*min((np.count_nonzero(means<=0)+1)/(len(means)+1),
                          (np.count_nonzero(means>=0)+1)/(len(means)+1))) if means.size else float("nan")
            stats[key]=dict(mean=est,ci95=ci,bootstrap_p=p,n_valid=int(np.isfinite(x[:,j]).sum()))
        results[group]=dict(instances=int(sel.sum()),images=int(len(np.unique(d["keys"][sel,0]))),stats=stats,regions=meta)
        print(group,int(sel.sum()),flush=True)
    primary=results["good_box_failure"]["stats"]
    pk=[f"r3/{r}/p_spatial/hybrid_gain" for r in regions]
    ps=np.array([primary[k]["bootstrap_p"] for k in pk]);order=np.argsort(ps)
    adjusted=np.maximum.accumulate((4-np.arange(4))*ps[order]).clip(0,1)
    for i,p in zip(order,adjusted):primary[pk[i]]["holm_p"]=float(p)
    # Conditional seed sample of two; no claim to broad training-seed variance.
    complete=json.loads((a.source/"COMPLETE.json").read_text())
    write(out/"RESULTS.json",dict(protocol="7R r=3 primary; r=1,5 descriptive; 2000 paired image cluster bootstrap",
        source=str(a.source),metrics_scope="original COCO GT mapped to640, fixed predicted support; not AP",
        cohorts=results,run_summary=complete))
    lines=["# 7R 空间分解结果","",f"复用7Q：{N:,}个候选、{G:,}张有效图、两个已训练种子；没有重新训练。",
           "区域定义、原始标注和互斥规则见 ../../PROTOCOL.md。区间为2,000次逐图配对bootstrap；两个已拟合种子先算指标再平均。",
           "本表IoU为640栅格的空间诊断，不是COCO AP；原图IoU复核见最后部分。",""]
    def fmt(st,scale=1):
        return f'{st["mean"]*scale:.3f} [{st["ci95"][0]*scale:.3f}, {st["ci95"][1]*scale:.3f}]'
    labels=dict(all="全部候选",failure="原Mask75失败",good_box_failure="框好且Mask75失败",success="原Mask75成功")
    zh=dict(boundary="边界带",interior="目标内部",neighbor="邻居远离边界部分",background="未标注背景")
    for group in groupnames:
        result=results[group];st=result["stats"]
        lines += [f"## {labels[group]}（n={result['instances']:,}）","",
                  "| 区域 | oracle能量占比 % | p-spatial方向cos | 幅度投影系数 | 仅该区oracle ΔIoU pp | 给p-spatial补该区oracle ΔIoU pp |",
                  "|---|---:|---:|---:|---:|---:|"]
        for region in regions:
            p=f"r3/{region}"
            lines.append(f"| {zh[region]} | {fmt(st[p+'/energy_share'],100)} | {fmt(st[p+'/p_spatial/cos'])} | {fmt(st[p+'/p_spatial/amplitude'])} | {fmt(st[p+'/oracle_only_gain'],100)} | {fmt(st[p+'/p_spatial/hybrid_gain'],100)} |")
        lines += ["",f"完整oracle的ΔIoU：{fmt(st['oracle_gain'],100)} pp；p-spatial：{fmt(st['p_spatial/gain'],100)} pp；p-coefficient：{fmt(st['p_coefficient/gain'],100)} pp。","",
                  "| 区域 | h-coef cos | p-coef cos | p-spatial cos | wrong-instance cos | wrong-image cos |",
                  "|---|---:|---:|---:|---:|---:|"]
        for region in regions:
            p=f"r3/{region}/"
            lines.append("| "+zh[region]+" | "+" | ".join(f'{st[p+m+"/cos"]["mean"]:.3f}' for m in modes)+" |")
        lines += [""]
    st=results["good_box_failure"]["stats"]
    lines += ["## 边界宽度敏感性：框好且失败，给p-spatial补回区域oracle","",
              "| 半径(640像素) | 边界 ΔIoU pp | 内部 | 邻居 | 背景 |","|---|---:|---:|---:|---:|"]
    for radius in d["radii"]:
        lines.append(f"| {radius} | "+" | ".join(fmt(st[f"r{radius}/{r}/p_spatial/hybrid_gain"],100) for r in regions)+" |")
    lines += ["","## 原图复核及限制","",
              f"- 重建原型ROI最大绝对误差：{complete['max_p_error']:.8g}；系数：{complete['max_c_error']:.8g}。",
              f"- 原图六组IoU相对7Q最大差：{complete['original_resolution_iou_max_error']:.8g}；超过0.001的候选数：{complete['original_resolution_iou_error_above_1e_3']}。",
              "- 四区域oracle替换使用GT，是诊断；区域IoU增益不相加，不能称因果贡献份额。",
              "- boundary优先，外边界上的邻居像素计入boundary；单独子区域指标已保存在RESULTS.json。",
              "- 同一测试集用于探索性后续分析，不能当作新的独立确认集；这里只条件于已有两个种子。",
              "- 能量或方向不足不能单独证明缺少高频信息、邻居信息或特定网络模块导致失败。",""]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    write(out/"COMPLETE.json",dict(status="completed",cohorts=list(results),instances=N,images=G))
    record=json.loads((out/"run.json").read_text());write(out/"run.json",dict(record,status="completed",finished_at=datetime.now(timezone.utc).isoformat()))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);main(p.parse_args())

