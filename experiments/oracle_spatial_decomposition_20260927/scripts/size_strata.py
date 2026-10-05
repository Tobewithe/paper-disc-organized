"""Post-hoc size audit motivated by boundary occupying much of small-object support."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch
torch.set_num_threads(4)
def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    run=dict(run_id=a.out.name,status="running",started_at=datetime.now(timezone.utc).isoformat(),
        purpose="Exploratory size stratification after primary results; not preregistered primary comparison")
    (a.out/"run.json").write_text(json.dumps(run,indent=2))
    d=np.load(a.source/"DECOMPOSITION.npz");g=np.load(a.grid/"GRID_CEILING.npz")
    coco=json.loads(Path(r"C:\Dpan\codexproject\paper-disc-organized\assets\datasets\coco\annotations\instances_val2017.json").read_text())
    aa={x["id"]:x["area"] for x in coco["annotations"]};area=np.array([aa[int(k[1])] for k in d["keys"]])
    ids,inv=np.unique(d["keys"][:,0],return_inverse=True);G=len(ids);rng=np.random.default_rng(72727)
    W=torch.from_numpy(np.stack([np.bincount(rng.integers(G,size=G),minlength=G) for _ in range(2000)]).astype(np.float64))
    ix=[list(d["names"]).index("p_spatial_s"+str(s)) for s in (0,1)]
    rm=d["regional"][:,1];hy=d["hybrid_iou"][:,1];io=d["iou640"];gm=d["geometry"][:,1]
    base=(d["original_iou"]<.75)&(d["box_iou"]>=.75)
    regimes=dict(small=area<32**2,medium=(area>=32**2)&(area<96**2),large=area>=96**2,
                 boundary_under_25pct=gm[:,0,0]/d["supportarea"]<=.25)
    vectors={"boundary_area_fraction":gm[:,0,0]/d["supportarea"],
             "predictor_gain":np.mean(io[:,np.array(ix)+2],1)-io[:,0],"oracle_gain":io[:,1]-io[:,0]}
    for r,region in enumerate(d["regions"]):
        vals=rm[:,r][:,ix,0];n=np.isfinite(vals).sum(1)
        co=np.divide(np.nansum(vals,1),n,out=np.full(len(n),np.nan),where=n>0)
        vectors[str(region)+"/cos"]=co
        vectors[str(region)+"/hybrid_gain"]=hy[:,r,ix].mean(1)-io[:,np.array(ix)+2].mean(1)
        vectors[str(region)+"/grid_only_gain"]=g["values"][:,r,4]
        vectors[str(region)+"/oracle_only_gain"]=g["values"][:,r,5]
    keys=list(vectors);x=np.stack([vectors[k] for k in keys],1);results={}
    for label,sel in regimes.items():
        sel=sel&base;v=x.copy();v[~sel]=np.nan
        sums=np.zeros((G,v.shape[1]));counts=sums.copy()
        np.add.at(sums,inv,np.nan_to_num(v));np.add.at(counts,inv,np.isfinite(v))
        den=(W@torch.from_numpy(counts)).numpy();num=(W@torch.from_numpy(sums)).numpy()
        boot=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        stats={}
        for j,key in enumerate(keys):
            stats[key]=dict(mean=float(np.nanmean(v[:,j])),ci95=np.nanquantile(boot[:,j],[.025,.975]).tolist(),n=int(np.isfinite(v[:,j]).sum()))
        results[label]=dict(instances=int(sel.sum()),stats=stats)
    (a.out/"SIZE_RESULTS.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    lines=["# 7R 目标大小补充分析（探索性）","","读取主结果后发现3像素边界带覆盖小目标很大比例，因此补此分层；不冒称预注册检验。使用原COCO annotation.area划分small<32²、medium 32²–96²、large>=96²。全部为原框好且掩码失败组，空间预测器两个种子。","",
       "| 大小 | n | 边界带面积占比 | 边界补回 ΔIoU pp | 内部 | 邻居 | 背景 |","|---|---:|---:|---:|---:|---:|---:|"]
    for k,v in results.items():
        s=v["stats"];fmt=lambda st:f'{100*st["mean"]:.3f} [{100*st["ci95"][0]:.3f}, {100*st["ci95"][1]:.3f}]'
        lines.append(f"| {k} | {v['instances']} | {100*s['boundary_area_fraction']['mean']:.1f}% | "+" | ".join(fmt(s[r+'/hybrid_gain']) for r in d["regions"])+" |")
    lines+=["","这些是640空间分解中的GT辅助收益，不是AP；四项不可相加。边界补回的全组均值高，不代表对大目标也是最重要的修复。"]
    (a.out/"SIZE_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    run.update(status="completed",finished_at=datetime.now(timezone.utc).isoformat());(a.out/"run.json").write_text(json.dumps(run,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--grid",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
