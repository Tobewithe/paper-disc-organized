import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch
torch.set_num_threads(4)
def main(a):
    main=np.load(a.source/"DECOMPOSITION.npz");grid=np.load(a.grid/"GRID_CEILING.npz")
    assert np.array_equal(main["keys"],grid["keys"])
    ids,inv=np.unique(main["keys"][:,0],return_inverse=True);G=len(ids);rng=np.random.default_rng(72727)
    w=np.stack([np.bincount(rng.integers(G,size=G),minlength=G) for _ in range(2000)]).astype(np.float64)
    groups=dict(all=np.ones(len(inv),bool),failure=main["original_iou"]<.75,
        good_box_failure=(main["original_iou"]<.75)&(main["box_iou"]>=.75),success=main["original_iou"]>=.75)
    result={}
    for name,keep in groups.items():
        arr=grid["values"].reshape(len(inv),-1).copy();arr[~keep]=np.nan
        sums=np.zeros((G,arr.shape[1]));counts=sums.copy()
        np.add.at(sums,inv,np.nan_to_num(arr));np.add.at(counts,inv,np.isfinite(arr))
        bsum=(torch.from_numpy(w)@torch.from_numpy(sums)).numpy()
        bcount=(torch.from_numpy(w)@torch.from_numpy(counts)).numpy()
        boot=np.divide(bsum,bcount,out=np.full_like(bsum,np.nan),where=bcount>0).reshape(2000,4,7)
        estimates=np.nanmean(arr,axis=0).reshape(4,7)
        d={}
        for ri,region in enumerate(grid["regions"]):
            d[str(region)]={}
            for mi,metric in enumerate(grid["metrics"]):
                d[str(region)][str(metric)]=dict(mean=float(estimates[ri,mi]),ci95=np.nanquantile(boot[:,ri,mi],[.025,.975]).tolist())
            ratios=boot[:,ri,4]/boot[:,ri,5]
            d[str(region)]["retained_regional_gain_ratio"]=dict(mean=float(estimates[ri,4]/estimates[ri,5]),ci95=np.nanquantile(ratios,[.025,.975]).tolist())
        result[name]=d
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"run.json").write_text(json.dumps(dict(run_id=a.out.name,status="completed",recorded_at=datetime.now(timezone.utc).isoformat(),source=str(a.grid)),indent=2),encoding="utf-8")
    (a.out/"GRID_RESULTS.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    lines=["# 7R 输出网格上界对照","","下表为框好且原Mask75失败组；r=3；640栅格原始GT；grid为同一16x16表示。","",
        "| 区域 | grid与oracle cos | 仅该区完整oracle ΔIoU pp | 仅该区grid-oracle ΔIoU pp | 区域收益保留率 |","|---|---:|---:|---:|---:|"]
    for region,v in result["good_box_failure"].items():
        lines.append(f"| {region} | {v['cos']['mean']:.4f} | {100*v['oracle_only_gain']['mean']:.3f} | {100*v['grid_only_gain']['mean']:.3f} | {100*v['retained_regional_gain_ratio']['mean']:.1f}% |")
    lines+=["","该比例是区域mean gain的比值，不是各实例比值的平均。所有区间见GRID_RESULTS.json。该网格oracle使用GT，只是表示上界，不是可部署结果。"]
    (a.out/"GRID_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--grid",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
