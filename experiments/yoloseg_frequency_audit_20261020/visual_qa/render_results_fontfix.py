"""Publication-style static figures from the laptop audit's real outputs."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, json, csv, io, contextlib, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT / "_site"))
from pycocotools.coco import COCO
from pycocotools import mask as masks
sys.path.insert(0, str(STUDY / "scripts"))
from spectral import split_bands

READ = lambda p: json.loads(Path(p).read_text(encoding="utf-8-sig"))
SOURCE = ROOT / "experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170"
COLORS = ["#31688e", "#35b779", "#e9a23b"]
LABELS = ["低频  q<8", "中频  8≤q<32", "高频  q≥32"]


def main():
    run = STUDY / "runs" / (sys.argv[1] if len(sys.argv) > 1 else READ(STUDY/"LATEST_RUN.json")["run_id"])
    assert READ(run/"run.json")["execution_status"] == "completed"
    out = run / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 11,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
                         "svg.fonttype": "path", "savefig.facecolor": "#ffffff"})
    summary = READ(run / "SUMMARY.json")
    groups = summary["groups"]
    def save(fig, name):
        fig.savefig(out/(name+".png"), dpi=175, bbox_inches="tight")
        fig.savefig(out/(name+".svg"), bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    fig.suptitle(f"YOLO26m-seg · COCO 原型与分割误差频谱\n{summary['images']:,} 张图 / {summary['paired_instances']:,} 条固定 Box75 配对 / 零训练", fontsize=19)
    values = [summary["prototypes"]["proto"]["image_macro_fraction"]] + [
        groups["all_all"][k]["image_macro_fraction"] for k in ["logit", "gt", "pred"]]
    names = ["32通道原型 P", "线性读出 z0", "真实掩码 GT", "原生预测掩码"]
    left = np.zeros(4)
    for b in range(3):
        v = np.array(values)[:, b]*100
        ax[0,0].barh(names, v, left=left, color=COLORS[b], label=LABELS[b], height=.6)
        for j, (a, vv) in enumerate(zip(left,v)):
            if vv>6:
                ax[0,0].text(a+vv/2,j,f"{vv:.1f}%",ha="center",va="center",fontsize=10,color="white" if b==0 else "#172b33")
        left += v
    ax[0,0].invert_yaxis()
    ax[0,0].set(xlim=(0,100),xlabel="去均值能量占比（%，图等权）",title="A  模型信号与掩码的频段分布")
    ax[0,0].legend(loc="lower center",bbox_to_anchor=(.5,-.33),ncol=3,frameon=False,fontsize=9)
    rad = READ(run/"RADIAL.json")
    x = (np.array(rad["bin_edges"][:-1])+np.array(rad["bin_edges"][1:]))/2
    for key, label, color in [("proto","原型 P","#4969ac"),("logit","线性读出","#8952a1"),
                            ("gt","GT","#35a47d"),("pred","预测","#333333"),("error","误差","#d76445")]:
        ax[0,1].plot(x,np.array(rad["curves"][key])[:48]*100,label=label,color=color,lw=1.8)
    for v in [8,32]:
        ax[0,1].axvline(v,color="#777777",ls="--",lw=.8)
    ax[0,1].set(xlim=(0,160),yscale="log",ylim=(.001,100),xlabel="q：每640输入像素的周期数",
                ylabel="每个等宽环带的能量份额（%，对数轴）",title="B  径向频谱（尾部 q>160 另存数据）")
    ax[0,1].legend(frameon=False,ncol=3,fontsize=9)
    xs=np.arange(3)
    for off,(group,label,color) in zip([-.2,.2],[("failure_all","自身 Mask75 失败","#d76445"),
                                               ("success_all","自身 Mask75 成功","#4969ac")]):
        v=np.array(groups[group]["error"]["image_macro_fraction"])*100
        ci=np.array(groups[group]["error"]["ci95"])*100
        ax[1,0].bar(xs+off,v,width=.36,color=color,label=label,yerr=[v-ci[:,0],ci[:,1]-v],capsize=3)
        for i,a in enumerate(v):
            ax[1,0].text(i+off,ci[i,1]+1,f"{a:.1f}",ha="center",fontsize=10)
    ax[1,0].set(xticks=xs,xticklabels=LABELS,ylabel="误差去DC能量占比（%）",title="C  错误主要落在哪个频段？")
    ax[1,0].legend(frameon=False,fontsize=10)
    for j,(group,label) in enumerate([("failure_all","失败组"),("success_all","成功组")]):
        ratio=groups[group]["pooled_pred_gt_energy_ratio"]
        ax[1,1].plot(xs,ratio,marker="o" if j==0 else "s",lw=2,label=label,color=["#d76445","#4969ac"][j])
    ax[1,1].axhline(1,color="#777777",ls="--")
    ax[1,1].set(xticks=xs,xticklabels=LABELS,ylabel="Σ预测频段能量 / ΣGT频段能量",title="D  频段响应幅度（不是频域准确率）")
    ax[1,1].legend(frameon=False)
    fig.supxlabel("固定原生输出；P/z 为有效图像区域，GT/预测为原图。频率单位相同，采样上限不同；能量低不等于不可表达。\n误差条为2000次图簇bootstrap 95%区间；未比较新方法收益。",fontsize=10)
    save(fig,"01_frequency_overview")

    fig,axes=plt.subplots(1,3,figsize=(16,5.7),layout="constrained")
    fig.suptitle("分割失败的频段与目标尺度",fontsize=18)
    for a,size in zip(axes,["S","M","L"]):
        g=groups["failure_"+size]
        for off,key,label in [(-.19,"error","无窗（主）"),(.19,"error_hann","Hann（敏感性）")]:
            v=np.array(g[key]["image_macro_fraction"])*100
            ci=np.array(g[key]["ci95"])*100
            a.bar(xs+off,v,.36,color=COLORS,alpha=1 if off<0 else .42,
                  hatch=None if off<0 else "//",label=label,yerr=[v-ci[:,0],ci[:,1]-v],capsize=2)
        a.set(xticks=xs,xticklabels=["低","中","高"],ylim=(0,100),
              title=f"{size}  |  {g['pairs']:,} 实例 / {g['images']:,} 图",ylabel="误差去DC能量占比（%）",xlabel="频段")
        a.legend(frameon=False,fontsize=10)
    fig.supxlabel("S：面积<32²；M：32²≤面积<96²；L：面积≥96²（COCO原始GT面积）。各组先图内均值，再图等权。",fontsize=10)
    save(fig,"02_error_by_scale")

    chan=np.array(READ(run/"PROTOTYPE_CHANNELS.json")["fractions"])*100
    fig,axes=plt.subplots(1,2,figsize=(12,11),layout="constrained",gridspec_kw={"width_ratios":[1,1.8]})
    fig.suptitle("32个原型通道：频段分布及加窗敏感性",fontsize=18)
    im=axes[0].imshow(chan,cmap="YlGnBu",vmin=0,vmax=100,aspect="auto")
    axes[0].set(xticks=[0,1,2],xticklabels=["低","中","高"],yticks=np.arange(32),yticklabels=[f"P{i:02d}" for i in range(32)])
    for y in range(32):
        for x0 in range(3):
            axes[0].text(x0,y,f"{chan[y,x0]:.1f}",ha="center",va="center",fontsize=8,color="white" if chan[y,x0]>55 else "#172b33")
    fig.colorbar(im,ax=axes[0],label="通道内去DC能量占比（%）",shrink=.6)
    for i,(key,label) in enumerate([("proto","无窗 · 通道等权"),("proto_hann","Hann · 通道等权"),
                                   ("proto_energy_weighted","无窗 · 通道能量加权")]):
        v=np.array(summary["prototypes"][key]["image_macro_fraction"])*100
        axes[1].plot(xs,v,marker=["o","s","^"][i],lw=2,label=label)
    axes[1].set(xticks=xs,xticklabels=LABELS,ylim=(0,100),ylabel="原型频段能量占比（%）",
                title="原型边缘和通道幅度是否影响结论？")
    axes[1].legend(frameon=False)
    fig.supxlabel("原型取自去除letterbox后的同次推理缓存。频谱依赖原型基的参数化；低能量不能证明不可表达。模型参数与原生输出匹配均保持。",fontsize=10)
    save(fig,"03_prototype_channels")

    selected=READ(run/"EXAMPLES.json")
    needed={r["prediction_index"] for r in selected}
    allpred=READ(SOURCE/"trace/native/predictions.json")
    pred={i:r for i,r in enumerate(allpred) if i in needed}
    del allpred
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO(str(ROOT/"assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json"))
    trace={r["image_id"]:r for r in READ(SOURCE/"trace/TRACE_MANIFEST.json")["images"]}
    imroot=Path(READ(STUDY/"FREEZE_MANIFEST.json")["image_root"])
    for status,slug in [(0,"failure"),(1,"success")]:
        chosen=[r for r in selected if r["own_mask75"]==status]
        fig,axes=plt.subplots(len(chosen),7,figsize=(21,3.5*len(chosen)+1),squeeze=False,layout="constrained")
        fig.suptitle(f"按尺度×{'失败' if status==0 else '成功'}组选取IoU中位实例 · 无人工挑图",fontsize=18)
        for row,r in zip(axes,chosen):
            g=coco.annToMask(coco.anns[r["annotation_id"]]).astype(np.float64)
            m=masks.decode(pred[r["prediction_index"]]["segmentation"]).astype(np.float64)
            e=m-g
            imp=imroot/f"{r['image_id']:012d}.jpg"
            assert hashlib.sha256(imp.read_bytes()).hexdigest()==trace[r["image_id"]]["image_sha256"]
            rgb=np.array(Image.open(imp).convert("RGB"))
            parts,dc=split_bands(e,tuple(trace[r["image_id"]]["ratio_pad"][0]))
            visual_error=np.ones((*e.shape,3))*.96
            visual_error[e>0]=[.9,.3,.18]
            visual_error[e<0]=[.2,.45,.85]
            # Display ROI includes both full GT and full prediction, with fixed 10% margin.
            ys,xv=np.where((g+m)>0)
            pad=max(10,int(.1*max(np.ptp(xv)+1,np.ptp(ys)+1)))
            sl=(slice(max(0,int(ys.min())-pad),min(g.shape[0],int(ys.max())+pad+1)),
                slice(max(0,int(xv.min())-pad),min(g.shape[1],int(xv.max())+pad+1)))
            row[0].imshow(rgb[sl]); row[1].imshow(g[sl],cmap="gray",vmin=0,vmax=1)
            row[2].imshow(m[sl],cmap="gray",vmin=0,vmax=1); row[3].imshow(visual_error[sl])
            for j,value in enumerate([parts[0]+dc,parts[1],parts[2]],4):
                row[j].imshow(value[sl],cmap="RdBu_r",vmin=-1,vmax=1)
            titles=["原图局部","GT","预测","红=多分 / 蓝=漏分","低频 + DC","中频","高频"]
            for a,title in zip(row,titles):
                a.set_title(title,fontsize=11)
                a.set_xticks([]);a.set_yticks([])
                for spine in a.spines.values():spine.set_visible(False)
            row[0].set_ylabel(f"{r['size_group']} · {r['category']}\nIoU={r['iou']:.3f}\n图{r['image_id']} / GT{r['annotation_id']}",fontsize=10)
        fig.supxlabel("频谱与重建在完整原图上计算，展示时才裁局部；三频重建共用 −1..1 色标，低+中+高精确还原有符号误差。\n色彩较浅表示幅度较小；频域重建可能振铃，不能视作可部署修复。",fontsize=11)
        save(fig,"04_examples_"+slug)
    writeout=dict(status="completed",run_id=run.name,figures=[p.name for p in sorted(out.iterdir())],
                  renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run/"VISUALIZATION_RECEIPT.json").write_text(json.dumps(writeout,indent=2),encoding="utf-8")
    print(json.dumps(writeout),flush=True)


if __name__=="__main__":
    main()
