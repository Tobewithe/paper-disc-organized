"""Create standalone figures and representative, non-cherry-picked failure views."""
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

# Direct conda-python invocation on Windows must expose its BLAS dependencies.
# Without this, NumPy matmul called by Matplotlib's Bezier code can terminate
# with 0xc06d007f; this changes library discovery only in this plotting process.
if os.name == "nt":
    # This process only renders saved outputs; numerical evaluations were run
    # without this compatibility override in their frozen environments.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    library_bin = Path(sys.executable).parent / "Library" / "bin"
    if library_bin.is_dir():
        os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")

import numpy as np


def main():
    import faulthandler
    faulthandler.enable()
    ap=argparse.ArgumentParser()
    for key in ("root","raw","evaluation","repairs","rendering","out"):
        ap.add_argument("--"+key,type=Path,required=True)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    print("Loading plotting libraries",flush=True)
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    print("Plotting summary",flush=True)
    plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,"font.size":10})
    s=json.loads((a.evaluation/"SUMMARY.json").read_text())
    repairs=json.loads((a.repairs/"SUMMARY.json").read_text())
    rendering=json.loads((a.rendering/"SUMMARY.json").read_text())
    states=("joint_good","box_good_mask_unavailable","mask_good_box_unavailable","separate_good_no_joint","neither_good")
    labels=("同候选框/掩码都好","有好框，没有好掩码","有好掩码，没有好框","好框/好掩码分属不同候选","框/掩码均无达标候选")
    colors=("#1e8e6e","#d67b24","#467fbe","#8569af","#b64e51")
    fig,axes=plt.subplots(1,3,figsize=(17,5.6),gridspec_kw={"width_ratios":[1.35,1,1.1]})
    counts=[s["full"]["geometry_states"].get(k,0) for k in states]
    ax=axes[0];ax.barh(labels,counts,color=colors);ax.invert_yaxis()
    for i,n in enumerate(counts):ax.text(n+150,i,f"{n:,}",va="center")
    ax.set_xlim(0,max(counts)*1.22);ax.set_title("全量 raw 几何状态（IoU≥0.75）");ax.set_xlabel("GT 数；不使用类别/分数筛选")
    x=np.arange(3);groups=("small","medium","large")
    ax=axes[1]
    ax.bar(x-.18,[100*s["by_area"][g]["raw_mask75"]/s["by_area"][g]["gt"] for g in groups],.36,label="存在 Mask75 raw 候选",color="#467fbe")
    ax.bar(x+.18,[100*s["by_area"][g]["normal_mask75"]/s["by_area"][g]["gt"] for g in groups],.36,label="正常输出 COCO Mask75",color="#1e8e6e")
    ax.set_xticks(x,["小","中","大"]);ax.set_ylim(0,100);ax.set_ylabel("占对应 GT 比例（%）");ax.set_title("能力候选与实际输出");ax.legend(fontsize=8,loc="upper left")
    ax=axes[2]
    modes=("remove_fp","fill_fn_within_support","perfect_within_support","perfect_full_gt")
    vals=[repairs["results"][m]["delta_AP_points"] for m in modes]
    ax.barh(["只删误报像素","只补框内遗漏","理想框内掩码","直接使用完整 GT"],vals,color=["#467fbe","#d67b24","#1e8e6e","#b0b0b0"]);ax.invert_yaxis()
    for i,n in enumerate(vals):ax.text(n+.08,i,f"{n:+.2f}",va="center")
    ax.set_xlim(0,max(vals)*1.2);ax.set_xlabel("Δ Mask AP（点），独立修复，不可相加");ax.set_title("GT 辅助诊断：已有 Box75 输出")
    fig.suptitle("YOLO26m-seg · COCO val2017 5,000 张 · 36,335 个普通 GT",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(a.out/"capability_summary.png",dpi=170,bbox_inches="tight");fig.savefig(a.out/"capability_summary.svg",bbox_inches="tight");plt.close(fig)
    print("Rendering representative cases",flush=True)
    # Select median-quality examples, stratified by size, from the actual
    # full-raw 'good box, no good mask' state. No score-based selection.
    rows=[json.loads(line) for line in (a.evaluation/"gt_joined.jsonl").read_text().splitlines()]
    chosen=[]
    for group in groups:
        items=sorted((r for r in rows if r["geometry_state"]=="box_good_mask_unavailable" and r["area_group"]==group),key=lambda r:(r["mask_given_box75"],r["annotation_id"]))
        chosen.append(items[len(items)//2])
    sys.path.insert(0,str(a.root/"shared/vendor/ultralytics_8_4_100"))
    import cv2
    import torch
    import torch.nn.functional as F
    from ultralytics.utils import ops
    from pycocotools.coco import COCO
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO(str(a.root/"assets/datasets/coco/annotations/instances_val2017.json"))
    fig,axes=plt.subplots(3,2,figsize=(11,11))
    for i,row in enumerate(chosen):
        rid=row["best_mask_with_box75"]["raw_id"];image_id=row["image_id"]
        with np.load(a.raw/"images"/f"{image_id:012d}.npz") as z, torch.inference_mode():
            proto=torch.as_tensor(z["proto"],device="cuda");coef=torch.as_tensor(z["coefficients"],device="cuda")
            logits=(coef.T@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
            box=torch.as_tensor(z["boxes_input"][[rid]],device="cuda")
            grid=F.interpolate(logits[[rid]][None],tuple(z["input_shape"].tolist()),mode="bilinear")[0]
            mask=ops.crop_mask(grid,box).gt(0).byte()
            mask=ops.scale_masks(mask[None],tuple(z["original_shape"].tolist()))[0,0].byte().cpu().numpy().astype(bool)
            pred_box=z["boxes_original"][rid]
        gt=coco.annToMask(coco.anns[row["annotation_id"]]).astype(bool)
        ann=coco.anns[row["annotation_id"]]
        bx,by,bw,bh=ann["bbox"]
        x1=int(max(0,min(bx,pred_box[0])-max(8,bw*.15)));y1=int(max(0,min(by,pred_box[1])-max(8,bh*.15)))
        x2=int(min(gt.shape[1],max(bx+bw,pred_box[2])+max(8,bw*.15)))
        y2=int(min(gt.shape[0],max(by+bh,pred_box[3])+max(8,bh*.15)))
        image=cv2.cvtColor(cv2.imread(str(a.root/"assets/datasets/coco/images/val2017"/coco.imgs[image_id]["file_name"])),cv2.COLOR_BGR2RGB)
        overlay=image.copy()
        cv2.drawContours(overlay,cv2.findContours(gt.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(45,230,100),1)
        cv2.rectangle(overlay,tuple(pred_box[:2].astype(int)),tuple(pred_box[2:].astype(int)),(255,190,25),1)
        error=np.full_like(image,245);error[gt&mask]=[72,165,114];error[mask&~gt]=[220,85,77];error[gt&~mask]=[65,125,200]
        axes[i,0].imshow(overlay[y1:y2,x1:x2],interpolation="nearest");axes[i,1].imshow(error[y1:y2,x1:x2],interpolation="nearest")
        w=row["best_mask_with_box75"]
        axes[i,0].set_title(f"{('小','中','大')[i]}目标中位案例 · GT {row['annotation_id']} · raw {rid}\n绿线 GT；黄框为该候选实际框")
        axes[i,1].set_title(f"Box IoU {w['box_iou']:.3f} / Mask IoU {w['mask_iou']:.3f}\n绿 TP；红 FP；蓝 FN")
        for ax in axes[i]:ax.axis("off")
    fig.suptitle("已有 Box75 候选中，GT 选择的最佳 mask 仍然失败\n代表图用于解释；统计以全部 5,000 张为准",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(a.out/"raw_failure_examples.png",dpi=150,bbox_inches="tight");plt.close(fig)
    (a.out/"selected_cases.json").write_text(json.dumps(chosen,ensure_ascii=False),encoding="utf-8")
    (a.out/"COMPLETE.json").write_text(json.dumps(dict(figures=3,examples=3)),encoding="utf-8")


if __name__=="__main__":main()
