"""Standalone diagnostic figure from locked 7Q results, no selection."""
import argparse,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def main(a):
    data=json.loads((a.summary/"RESULTS.json").read_text())
    a.out.mkdir(parents=True,exist_ok=True)
    labels=["h","h + original logit","h + true local P","h + other-instance P","h + other-image P"]
    modes=["h","z","p","wrong_instance","wrong_image"]
    styles=[("coefficient","Coefficient", "#2166AC"),("spatial","Direct spatial","#D6604D"),
            ("spatial_projected","Spatial projected to P","#4D9221")]
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout="constrained")
    for row,group in enumerate(["all","good_box_failure"]):
        for col,metric in enumerate(["effect_cos","delta_iou"]):
            ax=axes[row,col]
            scale=100 if metric=="delta_iou" else 1
            for k,(suffix,label,color) in enumerate(styles):
                records=[data["results"][group][f"{mode}_{suffix}"][metric] for mode in modes]
                y=np.array([r["mean"] for r in records])*scale
                lo=np.array([r["ci95"][0] for r in records])*scale
                hi=np.array([r["ci95"][1] for r in records])*scale
                x=np.arange(5)+(k-1)*.23
                ax.errorbar(x,y,yerr=np.stack([y-lo,hi-y]),fmt="o",capsize=2,
                    color=color,label=label,markersize=4)
            ax.axhline(0,color="#777777",lw=.7)
            ax.set_xticks(np.arange(5),labels,rotation=22,ha="right",fontsize=8)
            ax.set_ylabel("Native-pixel direction cosine" if col==0 else "Mask IoU change (percentage points)")
            ax.set_title(("All candidates" if row==0 else "Original Mask75 failure with Box IoU >= 0.75")+
                (" | direction" if col==0 else " | utility"),fontsize=10)
            ax.grid(axis="y",alpha=.2);ax.spines[["top","right"]].set_visible(False)
    axes[0,0].legend(fontsize=8,loc="best")
    fig.suptitle("7Q: Does direct pixel-logit prediction improve recoverable mask errors?",fontsize=13)
    fig.supxlabel("2 training seeds; paired 95% image-bootstrap intervals; reused held-out COCO val images. Not COCO AP.",fontsize=9)
    for ext in ("png","pdf"):fig.savefig(a.out/f"DIRECTION_UTILITY.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    (a.out/"COMPLETE.json").write_text(json.dumps({"figure":"DIRECTION_UTILITY.png","pdf":"DIRECTION_UTILITY.pdf"}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);main(p.parse_args())

