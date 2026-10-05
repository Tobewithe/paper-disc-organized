"""Aggregate 7R scientific figure from paired image-cluster summaries."""
import os
os.environ["MKL_THREADING_LAYER"]="SEQUENTIAL"
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
REGIONS=("boundary","interior","neighbor","background")
LABELS=("Boundary band","Target interior","Other instances","Unannotated background")
def main(a):
    data=json.loads((a.stats/"RESULTS.json").read_text())
    s=data["cohorts"]["good_box_failure"]["stats"]
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    x=np.arange(4);colors=("#286DA8","#D87638")
    for j,(mode,label) in enumerate((("p_coefficient","Coefficient predictor"),("p_spatial","Spatial predictor"))):
        cos=[s[f"r3/{r}/{mode}/cos"] for r in REGIONS]
        amp=[s[f"r3/{r}/{mode}/amplitude"] for r in REGIONS]
        gains=[s[f"r3/{r}/{mode}/hybrid_gain"] for r in REGIONS]
        for ax,values,scale in ((axs[0,1],cos,1),(axs[1,0],amp,1),(axs[1,1],gains,100)):
            means=np.array([v["mean"] for v in values])*scale
            ci=np.array([v["ci95"] for v in values]).T*scale
            ax.bar(x+(j-.5)*.34,means,width=.32,label=label,color=colors[j],
                   yerr=np.maximum(np.stack((means-ci[0],ci[1]-means)),0),capsize=3)
    energies=[s[f"r3/{r}/energy_share"] for r in REGIONS]
    areas=[s[f"r3/{r}/area_share"] for r in REGIONS]
    for j,(v,label,col) in enumerate(((energies,"Oracle correction energy","#825BA8"),(areas,"Pixel area","#8C969D"))):
        axs[0,0].bar(x+(j-.5)*.34,[100*z["mean"] for z in v],width=.32,color=col,label=label)
    titles=("A. Where oracle correction is concentrated","B. Regional direction similarity",
            "C. Predicted amplitude along the oracle","D. Gain after replacing one region with oracle")
    ylabs=("Mean share (%)","Cosine","Dot(prediction, oracle) / squared norm(oracle)","IoU improvement (percentage points)")
    for ax,title,ylabel in zip(axs.flat,titles,ylabs):
        ax.set_title(title,fontsize=11,loc="left");ax.set_ylabel(ylabel,fontsize=9)
        ax.set_xticks(x,LABELS,rotation=15,ha="right",fontsize=8)
        ax.axhline(0,color="black",linewidth=.7);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
    fig.suptitle("7R: fixed good-box mask failures (Box IoU >= 0.75, original Mask IoU < 0.75)\n"
                 "Boundary radius 3 at 640; two fitted seeds; paired image-cluster 95% intervals",fontsize=12)
    a.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out/"SPATIAL_DECOMPOSITION.png",dpi=180);fig.savefig(a.out/"SPATIAL_DECOMPOSITION.pdf")
    plt.close(fig)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stats",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())

