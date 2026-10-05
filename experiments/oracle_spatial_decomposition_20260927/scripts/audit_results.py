import argparse,json
from pathlib import Path
import numpy as np
def main(a):
    d=np.load(a.source/"DECOMPOSITION.npz")
    gm=d["geometry"];rm=d["regional"]
    area=np.nansum(gm[:,:,:,0],2)
    pixel_error=float(np.max(np.abs(area-d["supportarea"][:,None])))
    # Sum disjoint TP/FP repair counts and reconstruct every predictor's full IoU.
    positive=np.nansum(gm[:,:,:,1],2)
    fn=np.nansum(gm[:,:,:,2],2);fp=np.nansum(gm[:,:,:,3],2)
    repair_fn=np.nansum(rm[:,:,:,:,6],2);repair_fp=np.nansum(rm[:,:,:,:,7],2)
    damage_tp=np.nansum(rm[:,:,:,:,8],2);damage_tn=np.nansum(rm[:,:,:,:,9],2)
    tp=positive[:,:,None]-fn[:,:,None]+repair_fn-damage_tp
    errors=fp[:,:,None]-repair_fp+damage_tn
    rec=tp/np.maximum(d["gtarea"][:,None,None]+errors,1)
    iou_error=float(np.nanmax(np.abs(rec-d["iou640"][:,None,2:])))
    assert pixel_error==0,pixel_error
    assert iou_error<1e-6,iou_error
    oracle_tp=positive-fn+np.nansum(gm[:,:,:,5],2)-np.nansum(gm[:,:,:,7],2)
    oracle_fp=fp-np.nansum(gm[:,:,:,6],2)+np.nansum(gm[:,:,:,8],2)
    oi=oracle_tp/np.maximum(d["gtarea"][:,None]+oracle_fp,1)
    oracle_error=float(np.nanmax(np.abs(oi-d["iou640"][:,None,1])))
    assert oracle_error<1e-6,oracle_error
    output=dict(unique_keys=len(np.unique(d["keys"],axis=0)),instances=len(d["keys"]),
        partition_pixel_error=pixel_error,model_iou_reconstructed_from_regional_counts_error=iou_error,
        oracle_iou_reconstructed_from_regional_counts_error=oracle_error,
        original_iou_replay_error_quantiles=np.quantile(d["replay_errors"][:,2],[.5,.95,.99,1]).tolist(),
        empty_support=int((d["supportarea"]==0).sum()),
        cohort_counts=dict(failure=int((d["original_iou"]<.75).sum()),good_box_failure=int(((d["original_iou"]<.75)&(d["box_iou"]>=.75)).sum())))
    (a.source/"BOOKKEEPING_AUDIT.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);main(p.parse_args())
