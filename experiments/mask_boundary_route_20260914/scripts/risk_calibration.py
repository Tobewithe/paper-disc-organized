"""GT-free inference adapter for the fixed train2017-fitted IoU-gain regressors.

Features match the candidate bank. CPU regressor latency must be included in
deployment measurement; this adapter does not claim zero overhead.
"""
from pathlib import Path
import numpy as np
import torch
from portable_risk import load_estimator
from mask_calibration import input_logits, export_masks, apply_threshold


def prediction_features(binary, base_export, smooth_export, mode):
    area=base_export.sum((1,2)).float()
    # Bbox of the original-grid mask, matching pycocotools.toBbox on its RLE.
    occupied_x=base_export.bool().any(1)
    occupied_y=base_export.bool().any(2)
    xs=torch.arange(base_export.shape[2],device=base_export.device)[None]
    ys=torch.arange(base_export.shape[1],device=base_export.device)[None]
    width=torch.where(occupied_x,xs,-1).amax(1)-torch.where(occupied_x,xs,base_export.shape[2]).amin(1)+1
    height=torch.where(occupied_y,ys,-1).amax(1)-torch.where(occupied_y,ys,base_export.shape[1]).amin(1)+1
    width=width.clamp_min(0);height=height.clamp_min(0)
    perimeter=((binary[:,:,1:]!=binary[:,:,:-1]).sum((1,2))+(binary[:,1:,:]!=binary[:,:-1,:]).sum((1,2))+
        binary[:,:,0].sum(1)+binary[:,:,-1].sum(1)+binary[:,0,:].sum(1)+binary[:,-1,:].sum(1)).float()
    compactness=perimeter.square()/(4*np.pi*binary.sum((1,2)).float().clamp_min(1))
    smooth_area=smooth_export.sum((1,2)).float()
    values=torch.stack((area,width.float(),height.float(),compactness,smooth_area),1).cpu().numpy().astype(np.float64)
    a,w,h,c,s=values.T
    columns=[np.log1p(a)]
    if mode in ('shape','response'):
        columns += [np.log1p(np.maximum(w,h)/np.maximum(np.minimum(w,h),1)),a/np.maximum(w*h,1),np.log1p(c)]
    if mode=='response':columns += [(a-s)/np.maximum(a,1)]
    if mode not in ('area','shape','response'):raise ValueError(mode)
    return np.column_stack(columns)


def make_risk_predictor(model_file, mode='shape', chunk=24):
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops
    estimator=load_estimator(Path(model_file))

    class RiskCalibratedPredictor(SegmentationPredictor):
        def construct_result(self,pred,img,orig_img,img_path,proto):
            if self.args.retina_masks:raise ValueError('Only the evaluated non-retina input-grid decoder is implemented')
            batches=[]
            for offset in range(0,len(pred),chunk):
                p=pred[offset:offset+chunk]
                logits=input_logits(proto,p[:,6:],p[:,:4],img.shape[2:])
                base=(logits>0).byte();base_export=export_masks(base,orig_img.shape[:2])
                area=base_export.sum((1,2)).float()
                smooth=apply_threshold(logits,area,'smooth')
                smooth_export=export_masks(smooth,orig_img.shape[:2])
                x=prediction_features(base,base_export,smooth_export,mode)
                gain=torch.as_tensor(estimator.predict(x),device=base.device)
                selected=(gain>0)&smooth_export.flatten(1).any(1).bool()
                batches.append(torch.where(selected[:,None,None],smooth,base))
            masks=torch.cat(batches) if batches else None
            pred=pred[:,:6].clone()
            if len(pred):
                pred[:,:4]=ops.scale_boxes(img.shape[2:],pred[:,:4],orig_img.shape)
                keep=masks.amax((-2,-1))>0
                pred,masks=pred[keep],masks[keep]
            return Results(orig_img,path=img_path,names=self.model.names,boxes=pred,masks=masks)

    return RiskCalibratedPredictor
