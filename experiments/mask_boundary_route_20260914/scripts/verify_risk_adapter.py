"""Real-forward parity of fitted risk adapter versus the candidate-bank feature definitions."""
import argparse,json,os,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--package-root',required=True);p.add_argument('--weights',required=True)
 p.add_argument('--model',required=True);p.add_argument('--images',required=True);p.add_argument('--ids',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();sys.path.insert(0,a.package_root)
 import torch,numpy as np,joblib
 from pycocotools import mask as mu
 from ultralytics import YOLO
 from mask_calibration import input_logits,export_masks,apply_threshold
 from risk_calibration import make_risk_predictor,prediction_features
 from fit_risk_calibration import features
 torch.set_num_threads(4)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 estimator=joblib.load(a.model);parent=make_risk_predictor(a.model,'response');checks=[]
 class VerifyPredictor(parent):
  def construct_result(self,pred,img,orig_img,img_path,proto):
   expected=[];maxdiff=0.;decisions=0
   for offset in range(0,len(pred),24):
    r=pred[offset:offset+24];logits=input_logits(proto,r[:,6:],r[:,:4],img.shape[2:]);b=(logits>0).byte()
    exported=export_masks(b,orig_img.shape[:2]);area=exported.sum((1,2)).float()
    sm=apply_threshold(logits,area,'smooth');exportsm=export_masks(sm,orig_img.shape[:2])
    rows=[]
    for mask,g,smooth in zip(exported.cpu().numpy(),b.cpu().numpy(),exportsm.cpu().numpy()):
     ar=int(mask.sum());_,_,w,h=mu.toBbox(mu.encode(np.asfortranarray(mask)))
     per=np.count_nonzero(g[:,1:]!=g[:,:-1])+np.count_nonzero(g[1:,:]!=g[:-1,:])+int(g[:,0].sum()+g[:,-1].sum()+g[0,:].sum()+g[-1,:].sum())
     # Match the recorded float32 grid compactness, all other inputs are integer-derived float64.
     c=np.float32(per*per)/np.float32(4*np.pi*max(int(g.sum()),1))
     rows.append(dict(baseline_area=ar,mask_elongation=max(w,h)/max(min(w,h),1),mask_extent=ar/max(w*h,1),grid_compactness=c,
         removed_fraction=(ar-int(smooth.sum()))/max(ar,1)))
    expected_features=features(rows,'response');actual=prediction_features(b,exported,exportsm,'response')
    maxdiff=max(maxdiff,float(np.abs(actual-expected_features).max()))
    assert np.allclose(actual,expected_features,rtol=1e-6,atol=1e-6)
    use=estimator.predict(expected_features)>0
    assert np.array_equal(use,estimator.predict(actual)>0)
    selected=torch.as_tensor(use,device=b.device)&exportsm.flatten(1).any(1).bool();decisions+=int(selected.sum())
    expected.append(torch.where(selected[:,None,None],sm,b))
   result=super().construct_result(pred.clone(),img,orig_img,img_path,proto)
   if expected:
    masks=torch.cat(expected);masks=masks[masks.amax((-2,-1))>0]
    assert torch.equal(masks,result.masks.data)
    count=len(masks)
   else:count=0
   checks.append(dict(image=Path(img_path).name,candidates=len(pred),output_masks=count,calibrated=decisions,
       feature_max_abs_error=maxdiff,mask_mismatch_pixels=0))
   return result
 ids=json.loads(Path(a.ids).read_text())[:3]
 source=out/'images.txt';source.write_text('\n'.join(str(Path(a.images)/f'{i:012d}.jpg') for i in ids)+'\n')
 model=YOLO(a.weights);model.model.model[-1].end2end=True
 for _ in model.predict(source=str(source),imgsz=640,conf=.001,max_det=300,device='cpu',batch=1,half=False,stream=True,verbose=False,predictor=VerifyPredictor):pass
 result=dict(run_id=os.environ.get('RESEARCH_RUN_ID'),checks=checks,images=len(checks),device='cpu',
     claim='prediction-only feature and mask-output parity on identical real-forward tensors; no GPU/CPU parity or end-to-end latency claim')
 temp=out/'SUMMARY.json.tmp';temp.write_text(json.dumps(result,indent=2));os.replace(temp,out/'SUMMARY.json');print(json.dumps(result))

if __name__=='__main__':main()
