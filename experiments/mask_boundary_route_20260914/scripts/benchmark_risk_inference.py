"""Paired batch-one pipeline timing on preloaded images, CUDA-synchronized."""
import argparse, json, os, sys, time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for name in ('package-root','weights','model','images','ids','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();sys.path.insert(0,a.package_root)
    import cv2,numpy as np,torch,ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.engine.results import Results
    from ultralytics.utils import ops
    from mask_calibration import input_logits,export_masks,apply_threshold
    from risk_calibration import make_risk_predictor
    assert ultralytics.__version__=='8.4.100'
    torch.set_num_threads(4)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    def save(name,value):
        temp=out/(name+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temp,out/name)
    ids=json.loads(Path(a.ids).read_text());ids=[ids[i] for i in np.linspace(0,len(ids)-1,80,dtype=int)]
    images=[cv2.imread(str(Path(a.images)/f'{i:012d}.jpg')) for i in ids]
    assert all(im is not None for im in images)
    def chunked(smooth):
        class Chunked(SegmentationPredictor):
            def construct_result(self,pred,img,orig_img,img_path,proto):
                masks=[]
                for start in range(0,len(pred),24):
                    r=pred[start:start+24];logits=input_logits(proto,r[:,6:],r[:,:4],img.shape[2:]);base=(logits>0).byte()
                    if smooth:
                        area=export_masks(base,orig_img.shape[:2]).sum((1,2)).float()
                        base=apply_threshold(logits,area,'smooth')
                    masks.append(base)
                mask=torch.cat(masks) if masks else None
                pred=pred[:,:6].clone()
                if len(pred):
                    pred[:,:4]=ops.scale_boxes(img.shape[2:],pred[:,:4],orig_img.shape)
                    keep=mask.amax((-2,-1))>0;pred,mask=pred[keep],mask[keep]
                return Results(orig_img,path=img_path,names=self.model.names,boxes=pred,masks=mask)
        return Chunked
    factories={'official':SegmentationPredictor,'chunked_zero':chunked(False),'fixed_smooth':chunked(True),
               'risk_response':make_risk_predictor(a.model,'response',24)}
    model=YOLO(a.weights);model.model.model[-1].end2end=True
    rows=[];modes=list(factories)
    opts=dict(imgsz=640,conf=.001,max_det=300,device=0,batch=1,half=False,verbose=False,save=False,retina_masks=False)
    for repeat in range(4):
        for mode in modes[repeat:]+modes[:repeat]:
            model.predictor=None
            for im in images[:8]:model.predict(im,predictor=factories[mode],**opts)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            for iid,im in zip(ids,images):
                torch.cuda.synchronize();start=time.perf_counter()
                result=model.predict(im,**opts)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                rows.append(dict(repeat=repeat,mode=mode,image_id=iid,ms=elapsed*1000,candidates=len(result[0].boxes)))
                del result
            save('progress.json',dict(repeat=repeat,mode=mode,measured_calls=len(rows),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
            print(json.dumps(dict(repeat=repeat,mode=mode,mean_ms=float(np.mean([r['ms'] for r in rows if r['repeat']==repeat and r['mode']==mode])))),flush=True)
    stats={}
    for mode in modes:
        ms=np.array([r['ms'] for r in rows if r['mode']==mode])
        stats[mode]=dict(mean_ms=float(ms.mean()),median_ms=float(np.median(ms)),p95_ms=float(np.quantile(ms,.95)),
            fps=1000/float(ms.mean()),calls=len(ms),block_means_ms=[float(np.mean([r['ms'] for r in rows if r['mode']==mode and r['repeat']==j])) for j in range(4)])
    ref=stats['official']['mean_ms']
    for st in stats.values():st['extra_ms_vs_official']=st['mean_ms']-ref;st['slowdown_percent']=100*(st['mean_ms']/ref-1)
    save('timings.json',rows)
    save('SUMMARY.json',dict(stats=stats,images=ids,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        ultralytics=ultralytics.__version__,weights=a.weights,model=a.model,scope='80 evenly spaced existing4500 images;4 cyclic mode orders;8 warmup calls per block;batch1,FP32,conf.001,max_det300',
        included='preprocessing, forward, postprocessing, risk feature extraction, host transfer and portable forest prediction; synchronous model.predict call',
        excluded='disk image decoding, model construction, predictor setup/warmup, display/save, RLE export and COCO scoring',
        limitations=['single notebook GPU and operating condition','fixed_smooth removes empty outputs; risk restores them','chunked_zero isolates decoder chunking cost','no throughput/batched inference claim']))


if __name__=='__main__':main()
