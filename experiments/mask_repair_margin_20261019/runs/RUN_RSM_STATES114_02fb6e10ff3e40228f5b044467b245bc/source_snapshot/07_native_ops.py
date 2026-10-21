"""Full original GEMM shape, then one-row native resize/crop/byte decoding."""
import numpy as np
import torch
from ultralytics.utils import ops
from native_geometry import roi_for
from native_output_bridge import encode_rle, decode_native


class Decoder:
    def __init__(self,cache,rank):
        self.cache=cache;self.rank=rank
        self.p=cache['proto'].to('cuda').float()
        self.c=cache['c0'].to('cuda').clone()
        self.roi=roi_for(cache,rank,'cuda')
        self.box=cache['boxes_input'][rank:rank+1].to('cuda')
    def evaluate(self,coefficients,full_check=False):
        with torch.inference_mode():
            self.c[self.rank]=torch.as_tensor(coefficients,dtype=torch.float32,device='cuda')
            low=(self.c @ self.p.flatten(1))[self.rank].reshape(1,*self.p.shape[-2:])
            grid=ops.scale_masks(low[None],self.cache['native_decode_hw'])[0,0]
            # crop_mask has a CPU-only rounded-box branch; this follows native CUDA semantics.
            first=ops.crop_mask((grid[None]>0).byte(),self.box)[0]
            final=ops.scale_masks(first[None,None],self.cache['original_hw'],ratio_pad=self.cache['ratio_pad'])[0,0].byte()
            rle=encode_rle(final)
            if full_check:
                actual=decode_native(self.cache,self.c.cpu())[self.rank]
                if not torch.equal(final,actual):raise AssertionError('selected decoder differs from full native decoder')
            return rle,first.cpu().numpy().astype(bool),grid[self.roi].double().cpu().numpy()
    def rms(self,directions):
        values=[]
        with torch.inference_mode():
            for value in np.asarray(directions,dtype=np.float64).reshape(-1,32):
                self.c[self.rank]=torch.as_tensor(value,dtype=torch.float32,device='cuda')
                low=(self.c @ self.p.flatten(1))[self.rank].reshape(1,*self.p.shape[-2:])
                grid=ops.scale_masks(low[None],self.cache['native_decode_hw'])[0,0]
                values.append(float(grid[self.roi].double().square().mean().sqrt()))
        return np.array(values)


def measure(rle,gt):
    from pycocotools import mask as m
    inter=int(m.area(m.merge([rle,gt],intersect=True)))
    union=int(m.area(rle))+int(m.area(gt))-inter
    return {'rle':rle,'intersection':inter,'union':union,'iou':inter/union if union else 0.}


def matched_coefficients(decoder,center,vectors,amplitude,tolerance):
    norm=decoder.rms(vectors)
    if not np.isfinite(norm).all() or np.any(norm<=1e-10):raise ValueError('zero/nonfinite pressure direction')
    deltas=np.asarray(vectors,dtype=np.float64)*(amplitude/norm)[:,None]
    scales=np.ones(len(deltas));history=[]
    for iteration in range(9):
        result=(np.asarray(center,dtype=np.float64)[None]+deltas*scales[:,None]).astype(np.float32)
        actual=decoder.rms(result.astype(np.float64)-np.asarray(center,dtype=np.float64))
        if not np.isfinite(actual).all() or np.any(actual<=1e-10):raise ValueError('rounded pressure vanished')
        error=np.abs(actual/amplitude-1)
        history.append({'corrections':iteration,'max_relative_error':float(error.max()),'bad_directions':int((error>tolerance).sum())})
        if np.all(error<=tolerance):break
        if iteration<8:
            bad=error>tolerance;scales[bad]*=amplitude/actual[bad]
    if not np.all(np.abs(actual/amplitude-1)<=tolerance):raise ValueError('pressure functional RMS differs')
    return result,actual,norm,{'scales':scales.tolist(),'history':history}
