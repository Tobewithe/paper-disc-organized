"""Check whether large fitted coefficient norms explain zero-threshold masks."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from ultralytics.utils import ops
from candidate_lineage_probe import ROOT,read,save_csv,write_json,sha,need
from no_candidate_readout_probe import coefficient_logits

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    targets=[r for r in read(out/'targets.csv') if r['fit_status']=='ok'];rows=[]
    for iid in sorted({int(r['image_id']) for r in targets}):
        with np.load(ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz') as z:proto=torch.tensor(z['proto'],device='cuda');inp=tuple(map(int,z['input_shape']));shape=tuple(map(int,z['shape']))
        with np.load(out/'images'/f'{iid}.npz') as z:params={k:z[k] for k in z.files}
        for r in targets:
            if int(r['image_id'])!=iid:continue
            aid=int(r['annotation_id']);c0=torch.tensor(params[f'{aid}_original_coefficient'],device='cuda');c=torch.tensor(params[f'{aid}_free_coefficient_coefficient'],device='cuda')
            box=torch.tensor(params[f'{aid}_box'],device='cuda');scaled=c*(c0.norm()/c.norm())
            def decode(cc):return ops.process_mask(proto,cc[None],box,inp,upsample=True)[0]
            original=decode(c);normalized=decode(scaled)
            xor=int((original!=normalized).sum());origxor=int(((ops.scale_masks(original[None,None],shape)>.5)!=(ops.scale_masks(normalized[None,None],shape)>.5)).sum())
            rows.append(dict(image_id=iid,annotation_id=aid,original_norm=float(c0.norm()),fitted_norm=float(c.norm()),
                norm_ratio=float(c.norm()/c0.norm()),input_mask_xor=xor,original_mask_xor=origxor,
                direction_cosine=float(torch.nn.functional.cosine_similarity(c[None],c0[None]))))
    save_csv(out/'coefficient_scale_witness.csv',rows)
    result=dict(targets=len(rows),input_xor_total=sum(r['input_mask_xor'] for r in rows),original_xor_total=sum(r['original_mask_xor'] for r in rows),
        fitted_norm_ratio_median=float(np.median([r['norm_ratio'] for r in rows])),direction_cosine_median=float(np.median([r['direction_cosine'] for r in rows])),
        statement='No-bias zero-threshold logits are positively scale invariant mathematically. This witness restores each fitted coefficient to its original norm and checks exact mask pixels in fp32; no conclusions about learned-predictor attainability.',
        script_sha256=sha(__file__),csv_sha256=sha(out/'coefficient_scale_witness.csv'))
    write_json(out/'SCALE_WITNESS.json',result);print(json.dumps(result))

if __name__=='__main__':main()
