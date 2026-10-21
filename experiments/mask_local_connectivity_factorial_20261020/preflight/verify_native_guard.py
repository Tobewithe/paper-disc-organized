"""Supplementary zero-update identity guard required before formal launch."""
import sys
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STUDY/'scripts'))
from lcf_runtime import *

def main():
    import torch
    from native_stream import NativeBatches
    from adapter_model import CoefficientAdapter
    from native_output_bridge import decode_native,encode_rle
    from s1_geometry import iter_prediction_array
    pointer('smoke')
    with Run('native_guard') as run:
        profile,env=environment();write(run.path/'ENVIRONMENT.json',env)
        stream=NativeBatches(run.config,STUDY/'inputs','val',run.path/'native_val')
        before=state_hash(stream.model)
        caches,rows,features=stream.batch(0,masks=True)
        record={r['image_id']:r for r in read(run.config['historical_native_trace'])['images']}
        old=CoefficientAdapter('B').cuda().eval()
        old.load_state_dict(torch.load(run.config['existing_G'],map_location='cpu',weights_only=True)['state_dict'])
        wanted={c['image_id'] for c in caches};reference={i:[] for i in wanted}
        for row in iter_prediction_array(run.config['existing_G_predictions']):
            if row['image_id'] in wanted:reference[row['image_id']].append(row)
        from pycocotools import mask as masks
        checked=0;feature_tensors=0
        with torch.no_grad():
            for c,predictions in zip(caches,rows):
                previous=torch.load(Path(run.config['historical_native_trace']).parent/record[c['image_id']]['cache'],map_location='cpu',weights_only=True)
                for key in ('proto','c0','boxes_input','scores','classes'):
                    require(torch.equal(c[key].detach().cpu(),previous[key]),'native tensor '+key)
                require(len(c['features'])==len(previous['features']),'point-F count')
                for a,b in zip(c['features'],previous['features']):
                    require(torch.equal(a.cpu(),b),'native point-F identity');feature_tensors+=1
                native=decode_native(c);changed=decode_native(c,c['c0']+old(c,'eval:0'))
                actual=[]
                for i,p in enumerate(predictions):
                    require(encode_rle(native[i])==p['segmentation'],'native RLE')
                    actual.append(dict(p,segmentation=encode_rle(changed[i])) if int(masks.area(p['segmentation']))>=9216 else dict(p))
                require(actual==reference[c['image_id']],'old G full JSON/RLE mismatch')
                checked+=len(actual)
        require(state_hash(stream.model)==before,'base state changed')
        require(time.monotonic()-run.start<=run.config['hours']['smoke']*3600,'native guard time')
        run.finish({'status':'passed','images':16,'native_and_old_G_exact_detection_rows':checked,
                    'point_F_exact_tensors':feature_tensors,'optimizer_steps':0,
                    'source_sha256':sha(__file__),'native_source_branch_geometry_exact':True})

if __name__=='__main__':main()
