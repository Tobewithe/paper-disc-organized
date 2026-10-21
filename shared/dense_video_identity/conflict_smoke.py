from pathlib import Path
import numpy as np
from .common import Run,read_json,write_json,setup_paths
from .learn import load_identity,load_cache,embed_array
from .conflict_selector import components,sequence,Selector,FEATURES

def smoke(root):
    setup_paths(root);cfg=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json');model,ck=load_identity(root)
    with Run(root,'conflict_engineering_smoke',dict(frames=32,training=0,criterion='exact baseline plus conflict partition invariants')) as run:
        assert components([(0,0),(1,1)],[(0,1),(1,0)])==[([(0,0),(1,1)],[(0,1),(1,0)])]
        assert components([(0,0)],[(0,0)])==[]
        assert sum(p.numel() for p in Selector().parameters())==803
        checks=[]
        for name in ['pig_4','horse_5']:
            seq=next(s for s in manifest['sequences'] if s['name']==name);records,raw=load_cache(root,seq);z=embed_array(model,ck,raw)
            out,ev=sequence('motion',records[:16],z,cfg)
            old=read_json(Path(read_json(root/'v1_reference.json')['dev'])/'bytetrack'/name/'tracks.json')[:16]
            assert old==out,name
            alt,_=sequence('all_appearance',records[:16],z,cfg)
            for f in alt:
                ids=[t['identity'] for t in f['tracks']];ds=[t['det_id'] for t in f['tracks']]
                assert len(ids)==len(set(ids)) and len(ds)==len(set(ds))
            checks.append(name)
        write_json(run.path/'SUMMARY.json',dict(status='passed',baseline_exact=checks,feature_dim=len(FEATURES),parameters=803,frames=32))
        print('SMOKE_PASSED',checks,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();smoke(a.root)
