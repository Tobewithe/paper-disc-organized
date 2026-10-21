"""Analytic SOCP answer, scale invariance, degeneracy and clustered statistics."""
from common import *
import argparse,time


def main():
    import numpy as np
    from margin_math import recenter,geometric_margin,draw
    from margin_stats import analyze
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();started=time.monotonic()
    cfg=read_json(STUDY/'CONFIG.json');B=np.eye(32);c=np.linspace(.1,2,32);labels=np.ones(32,dtype=bool)
    result,receipt=recenter(B,c,labels,cfg['solver'])
    require(receipt['converged'],'analytic SOCP not solved')
    require(np.max(np.abs(result-result.mean()))<1e-5,'analytic max-min optimum differs')
    require(abs(np.linalg.norm(result)-np.linalg.norm(c))<1e-8,'scale not preserved')
    margin=geometric_margin(B,result,labels,cfg['solver']);require(abs(margin-1/np.sqrt(32))<1e-6,'known margin answer')
    require(abs(margin-geometric_margin(B,13*result,labels,cfg['solver']))<1e-12,'scale invariance')
    _,bad=recenter(np.zeros((40,32)),c,np.zeros(40,dtype=bool),cfg['solver']);require(not bad['converged'],'singular metric accepted')
    errors=np.arange(320,dtype=float).reshape(10,32);ids=np.arange(10)
    v,l=draw(errors,errors.mean(0),ids,100,2,cfg);v2,l2=draw(errors,errors.mean(0),ids,100,2,cfg)
    require(np.array_equal(v,v2) and l==l2 and len(l)==24,'pressure reproducibility')
    require(all(np.array_equal(v[i],-v[i+1]) for i in range(0,len(v),2)),'antithetic pairing')
    synthetic=[]
    for image_id in [10,10,20,30,40]:
        synthetic.append({'image_id':image_id,'accepted':True,'proxy_valid':True,'margin_C':.01,'margin_W':.02,
         'endpoints':{k:{'iou':x} for k,x in [('N',.6),('G_old',.65),('C',.8),('W',.8)]},
         'pressure':[{'arm':a,'C':{'iou':.7},'W':{'iou':.78}} for a in ['S','R','M'] for i in range(8)]})
    stats=analyze(synthetic,cfg,'full')
    require(stats['decision']=='STABILITY_DESIGN_SUPPORTED_REFERENCE_PENDING','positive gate test')
    require(abs(stats['stress_iou_gain']['mean']-.08)<1e-12,'image reduction')
    for r in synthetic:
        for x in r['pressure']:x['W']['iou']=x['C']['iou']
    require(analyze(synthetic,cfg,'full')['decision']=='NO_MATERIAL_STABILITY_GAIN_AT_SCOPE','negative gate test')
    synthetic[0]['accepted']=False
    require(analyze(synthetic,cfg,'full')['decision']=='INCONCLUSIVE_NUMERICAL','numerical versus scientific separation')
    for r in synthetic[1:]:r['endpoints']['C']['iou']=.5
    small=analyze(synthetic,cfg,'smoke')
    require(small['witness_retention']['ci95'] is None and small['decision']=='ENGINEERING_ONLY','smoke empty-denominator reporting')
    from native_ops import matched_coefficients
    class RoundingMetric:
        def __init__(self):self.calls=0
        def rms(self,v):
            self.calls+=1
            return np.sqrt(np.mean(np.asarray(v).reshape(-1,32)**2,axis=1))*(1 if self.calls==1 else 1.0001)
    _,actual,_,receipt_norm=matched_coefficients(RoundingMetric(),np.zeros(32),np.ones((2,32)),.1,5e-5)
    require(len(receipt_norm['history'])==2 and np.max(np.abs(actual/.1-1))<=5e-5,'post-addition normalization correction')
    receipt={'status':'passed','tests':12,'elapsed_seconds':time.monotonic()-started,'source_hashes':science_files()}
    write_new(Path(args.output),receipt);print(json.dumps({'status':'passed','tests':12}))


if __name__=='__main__':main()
