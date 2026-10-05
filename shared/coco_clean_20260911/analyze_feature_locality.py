"""S052 secondary paired scale/location comparisons and explicit patch masks."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
import argparse,json,shutil
import numpy as np,pandas as pd,cv2
from pycocotools import mask as mu
from summarize_subtype_neighbor import stat,dump
def encode(m):
    x=mu.encode(np.asfortranarray(m.astype('uint8')));x['counts']=x['counts'].decode('ascii');return x
def mapped(mask):
    h,w=mask.shape;g=min(640/h,640/w);rh,rw=round(h*g),round(w*g);top,left=round((640-rh)/2-.1),round((640-rw)/2-.1)
    m=np.zeros((640,640),np.uint8);m[top:top+rh,left:left+rw]=cv2.resize(mask.astype('uint8'),(rw,rh),interpolation=cv2.INTER_NEAREST_EXACT);return m
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;d=pd.read_csv(out/'metrics.csv');g=pd.read_csv(out/'PATCH_GEOMETRY.csv');source=out.parent/'subtype_neighbor_20260912';meta={r['image_id']:r for r in json.loads((source/'manifest.json').read_text())['pairs']}
    shapes=json.loads((out/'RUNTIME.json').read_text())['stages'];patches=[]
    for iid,item in meta.items():
        geo=np.load(source/'pairs'/str(iid)/'geometry.npz');own=geo['own'];edit=geo['edit'];ys,xs=np.nonzero(edit);shift=np.array(item['placement']['control']);control=np.zeros(own.shape,bool);control[ys+shift[0],xs+shift[1]]=True
        source_masks={k:mapped(v) for k,v in dict(target=own,neighbor=edit,background_control=control).items()}
        pack=dict(image_id=iid,target=item['target'],rules='Actual640letterbox nearest-exact plus blockmax; dimensions divide640, equals adaptive_max_pool2d.',stages={})
        for stage,shape in shapes.items():
            h,w=shape[-2:];stage_masks={k:v.reshape(h,640//h,w,640//w).max((1,3)).astype(bool) for k,v in source_masks.items()};pack['stages'][stage]={k:encode(v) for k,v in stage_masks.items()}
            q=g[(g.image_id==iid)&(g.stage==stage)&(g.fill=='texture')]
            for mode in ['neighbor','background_control']:
                r=q[q['mode']==mode].iloc[0];assert stage_masks['target'].sum()==r.target_cells;assert stage_masks[mode].sum()==r.edit_cells;assert (stage_masks['target']&stage_masks[mode]).sum()==r.overlap_cells
        dump(out/'pairs'/str(iid)/'feature_patch_masks.json',pack)
    rows=[]
    comparisons=[('P3_minus_P4_edit','input0','edit_location','input1','edit_location'),('P3_edit_minus_target','input0','edit_location','input0','target'),('fused_target_minus_edit','fused','target','fused','edit_location')]
    for kind in ['same_neighbor','target_fn','background']:
        for fill in ['texture','local_color']:
            for direction in ['insert','restore']:
                z=d[(d.error_type==kind)&(d.fill==fill)&(d.direction==direction)].pivot(index='image_id',columns=['mode','stage','region'],values='delta_mask_iou')
                for label,s1,r1,s2,r2 in comparisons:
                    n=z[('neighbor',s1,r1)]-z[('neighbor',s2,r2)];b=z[('background_control',s1,r1)]-z[('background_control',s2,r2)]
                    rows.append(dict(error_type=kind,fill=fill,direction=direction,comparison=label,absolute=stat(n),paired=stat(n-b)))
    # No direct target/edit pooled-cell overlap sensitivity, same completed runs.
    sensitivity=[]
    for stage in ['input0','input1','input2','fused','decoder']:
        ids=set(g[(g['mode']=='neighbor')&(g.stage==stage)&(g.overlap_cells==0)].image_id)
        for kind in ['same_neighbor','target_fn']:
            for fill in ['texture','local_color']:
                for direction in ['insert','restore']:
                    q=d[(d.error_type==kind)&d.image_id.isin(ids)&(d.stage==stage)&(d.region=='edit_location')&(d.fill==fill)&(d.direction==direction)]
                    if not len(q):continue
                    z=q.pivot(index='image_id',columns='mode',values='delta_mask_iou');sensitivity.append(dict(error_type=kind,stage=stage,fill=fill,direction=direction,n=len(z),absolute=stat(z.neighbor),paired=stat(z.neighbor-z.background_control)))
    result=dict(experiment='S052_SECONDARY_LOCALITY',comparisons=rows,no_direct_pooled_cell_overlap=sensitivity,
        scope='Exploratory within-image paired differences after S052 aggregate observation. P3 minus P4 evidence does not prove natural fusion competition; cross-input combinations perturb joint distribution. No-overlap excludes direct pooled-cell overlap only, not receptivefield/context interaction. No new inference.')
    dump(out/'LOCALITY_ANALYSIS.json',result);shutil.copy2(__file__,out/Path(__file__).name)
    for r in rows:
        if r['error_type']=='same_neighbor' and r['comparison']=='P3_minus_P4_edit' and r['direction']=='insert':print('SCALE',json.dumps(r))
    for r in sensitivity:
        if r['error_type']=='same_neighbor' and r['stage']=='input0':print('NO_OVERLAP',json.dumps(r))
if __name__=='__main__':main()
