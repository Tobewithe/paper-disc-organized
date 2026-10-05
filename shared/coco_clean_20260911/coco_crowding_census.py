"""Census of all available COCO2017 splits from ORIGINAL annotation IDs.

No model predictions, training labels, polygon pieces, or selected-subset JSON
enter the crowding calculation. Outputs include every image (empty included)
and every valid non-crowd instance. Crowds/invalid boxes are counted separately.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

DEFAULT_ROOT = Path('/root/autodl-tmp/coco_clean_20260911')
PUBLIC = Path('/autodl-pub/data/COCO2017')
BINS = ['I0_zero', 'I1_(0,0.5]', 'I2_(0.5,1]', 'I3_>1']
BOUNDARY_ATOL = 1e-10

def stable_ici(values):
    result=values.copy()
    for boundary in [.5,1.0]:
        result[np.abs(result-boundary)<=BOUNDARY_ATOL]=boundary
    return result

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def index_values(anns):
    if not anns: return np.empty(0), np.empty(0), np.empty((0,0)), np.empty(0)
    boxes=np.array([a['bbox'] for a in anns],dtype=np.float64)
    areas=boxes[:,2]*boxes[:,3]
    ends=boxes[:,:2]+boxes[:,2:]
    widths=np.maximum(0,np.minimum(ends[:,None,:],ends[None,:,:])-np.maximum(boxes[:,None,:2],boxes[None,:,:2]))
    overlap=widths.prod(-1)
    np.fill_diagonal(overlap,0)
    classes=np.array([a['category_id'] for a in anns])
    same=classes[:,None]==classes[None,:]
    return (overlap*same).sum(1)/areas, overlap.sum(1)/areas, overlap, areas

def scalar_reference(anns):
    values=[]
    for i,a in enumerate(anns):
        x,y,w,h=a['bbox'];total=0.0
        for j,b in enumerate(anns):
            if i==j or a['category_id']!=b['category_id']:continue
            xx,yy,ww,hh=b['bbox']
            total+=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
        values.append(total/(w*h))
    return np.array(values)

def bin_of(x): return BINS[0 if x==0 else 1 if x<=.5 else 2 if x<=1 else 3]
def mask_size(a):return 'small' if a['area']<1024 else 'medium' if a['area']<9216 else 'large'
def count_band(n):return '0' if n==0 else '1' if n==1 else '2-4' if n<=4 else '5-9' if n<=9 else '10+'
def ratio_band(r):return '0' if r==0 else '(0,25%]' if r<=.25 else '(25%,50%]' if r<=.5 else '(50%,75%]' if r<=.75 else '(75%,100%]'
def q(values):
    return {str(p):float(v) for p,v in zip([0,25,50,75,90,95,99,100],np.percentile(values,[0,25,50,75,90,95,99,100]))} if len(values) else {}

def self_test():
    # One instance has two neighbors each covering .3: cumulative ICI .6, not .3.
    a=[dict(id=1,category_id=1,bbox=[0,0,10,10]),dict(id=2,category_id=1,bbox=[0,0,3,10]),dict(id=3,category_id=1,bbox=[7,0,3,10]),dict(id=4,category_id=2,bbox=[0,0,10,10])]
    same,all_,_,_=index_values(a)
    np.testing.assert_allclose(same,scalar_reference(a),atol=1e-14,rtol=0)
    np.testing.assert_allclose(same,[.6,1,1,0],atol=1e-14,rtol=0)
    np.testing.assert_allclose(all_,[1.6,2,2,1.6],atol=1e-14,rtol=0)
    assert bin_of(0)==BINS[0] and bin_of(.5)==BINS[1] and bin_of(1)==BINS[2]
    assert count_band(0)=='0' and ratio_band(0)=='0'

def summarize_scope(image_rows, instance_rows, category_meta):
    inst=instance_rows
    total=len(inst)
    high=sum(r['ici_same']>.5 for r in inst)
    high_all=sum(r['ici_all']>.5 for r in inst)
    sizes={s:dict(instances=sum(r['mask_size']==s for r in inst),high_same=sum(r['mask_size']==s and r['ici_same']>.5 for r in inst)) for s in ['small','medium','large']}
    return dict(images=len(image_rows), valid_noncrowd_instances=total,
        same_class_bins=dict(Counter(r['ici_bin'] for r in inst)),
        all_class_bins=dict(Counter(bin_of(r['ici_all']) for r in inst)),
        high_same=high,high_same_fraction=high/total if total else None,
        high_all=high_all,high_all_fraction=high_all/total if total else None,
        tiny_bbox_lt100=sum(r['bbox_area']<100 for r in inst),
        high_same_tiny=sum(r['bbox_area']<100 and r['ici_same']>.5 for r in inst),
        images_any_high_same=sum(r['n_high_same']>0 for r in image_rows),
        images_any_high_all=sum(r['n_high_all']>0 for r in image_rows),
        images_at_least_2_high_same=sum(r['n_high_same']>=2 for r in image_rows),
        images_at_least_5_high_same=sum(r['n_high_same']>=5 for r in image_rows),
        images_high_same_ratio_gt50pct=sum(r['high_same_ratio']>.5 for r in image_rows),
        images_no_ordinary_gt=sum(r['n_instances']==0 for r in image_rows),
        high_same_count_distribution=dict(Counter(count_band(r['n_high_same']) for r in image_rows)),
        high_same_ratio_distribution=dict(Counter(ratio_band(r['high_same_ratio']) for r in image_rows)),
        image_instance_count_quantiles=q([r['n_instances'] for r in image_rows]),
        image_high_same_count_quantiles=q([r['n_high_same'] for r in image_rows]),
        instance_ici_same_quantiles=q([r['ici_same'] for r in inst]),
        mean_instances_per_image=total/len(image_rows) if image_rows else None,
        same_class_box_iou_gt05_pairs=sum(r['same_class_pairs_box_iou_gt05'] for r in image_rows),
        all_class_box_iou_gt05_pairs=sum(r['all_class_pairs_box_iou_gt05'] for r in image_rows),
        size_breakdown=sizes)

def process_split(root, out, split, old_ids):
    path=root/'data/annotations'/f'instances_{split}.json'
    obj=json.loads(path.read_text())
    ims={im['id']:im for im in obj['images']}
    assert len(ims)==len(obj['images'])
    assert len({a['id'] for a in obj['annotations']})==len(obj['annotations'])
    anns=defaultdict(list);crowds=Counter();invalid=[];multipoly=0;crowd_n=0;regular_n=0
    cats={c['id']:c['name'] for c in obj['categories']}
    for a in obj['annotations']:
        assert a['image_id'] in ims and a['category_id'] in cats
        if a.get('iscrowd',0):crowds[a['image_id']]+=1;crowd_n+=1;continue
        regular_n+=1
        if a['bbox'][2]<=0 or a['bbox'][3]<=0:
            invalid.append(dict(annotation_id=a['id'],image_id=a['image_id'],bbox=a['bbox']));continue
        anns[a['image_id']].append(a)
        multipoly+=isinstance(a['segmentation'],list) and len(a['segmentation'])>1
    image_rows=[];instance_rows=[];max_delta=0.;checks=0
    for idx,iid in enumerate(sorted(ims)):
        aa=anns[iid]
        same_raw,all_raw,overlaps,areas=index_values(aa)
        same,all_=stable_ici(same_raw),stable_ici(all_raw)
        # Independently coded scalar check on scattered real images, not just fixtures.
        if idx%499==0:
            ref=scalar_reference(aa)
            diff=float(np.max(np.abs(ref-same_raw))) if len(aa) else 0.
            max_delta=max(max_delta,diff);checks+=1
            np.testing.assert_allclose(same_raw,ref,atol=1e-12,rtol=1e-12)
        eligible=areas>=100
        legacy_max=float(same_raw[eligible].max()) if np.any(eligible) else 0.
        n=len(aa);nh=int((same>.5).sum());nhall=int((all_>.5).sum())
        class_ids=np.array([a['category_id'] for a in aa])
        if n:
            iou=overlaps/(areas[:,None]+areas[None,:]-overlaps)
            samepairs=int(np.triu((iou>.05)&(class_ids[:,None]==class_ids[None,:]),1).sum())
            allpairs=int(np.triu(iou>.05,1).sum())
        else:samepairs=allpairs=0
        image_rows.append(dict(split=split,image_id=iid,file_name=ims[iid]['file_name'],n_instances=n,n_crowd=crowds[iid],
            n_high_same=nh,high_same_ratio=nh/n if n else 0.,n_high_all=nhall,high_all_ratio=nhall/n if n else 0.,
            n_tiny_bbox_lt100=int((~eligible).sum()),n_high_same_bbox_ge100=int(((same>.5)&eligible).sum()),
            max_ici_same=float(same.max()) if n else 0.,max_ici_all=float(all_.max()) if n else 0.,
            mean_ici_same=float(same.mean()) if n else 0.,legacy_max_ici_same_bbox_ge100=legacy_max,
            legacy_selected=int(legacy_max>.5),historical_manifest_selected=int(iid in old_ids),
            same_class_pairs_box_iou_gt05=samepairs,all_class_pairs_box_iou_gt05=allpairs))
        for a,s,al,ar,sraw,araw in zip(aa,same,all_,areas,same_raw,all_raw):
            instance_rows.append(dict(split=split,image_id=iid,annotation_id=a['id'],category_id=a['category_id'],category_name=cats[a['category_id']],
                bbox_area=float(ar),mask_area=a['area'],mask_size=mask_size(a),polygon_components=len(a['segmentation']) if isinstance(a['segmentation'],list) else -1,
                ici_same=float(s),ici_all=float(al),ici_same_raw=float(sraw),ici_all_raw=float(araw),ici_bin=bin_of(s),legacy_image_selected=int(legacy_max>.5)))
        if (idx+1)%20000==0:print(split,'processed',idx+1,'/',len(ims),flush=True)
    assert len(instance_rows)+len(invalid)==regular_n
    selected={r['image_id'] for r in image_rows if r['legacy_selected']}
    scopes={'full':summarize_scope(image_rows,instance_rows,cats)}
    for name,ids in [('legacy_dense',selected),('outside_legacy_dense',set(ims)-selected)]:
        scopes[name]=summarize_scope([r for r in image_rows if r['image_id'] in ids],[r for r in instance_rows if r['image_id'] in ids],cats)
    coverage=scopes['legacy_dense']['high_same']/scopes['full']['high_same']
    category_counts=defaultdict(Counter)
    for r in instance_rows:
        c=category_counts[r['category_id']];c['instances']+=1;c['high_same']+=r['ici_same']>.5;c['high_all']+=r['ici_all']>.5
        if r['legacy_image_selected']:c['legacy_instances']+=1;c['legacy_high_same']+=r['ici_same']>.5
    catrows=[dict(split=split,category_id=k,category_name=v,**{col:int(category_counts[k][col]) for col in ['instances','high_same','high_all','legacy_instances','legacy_high_same']}) for k,v in cats.items()]
    for name,rows in [('images',image_rows),('instances',instance_rows),('categories',catrows)]:
        with (out/f'{split}_{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(original_json=str(path),original_json_sha256=digest(path),images=len(ims),annotations=len(obj['annotations']),
        noncrowd=regular_n,crowd=crowd_n,invalid_noncrowd=invalid,multipolygon_noncrowd=multipoly,
        independent_scalar_check_images=checks,scalar_max_abs_difference=max_delta,scopes=scopes,
        legacy_manifest_check=dict(old_count=len(old_ids),recomputed_count=len(selected),exact_match=selected==old_ids,only_in_old=sorted(old_ids-selected),only_in_recomputed=sorted(selected-old_ids)),
        legacy_dense_high_same_coverage=coverage,
        image_examples_few_vs_many=sorted(image_rows,key=lambda r:r['max_ici_same'],reverse=True)[:5])
    print('CENSUS_SPLIT_PASS',split,json.dumps(dict(images=len(ims),instances=regular_n,high_same=scopes['full']['high_same'],legacy_exact_match=selected==old_ids)),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);args=parser.parse_args()
    root=args.root;out=root/'census';out.mkdir(exist_ok=True)
    self_test()
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='All available COCO2017 instance-detection/segmentation splits',
        definition=dict(instance_ici='sum of intersections with other same-category noncrowd GT bboxes / reference bbox area',all_class_secondary='same sum without category restriction',
            high='ICI > 0.5 after boundary stabilization',boundary_rule='Snap ICI within absolute 1e-10 of 0.5 or 1 to that exact boundary; raw values retained',
            legacy_image_selection='Unchanged historical RAW max ICI over reference bbox area >=100 pixels >0.5; neighbors may be tiny',
            no_mask_occlusion_claim=True,original_annotation_identity=True,empty_image_ratio_convention='0, with empty image count separately reported'),
        code_sha256=digest(Path(__file__)),splits={},archives={})
    for split in ['train2017','val2017','test2017']:
        zpath=PUBLIC/f'{split}.zip'
        with zipfile.ZipFile(zpath) as z:
            entries=z.infolist();pics=[e for e in entries if not e.is_dir() and e.filename.lower().endswith(('.jpg','.jpeg','.png'))]
            report['archives'][split]=dict(path=str(zpath),image_count=len(pics),archive_bytes=zpath.stat().st_size,uncompressed_image_bytes=sum(e.file_size for e in pics),
                gt_annotation_json_available=(root/'data/annotations'/f'instances_{split}.json').exists())
    for split,manifest in [('train2017','train_dense_source.txt'),('val2017','val_dense_source.txt')]:
        old={int(Path(s.strip()).stem) for s in (root/manifest).read_text().splitlines() if s.strip()}
        report['splits'][split]=process_split(root,out,split,old)
        assert report['archives'][split]['image_count']==report['splits'][split]['images']
        (out/'COCO2017_CROWDING_CENSUS.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    report['test2017_note']='Only image archive is available here. No original instance GT JSON; ICI is uncomputable and is not imputed as zero.'
    report['status']='PASS'
    (out/'COCO2017_CROWDING_CENSUS.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('CENSUS_ALL_PASS',flush=True)

if __name__=='__main__':main()
