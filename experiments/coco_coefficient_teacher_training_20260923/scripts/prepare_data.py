import argparse, json, random, shutil
from collections import defaultdict
from pathlib import Path
import yaml
from ultralytics.data.converter import convert_coco, coco91_to_coco80_class
from ultralytics import YOLO


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--weights',type=Path,required=True);a=ap.parse_args()
    out=Path(__import__('os').environ['RESEARCH_RUN_DIRECTORY'])
    d=a.root/'data';d.mkdir(exist_ok=True)
    source=json.loads((a.data/'annotations/instances_train2017.json').read_text())
    ids=sorted(i['id'] for i in source['images']);random.Random(20260923).shuffle(ids)
    fit,select=ids[:2000],ids[2000:2500];wanted=set(fit+select)
    subset={**source,'images':[i for i in source['images'] if i['id'] in wanted], 'annotations':[v for v in source['annotations'] if v['image_id'] in wanted]}
    inp=d/'conversion_input';inp.mkdir(exist_ok=True)
    (inp/'instances_train2017.json').write_text(json.dumps(subset))
    conversion=d/'converted';assert not conversion.exists()
    convert_coco(str(inp),str(conversion),use_segments=True,cls91to80=True)
    image_dir=d/'images/train2017';image_dir.mkdir(parents=True)
    label_dir=d/'labels/train2017';label_dir.mkdir(parents=True)
    images={v['id']:v for v in subset['images']};anns=defaultdict(list)
    for ann in subset['annotations']:anns[ann['image_id']].append(ann)
    mappings=[];excluded=[];rows=0;multi=0;classmap=coco91_to_coco80_class()
    for iid in fit+select:
        name=images[iid]['file_name'];src=a.data/'images/train2017'/name;assert src.is_file(),src
        (image_dir/name).symlink_to(src)
        label=conversion/'labels/train2017'/Path(name).with_suffix('.txt')
        lines=label.read_text().splitlines() if label.exists() else []
        accepted=[];seen=set()
        for ann in anns[iid]:
            x,y,w,h=ann['bbox'];key=(ann['category_id'],x,y,w,h)
            reason='crowd' if ann.get('iscrowd') else 'nonpositive_box' if w<=0 or h<=0 else 'duplicate_box' if key in seen else None
            if reason:excluded.append({'annotation_id':ann['id'],'reason':reason});continue
            assert ann.get('segmentation'),ann['id']
            assert isinstance(ann['segmentation'],list),ann['id']
            seen.add(key);accepted.append(ann);multi+=len(ann['segmentation'])>1
        assert len(lines)==len(accepted),(iid,len(lines),len(accepted))
        for row,(line,ann) in enumerate(zip(lines,accepted)):
            vals=[float(x) for x in line.split()]
            assert len(vals)>=7 and len(vals)%2==1,(iid,row,len(vals))
            assert int(vals[0])==classmap[ann['category_id']-1]
            assert all(0<=x<=1 for x in vals[1:]),(iid,row)
            mappings.append({'image_id':iid,'line':row,'annotation_id':ann['id'],'polygon_count':len(ann['segmentation'])})
        (label_dir/Path(name).with_suffix('.txt')).write_text('\n'.join(lines)+ ('\n' if lines else ''))
        rows+=len(lines)
    split={'fit_ids':fit,'selection_ids':select,'split_seed':20260923,'val_ids':sorted(i['id'] for i in json.loads((a.data/'annotations/instances_val2017.json').read_text())['images'])}
    (d/'SPLIT.json').write_text(json.dumps(split))
    for key,ii in [('fit',fit),('selection',select),('smoke',fit[:8])]:
        (d/f'{key}.txt').write_text(''.join(str(image_dir/images[i]['file_name'])+'\n' for i in ii))
    names=YOLO(str(a.weights)).model.names
    for key,train in [('pilot','fit.txt'),('smoke','smoke.txt')]:
        cfg={'path':str(d),'train':str(d/train),'val':str(d/'selection.txt'),'names':names}
        (d/f'{key}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (d/'ANNOTATION_MAP.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in mappings))
    report={'images':2500,'fit_images':2000,'selection_images':500,'label_instances':rows,'multi_polygon_instances':multi,'excluded':excluded,'converter':'Ultralytics 8.4.100 official convert_coco','instance_mapping_verified':True,'source':str(a.data),'original_evaluation_annotations':str(a.data/'annotations/instances_val2017.json')}
    (out/'RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
