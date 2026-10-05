import argparse,csv,json,random
from pathlib import Path
import yaml
from recording import atomic_json,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.study/'protocol.json').read_text());dest=a.study/'data';dest.mkdir(exist_ok=False)
    source=yaml.safe_load(Path(cfg['source_data']).read_text());root=Path(source['path'])
    pool=Path(source['train']).read_text().splitlines();assert len(pool)==len(set(pool))==20000
    train=sorted(random.Random(cfg['sample_seed']).sample(sorted(pool),cfg['train_images']))
    assert all(Path(f).is_file() for f in train)
    raw=json.loads((root/'annotations/instances_val2017.json').read_text());images={x['id']:x for x in raw['images']}
    randomids=sorted(random.Random(cfg['sample_seed']+1).sample(sorted(images),cfg['random_val_images']))
    with Path(cfg['selection']).open(encoding='utf-8-sig',newline='') as f:selection=list(csv.DictReader(f))
    groups={c:{r['pair_id']:r for r in selection if r['cohort']==c} for c in ['raw_geometry_small','matched_small_control']}
    common=sorted(set.intersection(*(set(v) for v in groups.values())),key=int)
    pairs=random.Random(cfg['sample_seed']+2).sample(common,cfg['panel_pairs'])
    panel=[groups[c][pair] for pair in sorted(pairs,key=int) for c in groups]
    panelids=sorted({int(r['image_id']) for r in panel}); union=sorted(set(randomids)|set(panelids))
    # Full original annotations and competitors retained for every evaluated image.
    allann={x['id']:x for x in raw['annotations']}
    assert all(allann[int(r['annotation_id'])]['image_id']==int(r['image_id']) for r in panel)
    (dest/'train.txt').write_text('\n'.join(train)+'\n')
    for name,ids in [('val_native',randomids[:32]),('val_union',union)]:
        (dest/(name+'.txt')).write_text('\n'.join(str(root/'images/val2017'/images[i]['file_name']) for i in ids)+'\n')
    for name,vallist in [('train','val_native'),('evaluate','val_union')]:
        (dest/(name+'.yaml')).write_text(yaml.safe_dump(dict(path=str(root),train=str(dest/'train.txt'),val=str(dest/(vallist+'.txt')),names=source['names']),sort_keys=False))
    with (dest/'panel.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(panel[0]));w.writeheader();w.writerows(panel)
    manifest=dict(random_image_ids=randomids,panel_image_ids=panelids,union_image_ids=union,
        panel_annotation_ids=[int(r['annotation_id']) for r in panel],sampling_seed=cfg['sample_seed'],
        train_images=len(train),train_pool='existing random 20,000 training images; existing official converted labels reused',
        panel_pairs=len(pairs),panel_targets=len(panel),union_images=len(union),
        random_gt_count=sum(x['image_id'] in randomids and not x['iscrowd'] for x in raw['annotations']),
        limitation='128 targets sampled from previously matched failure/control pairs, not all 438 targets; random val subset reported separately')
    atomic_json(dest/'manifest.json',manifest);atomic_json(a.out/'COMPLETE.json',dict(status='complete',at=now(),**manifest))
    print('DATA_PREPARED',json.dumps(manifest),flush=True)
if __name__=='__main__':main()
