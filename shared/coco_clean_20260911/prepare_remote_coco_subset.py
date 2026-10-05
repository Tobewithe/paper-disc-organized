"""Prepare a reproducible COCO subset from the server's official zip files."""
import argparse, json, shutil, zipfile
from pathlib import Path

from ultralytics.data.converter import convert_coco, coco91_to_coco80_class


def ids_from_list(path):
    out=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        # The synced lists may contain Windows absolute paths.
        raw=line.strip().replace('\\','/')
        name=Path(raw).name
        if name.endswith('.jpg'): out.append(int(Path(name).stem))
    return out


def make_split(root, zip_path, ann_obj, split, image_ids):
    by_id={im['id']:im for im in ann_obj['images']}
    selected=[by_id[i] for i in image_ids if i in by_id]
    selected_ids={im['id'] for im in selected}
    anns=[a for a in ann_obj['annotations'] if a['image_id'] in selected_ids]
    obj={'info':ann_obj.get('info',{}),'licenses':ann_obj.get('licenses',[]),'images':selected,
         'annotations':anns,'categories':ann_obj['categories']}
    inp=root/'conversion_input'/split; inp.mkdir(parents=True,exist_ok=True)
    (inp/f'instances_{split}.json').write_text(json.dumps(obj),encoding='utf-8')
    out=root/'conversion'/split
    if out.exists(): shutil.rmtree(out)
    convert_coco(labels_dir=str(inp),save_dir=str(out),use_segments=True)
    img_out=root/'images'/split; img_out.mkdir(parents=True,exist_ok=True)
    prefix=split+'/'
    with zipfile.ZipFile(zip_path) as z:
        names=set(z.namelist())
        for im in selected:
            member=prefix+im['file_name']
            if member not in names: raise FileNotFoundError(member)
            with z.open(member) as src, (img_out/im['file_name']).open('wb') as dst: shutil.copyfileobj(src,dst)
    lbl_src=out/'labels'/split; lbl_out=root/'labels'/split; lbl_out.mkdir(parents=True,exist_ok=True)
    for im in selected:
        src=lbl_src/Path(im['file_name']).with_suffix('.txt'); dst=lbl_out/src.name
        if src.exists(): shutil.copy2(src,dst)
        else: dst.write_text('',encoding='utf-8')
    list_path=root/f'{split}.txt'
    list_path.write_text(''.join(str(img_out/im['file_name'])+'\n' for im in selected),encoding='utf-8')
    return len(selected),len(anns)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--coco-root',type=Path,required=True)
    ap.add_argument('--train-list',type=Path,required=True); ap.add_argument('--val-list',type=Path,required=True); a=ap.parse_args()
    root=a.root.resolve(); root.mkdir(parents=True,exist_ok=True)
    ann_zip=a.coco_root/'annotations_trainval2017.zip'
    with zipfile.ZipFile(ann_zip) as z:
        train=json.loads(z.read('annotations/instances_train2017.json')); val=json.loads(z.read('annotations/instances_val2017.json'))
    tr=make_split(root,a.coco_root/'train2017.zip',train,'train2017',ids_from_list(a.train_list))
    va=make_split(root,a.coco_root/'val2017.zip',val,'val2017',ids_from_list(a.val_list))
    mapping=coco91_to_coco80_class(); cats=sorted(train['categories'],key=lambda c:c['id']); names={mapping[c['id']-1]:c['name'] for c in cats if mapping[c['id']-1]>=0}
    import yaml
    data={'path':str(root),'train':'train2017.txt','val':'val2017.txt','names':names}
    (root/'boundary_ownership_remote.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    (root/'PREPARED.json').write_text(json.dumps({'train_images':tr[0],'train_annotations':tr[1],'val_images':va[0],'val_annotations':va[1],'data_yaml':str(root/'boundary_ownership_remote.yaml')},indent=2),encoding='utf-8')
    print(json.dumps({'train':tr,'val':va,'yaml':str(root/'boundary_ownership_remote.yaml')}),flush=True)


if __name__=='__main__': main()
