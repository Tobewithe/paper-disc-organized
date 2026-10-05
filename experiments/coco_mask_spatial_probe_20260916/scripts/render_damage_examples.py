"""Six deterministic median-change cases, one per transition and size."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu


def main():
    p=argparse.ArgumentParser()
    for k in ('selection','images','annotations','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cases=json.loads(a.selection.read_text(encoding='utf-8'));coco=COCO(str(a.annotations))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    cw,ch=300,230;canvas=Image.new('RGB',(4*cw,70+len(cases)*ch),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,8),'Fixed original prediction: median IoU-change cases in each size / outcome group',fill='black',font=font)
    for col,title in enumerate(['Image crop','GT','Original prediction','Learned combination']):draw.text((col*cw+10,40),title,fill='black',font=font)
    for row,case in enumerate(cases):
        im=Image.open(a.images/case['filename']).convert('RGB');gt=coco.annToMask(coco.anns[case['annotation_id']]).astype(bool)
        m0=mu.decode(case['baseline_rle']).astype(bool);m1=mu.decode(case['corrected_rle']).astype(bool) if case['corrected_rle'] else np.zeros_like(gt)
        bx,by,bw,bh=case['prediction_box'];gx,gy,gw,gh=coco.anns[case['annotation_id']]['bbox']
        pad=max(bw,bh,gw,gh)*.3+8;x1=max(0,int(min(bx,gx)-pad));y1=max(0,int(min(by,gy)-pad));x2=min(im.width,int(max(bx+bw,gx+gw)+pad));y2=min(im.height,int(max(by+bh,gy+gh)+pad))
        crop=(x1,y1,x2,y2);original=np.asarray(im)
        for col,mask in enumerate([None,gt,m0,m1]):
            rgb=original.copy()
            if mask is not None:rgb[mask]=(rgb[mask]*.4+np.array([25,200,130])*.6).astype(np.uint8)
            tile=Image.fromarray(rgb).crop(crop);tile.thumbnail((cw-12,ch-48));ox=col*cw+(cw-tile.width)//2;oy=70+row*ch+24
            canvas.paste(tile,(ox,oy))
            label=f"{case['group']} / {case['area_group']} / GT {case['annotation_id']}" if col==0 else ('GT' if col==1 else f"IoU {case['baseline' if col==2 else 'corrected']['iou']:.3f}")
            draw.text((col*cw+7,70+row*ch+3),label,fill='black',font=small)
    canvas.save(a.out/'repair_damage_examples.png');(a.out/'selection.json').write_text(json.dumps(cases),encoding='utf-8')


if __name__=='__main__':main()
