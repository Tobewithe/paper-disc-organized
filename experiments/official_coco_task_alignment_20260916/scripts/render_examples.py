"""Annotate saved binary-mask images with Pillow, no Torch or plotting runtimes."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap=argparse.ArgumentParser()
ap.add_argument("--source",type=Path,required=True)
ap.add_argument("--out",type=Path,required=True)
a=ap.parse_args()
examples=json.loads((a.source/"examples.json").read_text())
width,height=330,275
canvas=Image.new("RGB",(5*width,len(examples)*height),"white")
draw=ImageDraw.Draw(canvas)
font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",17)

def boundary(mask):
    pad=np.pad(mask,1,constant_values=False)
    eroded=mask & pad[:-2,1:-1] & pad[2:,1:-1] & pad[1:-1,:-2] & pad[1:-1,2:]
    return mask & ~eroded

for row,e in enumerate(examples):
    data=np.load(a.source/e["file"])
    for col in range(5):
        original=Image.fromarray(data["image"])
        scale=min((width-12)/original.width,(height-66)/original.height)
        original=original.resize((round(original.width*scale),round(original.height*scale)),Image.Resampling.BICUBIC)
        size=original.size
        pixels=np.array(original)
        gt=np.array(Image.fromarray(data["gt"]).resize(size,Image.Resampling.NEAREST))
        pixels[boundary(gt)]=(0,220,230)
        if col:
            mask=np.array(Image.fromarray(data["masks"][col-1]).resize(size,Image.Resampling.NEAREST))
            pixels[boundary(mask)]=(255,145,0)
            title=f"{['A: original','B: head only','C: position only','D: both'][col-1]}\nIoU {e['ious'][col-1]:.3f}"
        else:title=f"GT {e['annotation_id']}\ncyan: GT; orange: prediction"
        x=col*width+(width-size[0])//2;y=row*height+61
        canvas.paste(Image.fromarray(pixels),(x,y))
        draw.multiline_text((col*width+10,row*height+8),title,fill="black",font=font,spacing=4)
canvas.save(a.out)
