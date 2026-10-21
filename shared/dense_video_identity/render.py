"""Render truthful side-by-side videos from original frames and frozen predictions."""
from __future__ import annotations
import colorsys
from functools import lru_cache
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .common import read_json,sha256,write_json


@lru_cache(None)
def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",size)


def color(identity):
    rgb=colorsys.hsv_to_rgb(((identity or 0)*.381966)%1,.6,.98)
    return tuple(int(v*255) for v in rgb)


def overlay(frame, record, outputs, idx, width=900, height=506):
    h,w=frame.shape[:2];sx,sy=width/w,height/h
    rgb=cv2.cvtColor(cv2.resize(frame,(width,height)),cv2.COLOR_BGR2RGB)
    layer=rgb.copy();tracks=outputs[idx]["tracks"]
    for t in tracks:
        p=record["polygons"][t["det_id"]]
        if len(p)>=3:
            pts=(np.asarray(p)*[sx,sy]).astype(np.int32)
            cv2.fillPoly(layer,[pts],color(t["identity"]) if t["identity"] is not None else (255,195,70))
    rgb=cv2.addWeighted(layer,.2,rgb,.8,0)
    for t in tracks:
        identity=t["identity"];c=color(identity) if identity is not None else (255,195,70)
        p=record["polygons"][t["det_id"]]
        if len(p)>=3:cv2.polylines(rgb,[(np.asarray(p)*[sx,sy]).astype(np.int32)],True,c,1,cv2.LINE_AA)
        if identity is not None:
            points=[]
            for f in range(max(0,idx-20),idx+1):
                tt=next((x for x in outputs[f]["tracks"] if x["identity"]==identity),None)
                if tt:points.append([int((tt["box"][0]+tt["box"][2])*.5*sx),int((tt["box"][1]+tt["box"][3])*.5*sy)])
            if len(points)>1:cv2.polylines(rgb,[np.asarray(points)],False,c,2,cv2.LINE_AA)
        x,y=int(t["box"][0]*sx),int(t["box"][1]*sy)
        text=f"ID {identity}" if identity is not None else "WAIT"
        if t["status"]=="recovered":text+=" +"
        tw=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.52,1)[0][0]
        x=max(1,min(x,width-tw-14));y=max(25,min(y,height-1))
        cv2.rectangle(rgb,(x,y-24),(x+tw+12,y),(15,31,49),-1)
        cv2.putText(rgb,text,(x+6,y-7),cv2.FONT_HERSHEY_SIMPLEX,.52,c,1,cv2.LINE_AA)
    return Image.fromarray(rgb)


def render(root:Path):
    catalog=read_json(root/"showcase/catalog.json");out=root/"showcase/videos";out.mkdir(exist_ok=True)
    selections=[s for s in catalog["sequences"] if s["split"]=="demo"]+[s for s in catalog["sequences"] if s["name"] in {"pig_2","horse_1"}]
    outputs=[]
    for k,seq in enumerate(selections):
        left=read_json(root/seq["tracks"]["deepocsort"]);right=read_json(root/seq["tracks"]["utility_recovery"])
        det=read_json(root/seq["detections"]);events=read_json(root/seq["events"]["utility_recovery"])["recovery_events"]
        cap=cv2.VideoCapture(str(root/seq["video"]));fps=15
        n=int(seq["frames"]/seq["fps"]*fps);dest=out/f"comparison_{k+1:02d}.mp4"
        proc=subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1920x1080","-r",str(fps),"-i","-","-an","-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(dest)],stdin=subprocess.PIPE)
        base=Image.new("RGB",(1920,1080),"#f2f5fb");d=ImageDraw.Draw(base)
        d.text((44,34),"TraceLab  /  密集视频个体跟踪",font=font(25),fill="#2863dd")
        d.text((44,84),"从可见个体，到跨帧身份连续",font=font(42),fill="#1b304b")
        d.text((44,143),"同一视频 · 同一检测 · 身份记忆与重现决策对照",font=font(23),fill="#6b7e94")
        for x,title,accent in [(44,"对照：Deep OC-SORT","#6a87af"),(976,"候选：收益记忆 + 重现确认","#2863dd")]:
            d.rounded_rectangle((x,192,x+900,774),radius=16,fill="white",outline="#dbe4ef",width=2)
            d.rectangle((x,192,x+7,245),fill=accent)
            d.text((x+20,205),title,font=font(24),fill="#20374f")
        d.rounded_rectangle((44,805,1876,974),radius=16,fill="white",outline="#dbe4ef",width=2)
        d.text((70,831),"现场状态",font=font(22),fill="#62778f")
        d.text((70,1003),"公开基准：真实 ID 评价；算法接回不自动等于正确恢复。" if seq["gt"] else "现场应用视频：暂无跨帧 ID 标注，本片仅展示算法行为。",font=font(23),fill="#60758c")
        source_idx=-1;img=None
        for j in range(n):
            target=min(seq["frames"]-1,int(j*seq["fps"]/fps))
            while source_idx<target:
                ok,img=cap.read();source_idx+=1
                if not ok:raise RuntimeError("Video ended early during rendering")
            canvas=base.copy();canvas.paste(overlay(img,det[target],left,target),(44,253));canvas.paste(overlay(img,det[target],right,target),(976,253));dr=ImageDraw.Draw(canvas)
            curr=right[target]["tracks"];confirmed=sum(t["identity"] is not None for t in curr);pending=len(curr)-confirmed
            recovered=sum(e["type"]=="recovered" and e["frame"]<=target+1 for e in events)
            dr.text((230,826),f"已确认 {confirmed} 个",font=font(29),fill="#1c746d")
            dr.text((550,826),f"待确认 {pending} 个",font=font(29),fill="#a26a15")
            dr.text((880,826),f"算法重接 {recovered} 次",font=font(29),fill="#2863dd")
            dr.text((1280,829),f"{j/fps:05.1f} s / {n/fps:.1f} s",font=font(25),fill="#425974")
            dr.text((72,903),"稳定身份：彩色 ID       证据不足：WAIT       重新确认：ID +",font=font(25),fill="#546c87")
            dr.rounded_rectangle((1340,911,1836,920),radius=4,fill="#e6edf6")
            dr.rounded_rectangle((1340,911,1340+int(496*(j+1)/n),920),radius=4,fill="#2863dd")
            proc.stdin.write(np.asarray(canvas).tobytes())
            if j==min(n-1,75):canvas.save(out/f"poster_{k+1:02d}.jpg",quality=93)
        proc.stdin.close();ret=proc.wait();cap.release()
        if ret:raise RuntimeError(f"ffmpeg exited {ret}")
        outputs.append(dict(sequence=seq["name"],video=dest.relative_to(root).as_posix(),frames=n,fps=fps,sha256=sha256(dest)))
        print("VIDEO",dest,n,flush=True)
    concat=out/"concat.txt";concat.write_text("\n".join(f"file 'comparison_{i+1:02d}.mp4'" for i in range(len(outputs))),encoding="utf-8")
    final=out/"TraceLab_成果展示.mp4"
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","concat","-safe","0","-i",str(concat),"-c","copy","-movflags","+faststart",str(final)],check=True)
    write_json(out/"VIDEO_MANIFEST.json",dict(clips=outputs,combined=final.relative_to(root).as_posix(),sha256=sha256(final),source="frozen real predictions; no manual ID editing"))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();render(a.root)
