"""Full-sequence paired videos, including both improvements and regressions."""
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw
from .common import Run,read_json,write_json,sha256
from .render import overlay,font


def main(root):
    data=read_json(root/'showcase/results.json');manifest=read_json(root/'input_manifest.json')
    selections=[('zebra_1','trajectory'),('zebra_2','trajectory'),('horse_3','trajectory'),('zebra_4','extension')]
    with Run(root,'paired_evidence_videos',dict(selections=selections,whole_sequence=True,fps=15,manual_trajectory_edit=False)) as run:
        receipts=[]
        for name,arm in selections:
            seq=next(s for s in manifest['sequences'] if s['name']==name)
            l=next(r for r in data['rows'] if r['sequence']==name and r['arm']=='motion')
            r=next(r for r in data['rows'] if r['sequence']==name and r['arm']==arm)
            left,right=read_json(l['track_path']),read_json(r['track_path'])
            cache=Path(seq['cache_path']) if seq.get('cache_path') else root/read_json(root/'features_pointer.json')['cache']/name
            det=read_json(cache/'detections.json');out=root/'showcase/videos';out.mkdir(exist_ok=True)
            dest=out/(name+'.mp4');cap=cv2.VideoCapture(str(root/seq['video']));n=int(seq['frames']/seq['fps']*15)
            ffmpeg=str(root/'tools/ffmpeg.exe') if (root/'tools/ffmpeg.exe').exists() else 'ffmpeg'
            proc=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r','15','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
            canvas=Image.new('RGB',(1600,900),'#f2f5fa');d=ImageDraw.Draw(canvas)
            d.text((40,27),'TraceLab  /  密集视频个体身份保持',font=font(24),fill='#2863dd')
            d.text((40,73),name+'  ·  完整序列配对比较',font=font(37),fill='#183249')
            d.text((40,134),'ByteTrack · 同检测基线',font=font(25),fill='#526c82')
            d.text((822,134),data['labels'][arm],font=font(25),fill='#087f8c')
            d.rounded_rectangle((40,627,1558,798),radius=16,fill='white')
            delta=r['IDF1']-l['IDF1'];accent='#087f8c' if delta>0 else '#b44842'
            d.text((66,651),'全视频 IDF1',font=font(23),fill='#657b8e')
            d.text((66,689),f"{l['IDF1']:.2f}  →  {r['IDF1']:.2f}   ({delta:+.2f})",font=font(35),fill=accent)
            d.text((710,651),'身份切换 IDSW',font=font(23),fill='#657b8e')
            d.text((710,695),f"{l['IDSW']}  →  {r['IDSW']}",font=font(32),fill='#29465e')
            d.text((40,832),'彩色轮廓与 ID 来自真实算法输出；全量成绩包含漏输和错接。',font=font(23),fill='#647a8e')
            source_idx=-1;img=None
            for j in range(n):
                target=min(seq['frames']-1,int(j*seq['fps']/15))
                while source_idx<target:
                    ok,img=cap.read();source_idx+=1
                    if not ok:raise RuntimeError('Unexpected video end')
                frame=canvas.copy();frame.paste(overlay(img,det[target],left,target,736,414),(40,182));frame.paste(overlay(img,det[target],right,target,736,414),(822,182))
                dd=ImageDraw.Draw(frame);dd.text((1210,653),f'{j/15:.1f} / {n/15:.1f} 秒',font=font(24),fill='#29465e')
                dd.text((1210,702),f'原帧 {target+1}',font=font(23),fill='#657b8e')
                if j==0:frame.save(out/(name+'.jpg'),quality=92)
                proc.stdin.write(np.asarray(frame).tobytes())
            proc.stdin.close();rc=proc.wait();cap.release();assert rc==0
            receipts.append(dict(sequence=name,arm=arm,path='videos/'+dest.name,poster='videos/'+name+'.jpg',frames=n,fps=15,sha256=sha256(dest),delta_IDF1=delta))
            print('VIDEO',receipts[-1],flush=True)
        write_json(root/'showcase/video_catalog.json',receipts);write_json(run.path/'SUMMARY.json',dict(videos=receipts))
        write_json(root/'paired_video_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
