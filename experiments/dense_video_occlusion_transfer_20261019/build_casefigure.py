"""GT-attributed local improvement; does not claim a full physical-occlusion event."""
import json,pathlib,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from build_videos import load,OUT,read,V

def main():
 catalog=read(OUT/'video_catalog.json');case=next(v for v in catalog if v['sequence']=='duck_3');crop=case['fixed_crop_in_640x360'];frames=[1988,1991,2115];arms=['N','C','FULL','R'];outputs={a:load('duck_3',a) for a in arms};meta=next(s for s in read(V/'data/sequences.json')['sequences'] if s['name']=='duck_3');scale=np.asarray([640/meta['width'],360/meta['height']]*2);chain=set(case['highlighted_base_ids']);font=lambda s:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',s)
 canvas=Image.new('RGB',(1380,1138),'#f1f5f8');draw=ImageDraw.Draw(canvas);draw.text((28,15),'一次正确补接：同一只个体的身份链得到延续',font=font(31),fill='#142a3b');draw.text((28,61),'官方GT74仅用于事后核验；图中均为实际预测，允许原策略4帧确认缓冲。',font=font(19),fill='#536c7e');draw.text((28,88),'此例是轨迹断裂补接，尚未核定为完全物理遮挡恢复。',font=font(17),fill='#536c7e')
 cap=cv2.VideoCapture('C:/Users/Public/occlusion-training-20261019/real_videos/duck_3.mp4');idtable={};names={'N':'ByteTrack','C':'原确认基线 C','FULL':'学习可见性','R':'BoT-SORT'}
 for j,f in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f-1);ok,bgr=cap.read();assert ok;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);target=next(t for t in outputs['FULL'][f-1]['tracks'] if t['base_id'] in chain);det=target['det_id'];idtable[str(f)]={};draw.text((190+j*395,118),f'帧 {f} · '+['断裂前','重新检出','再次断裂'][j],font=font(20),fill='#142a3b')
  for i,a in enumerate(arms):
   im=Image.fromarray(rgb);d=ImageDraw.Draw(im);t=next((t for t in outputs[a][f-1]['tracks'] if t['det_id']==det),None);pid=t['identity'] if t else None;idtable[str(f)][a]=pid
   if t:
    box=np.asarray(t['box'])*scale;color='#91ee58' if a=='FULL' else '#e7b356' if a=='R' else '#ff8569';d.rectangle(tuple(box),outline=color,width=3)
   im=im.crop(crop).resize((382,215),Image.Resampling.LANCZOS);imdraw=ImageDraw.Draw(im);imdraw.rounded_rectangle((10,10,180,48),radius=5,fill='#102f40');imdraw.text((20,12),'ID '+str(pid) if pid else '未输出',font=font(23),fill='white');canvas.paste(im,(180+j*395,157+i*238))
 for i,a in enumerate(arms):
  y=157+i*238;draw.text((20,y+62),names[a],font=font(21),fill='#168c82' if a=='FULL' else '#142a3b');draw.text((20,y+99),'保持同一ID' if a in ['FULL','R'] else '出现身份分裂',font=font(15),fill='#536c7e')
 draw.text((28,1102),'整段 duck_3：学习模型相对 C +0.647 IDF1；BoT-SORT也已保持身份。总体11段增量+0.263，机制和稳定优越性仍未成立。',font=font(17),fill='#536c7e');canvas.save(OUT/'correct_reconnection_case.png');(OUT/'case_identity_table.json').write_text(json.dumps(idtable,indent=2));cap.release();print(idtable)
if __name__=='__main__':main()
