"""Side-by-side actual predictions, select best and worst sequence deltas explicitly."""
import pathlib,json,gzip,subprocess,sys,collections
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from build_showcase import V,OUT,pointer,read

FONTS=[ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',s) for s in [15,19,26]]
def load(name,arm):
 with gzip.open(pointer('evaluate')/arm/name/'tracks.json.gz','rt') as f:return json.load(f)
def panel(rgb,tracks,title,meta,weights,frame,full=False,focus=None,crop=None,highlight=None):
 im=Image.fromarray(rgb.copy());d=ImageDraw.Draw(im);w,h=im.size
 for t in tracks:
  sx,sy=640/meta['width'],360/meta['height'];b=[t['box'][i]*(sx if i%2==0 else sy) for i in range(4)];b=[int(round(v)) for v in b];identity=t['identity'];k=t['base_id']
  selected=t['det_id'] in (highlight or set());color=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[(identity*41)%180,155,245]]]),cv2.COLOR_HSV2RGB)[0,0]) if selected else (133,155,163);label='ID '+str(identity)
  if full:
   v=weights.get((frame,k));label+=f' | V {v:.2f}' if v is not None else ''
  d.rectangle(b,outline=color,width=3 if selected else 1)
  if selected:
   box=d.textbbox((b[0],b[1]),label,font=FONTS[0]);labelw=box[2]-box[0]+8;y=max(0,b[1]-24);d.rectangle((max(0,b[0]),y,min(639,b[0]+labelw),y+23),fill='#102f40');d.text((max(0,b[0])+3,y),label,font=FONTS[0],fill='white')
 if crop:im=im.crop(crop).resize((640,360),Image.Resampling.LANCZOS)
 result=Image.new('RGB',(640,407),'#102b3b');draw=ImageDraw.Draw(result);draw.text((15,10),title,font=FONTS[1],fill='white');draw.text((465,13),f'frame {frame:04d}',font=FONTS[0],fill='#b8cfdd');result.paste(im,(0,47));return result

def main():
 OUT.mkdir(parents=True,exist_ok=True);dest=OUT/'videos';dest.mkdir(exist_ok=True);ev=read(pointer('evaluate')/'SUMMARY.json');deltas=ev['per_video_FULL_minus_C_IDF1'];ordered=sorted(deltas,key=deltas.get);names=list(dict.fromkeys([ordered[-1],ordered[0]]));seqmap={s['name']:s for s in read(V/'data/sequences.json')['sequences']};catalog=[]
 for ci,name in enumerate(names):
  outputs={a:load(name,a) for a in ['N','C','FULL','R']};full=outputs['FULL'];c=outputs['C'];different=[]
  for f,g in zip(full,c):
   byid={t['base_id']:t for t in g['tracks']}
   for t in f['tracks']:
    if t['identity']!=byid[t['base_id']]['identity']:different.append((f['frame'],t['base_id']))
  center,focus=different[0] if different else (len(full)//2,None)
  if not different and not full[center-1]['tracks']:
   populated=[f['frame'] for f in full if f['tracks']]
   if populated:center=min(populated,key=lambda f:abs(f-center))
  lo=max(1,center-70);hi=min(len(full),center+130);meta=seqmap[name]
  focus_track=next((t for t in full[center-1]['tracks'] if t['base_id']==focus),None)
  if focus_track is None and full[center-1]['tracks']:focus_track=max(full[center-1]['tracks'],key=lambda t:t['score'])
  identity=focus_track['identity'] if focus_track else None;chain={t['base_id'] for f in full[lo-1:hi] for t in f['tracks'] if t['identity']==identity};bs=np.asarray([t['box'] for f in full[lo-1:hi] for t in f['tracks'] if t['base_id'] in chain]).reshape(-1,4);crop=None
  if len(bs):
   scale=np.asarray([640/meta['width'],360/meta['height']]*2);bs=bs*scale;loxy=bs[:,:2].min(0)-[60,40];hixy=bs[:,2:].max(0)+[60,40];c=(loxy+hixy)/2;ww=max(230,hixy[0]-loxy[0],(hixy[1]-loxy[1])*16/9);hh=ww*9/16
   ww=min(640,ww);hh=min(360,hh);cx=np.clip(c[0],ww/2,640-ww/2);cy=np.clip(c[1],hh/2,360-hh/2);crop=[int(cx-ww/2),int(cy-hh/2),int(cx+ww/2),int(cy+hh/2)]
  vis=np.load(pointer('evaluate')/(name+'_visibility.npz'));weights={(int(f),int(k)):float(v) for f,k,v,p in zip(vis['frames'],vis['ids'],vis['FULL_visibility'],vis['present']) if p};cap=cv2.VideoCapture(str(pathlib.Path('C:/Users/Public/occlusion-training-20261019/real_videos')/(name+'.mp4')));cap.set(cv2.CAP_PROP_POS_FRAMES,lo-1)
  temp=dest/(name+'_temporary.mp4');writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),min(25,float(meta['fps'])),(1280,888));poster=None
  for frame in range(lo,hi+1):
   ok,bgr=cap.read();assert ok;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);canvas=Image.new('RGB',(1280,888),'#eaf0f4');draw=ImageDraw.Draw(canvas);draw.text((20,8),f"{name}  |  完整模型相对 C：IDF1 {deltas[name]:+.3f} 点",font=FONTS[2],fill='#142a3b')
   titles={'N':'ByteTrack 原生','C':'三观测确认 C','FULL':'完整模型：学习可见性加权','R':'BoT-SORT 已有参照'}
   highlighted={t['det_id'] for t in full[frame-1]['tracks'] if t['base_id'] in chain}
   for j,a in enumerate(['N','C','FULL','R']):canvas.paste(panel(rgb,outputs[a][frame-1]['tracks'],titles[a],meta,weights,frame,a=='FULL',focus,crop,highlighted),((j%2)*640,50+(j//2)*407))
   draw.text((20,864),'同一固定裁剪 / 时间轴；彩色标记所选身份链，其它检测保留淡框；V 为预测比例；确认允许4帧延迟。',font=FONTS[0],fill='#536c7e');writer.write(cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR))
   if frame==center:poster=canvas.copy()
  cap.release();writer.release();target=dest/(name+'.mp4');subprocess.run(['ffmpeg','-y','-v','error','-i',str(temp),'-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True);temp.unlink();poster=poster or canvas;poster.save(dest/(name+'.jpg'),quality=92)
  kind='最大收益视频' if ci==0 and deltas[name]>0 else '最大退化视频' if deltas[name]<0 else '无新增收益视频'
  catalog.append({'sequence':name,'title':kind+' · '+name,'description':f'选择规则：11段中完整模型相对C的IDF1差值排序，展示最大与最小两段；片段以首个身份分歧为中心，无分歧时取距中点最近的有检测帧。整段增量{deltas[name]:+.3f}点。片段本身未核定为完全物理遮挡，不能冒充该类成功案例。','video':'videos/'+name+'.mp4','poster':'videos/'+name+'.jpg','frame_range':[lo,hi],'fixed_crop_in_640x360':crop,'highlighted_base_ids':sorted(chain),'focus_base_id':focus,'selection':'largest/smallest full-video IDF1 delta vs C; first disagreement or nearest populated midpoint'})
 (OUT/'video_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8');print(catalog)
if __name__=='__main__':main()
