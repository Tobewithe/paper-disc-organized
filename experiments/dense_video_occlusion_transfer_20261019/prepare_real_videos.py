import concurrent.futures,hashlib,json,pathlib,subprocess,tarfile,sys,os
REPO=pathlib.Path('C:/Dpan/codexproject/paper-disc-organized');sys.path.insert(0,str(REPO));os.environ['OCC_ROOT']=str(REPO)
from shared.dense_video_occlusion.common import Run,save,sha
OUT=pathlib.Path('C:/Users/Public/occlusion-training-20261019')

def main():
 manifest=json.loads(pathlib.Path('C:/Users/Public/thesis-video-v4-20261019/verified_delivery/input_manifest.json').read_text(encoding='utf-8'))
 roots=[pathlib.Path('C:/Users/Public/thesis-video-20261018/data'),pathlib.Path('C:/Users/Public/thesis-video-v2-20261018/data'),pathlib.Path('C:/Users/Public/thesis-video-v2-20261018/transfer_data'),pathlib.Path('C:/Users/Public/thesis-video-v3-20261019/data'),pathlib.Path('C:/Users/Public/thesis-video-v4-20261019/data')]
 index={}
 for root in roots:
  for p in root.rglob('*.mp4'):index.setdefault(p.name,[]).append(p)
 resume='--resume' in sys.argv
 dest=OUT/'real_videos';dest.mkdir(exist_ok=resume)
 with Run('real_video_derivation',{'resolution':[640,360],'training_steps':0,'frame_policy':'preserve decoded frame ordering; no interpolation'}) as run:
  def process(s):
   choices=index.get(s['name']+'.mp4',[]);source=next((p for p in choices if sha(p)==s['video_sha256']),None);assert source is not None,s['name']
   target=dest/(s['name']+'.mp4');cmd=['ffmpeg','-y','-v','error','-threads','2','-i',str(source),'-vf','scale=640:360','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-fps_mode','passthrough',str(target)]
   if not(resume and target.exists()):subprocess.run(cmd,check=True)
   probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height','-of','json',str(target)],text=True))['streams'][0]
   assert int(probe['nb_read_frames'])==s['frames'],(s['name'],probe,s['frames'])
   print('VIDEO',s['name'],target.stat().st_size,flush=True)
   return {'name':s['name'],'source_sha256':s['video_sha256'],'sha256':sha(target),'frames':s['frames'],'size_bytes':target.stat().st_size,'width':640,'height':360,'command':cmd}
  rows=list(concurrent.futures.ThreadPoolExecutor(3).map(process,manifest['sequences']))
  save(dest/'MANIFEST.json',rows);save(run.path/'SUMMARY.json',{'videos':len(rows),'frames':sum(r['frames'] for r in rows),'size_bytes':sum(r['size_bytes'] for r in rows)})
  with tarfile.open(OUT/'real-videos.tar','w') as tf:tf.add(dest,arcname='real_videos')
  save(OUT/'VIDEO_RECEIPT.json',{'run_id':run.id,'sha256':sha(OUT/'real-videos.tar'),'size_bytes':(OUT/'real-videos.tar').stat().st_size,'videos':len(rows)})
if __name__=='__main__':main()
