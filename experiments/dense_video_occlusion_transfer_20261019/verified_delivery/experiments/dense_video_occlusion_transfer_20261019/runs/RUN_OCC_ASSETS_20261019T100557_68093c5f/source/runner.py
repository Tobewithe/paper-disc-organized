import argparse,json,os,pathlib,subprocess,tarfile
from .common import ROOT,STUDY,Run,read,save,sha

def blender(run,script,args):
 install=read('/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion/INSTALLATION.json')
 assert sha(install['blender'])==install['blender_executable_sha256']
 env=dict(os.environ,LD_LIBRARY_PATH=install['private_runtime']+':'+os.environ.get('LD_LIBRARY_PATH',''))
 cmd=[install['blender'],'--background','--factory-startup','--python-exit-code','1','--python',str(ROOT/'shared/dense_video_occlusion'/script),'--',*args]
 save(run.path/'COMMAND.json',cmd)
 with open(run.path/'blender.log','w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,'Blender stage failed; inspect log'

def assets():
 with Run('assets',{'archive_sha256':'c0783c417a5e3190dcc2cb757a66ff8e1bcc916a4d7ab3a000d394cb4e261bac','training_steps':0}) as run:
  assert sha(ROOT/'assets-v1.tar.gz')==run.contract['archive_sha256']
  dest=ROOT/'assets';dest.mkdir(exist_ok=False)
  with tarfile.open(ROOT/'assets-v1.tar.gz') as t:
   for m in t.getmembers():assert (dest/m.name).resolve().is_relative_to(dest.resolve()) and not m.issym() and not m.islnk()
   t.extractall(dest)
  for row in read(dest/'PACKAGE_MANIFEST.json'):assert sha(dest/row['path'])==row['sha256']
  blender(run,'inspect_assets.py',[])
  train=['Horse','Pig','Zebra','Cow','Sheep'];scenes=[]
  for i in range(72):
   split='train' if i<48 else 'validation' if i<60 else 'synthetic_test'
   asset=train[(i//6)%5] if i<48 else 'Llama' if i<60 else ['Pug','Fox'][(i-60)//6]
   scenes.append({'id':i,'split':split,'asset':asset,'scenario':['wall','tree','rock','fence','animal_cross','exit'][i%6],'seed':20261019+i,'frames':32})
  save(ROOT/'scenes.json',scenes);save(run.path/'SUMMARY.json',{'assets':read(ROOT/'ASSET_INSPECTION.json'),'scenes':len(scenes),'splits':{'train':48,'validation':12,'synthetic_test':12}})

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['assets','smoke','generate','features','train','evaluate']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.stage=='assets':assets();return
 if a.stage in ['smoke','generate']:
  with Run(a.stage+('_'+str(a.shard) if a.stage=='generate' else ''),{'shard':a.shard,'frames':4 if a.stage=='smoke' else 576,'training_steps':0}) as run:
   blender(run,'generate.py',['--mode',a.stage,'--shard',str(a.shard),'--receipt',str(run.path/'SUMMARY.json')]);return
 if a.stage=='features':
  from .features import main as f
 elif a.stage=='train':
  from .train import main as f
 elif a.stage=='evaluate':
  from .evaluate import main as f
 f()
if __name__=='__main__':main()
