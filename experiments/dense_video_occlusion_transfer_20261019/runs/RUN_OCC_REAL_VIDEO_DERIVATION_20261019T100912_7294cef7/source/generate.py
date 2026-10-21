import argparse,collections,json,math,pathlib,random,sys,time
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=pathlib.Path('/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2')
W,H=192,128

def mat(name,color):
 m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
 m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(*color,1)
 m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value=.8
 return m
def cube(xyz,size,identity,m):
 bpy.ops.mesh.primitive_cube_add(size=1,location=xyz);o=bpy.context.object;o.scale=size;o['track_id']=identity;o.data.materials.append(m);return o
def load_animal(name,identity,rng):
 old=set(bpy.data.objects);old_actions=set(bpy.data.actions)
 if name=='Fox':bpy.ops.import_scene.gltf(filepath=str(R/'assets/Fox.glb'))
 else:
  p=next((R/'assets').rglob(name+'.blend'))
  with bpy.data.libraries.load(str(p),link=False) as (a,b):b.objects=a.objects;b.actions=a.actions
  for o in b.objects:
   if o is not None:bpy.context.collection.objects.link(o)
 loaded=[o for o in bpy.data.objects if o not in old]
 for o in list(loaded):
  if o.type not in ['MESH','ARMATURE','EMPTY']:
   bpy.data.objects.remove(o,do_unlink=True);loaded.remove(o)
 root=bpy.data.objects.new('individual_%d'%identity,None);bpy.context.collection.objects.link(root)
 raw=bpy.data.objects.new('asset_coordinates_%d'%identity,None);bpy.context.collection.objects.link(raw);raw.parent=root
 for o in loaded:
  if o.parent not in loaded:o.parent=raw
  if o.type=='MESH':
   o['track_id']=identity
   for slot in o.material_slots:
    if slot.material:
     m=slot.material.copy();slot.material=m
     if not m.use_nodes:
      color=list(m.diffuse_color);m.use_nodes=True;m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=color
     bs=m.node_tree.nodes.get('Principled BSDF')
     if bs:
      color=list(bs.inputs['Base Color'].default_value)
      if not bs.inputs['Base Color'].is_linked:bs.inputs['Base Color'].default_value=tuple(min(1,c*rng.uniform(.85,1.12)) for c in color[:3])+(1,)
      bs.inputs['Roughness'].default_value=.75
 acts=[a for a in bpy.data.actions if a not in old_actions]
 action=next((a for a in acts if 'walk' in a.name.lower()),next((a for a in acts if 'run' in a.name.lower()),acts[0] if acts else None))
 for o in loaded:
  if o.type=='ARMATURE' and action:
   o.animation_data_create()
   for t in o.animation_data.nla_tracks:t.mute=True
   o.animation_data.action=action
   for fc in action.fcurves:
    if not any(m.type=='CYCLES' for m in fc.modifiers):fc.modifiers.new('CYCLES')
 bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get();pts=[]
 for o in loaded:
  if o.type=='MESH':
   e=o.evaluated_get(dg);mesh=e.to_mesh();pts += [e.matrix_world@v.co for v in mesh.vertices];e.to_mesh_clear()
 q=np.asarray([list(p) for p in pts]);lo=q.min(0);hi=q.max(0);size=hi-lo
 assert max(size[:2])>0 and size[2]>0
 scale=2/max(size[:2]);raw.location=(-float((lo[0]+hi[0])/2),-float((lo[1]+hi[1])/2),-float(lo[2]))
 root.scale=(scale,)*3
 angle=math.pi/2 if size[1]>size[0] else 0
 return root,angle,{'asset':name,'action':action.name if action else None,'source_extent':size.tolist(),'scale':scale}

def setup(cfg):
 bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene;rng=random.Random(cfg['seed'])
 s.render.engine='CYCLES';s.cycles.samples=8;s.cycles.use_denoising=True;s.cycles.use_adaptive_sampling=False;s.cycles.seed=cfg['seed']
 s.render.resolution_x=384;s.render.resolution_y=256;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB';s.render.threads_mode='FIXED';s.render.threads=4;s.render.use_persistent_data=True
 prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
 devices=[]
 for d in prefs.devices:
  d.use=d.type=='OPTIX'
  if d.use:devices.append(d.name)
 assert len(devices)==1,devices
 s.cycles.device='GPU';s.world=bpy.data.worlds.new('environment');s.world.use_nodes=True;n=s.world.node_tree.nodes;l=s.world.node_tree.links
 env=n.new('ShaderNodeTexEnvironment');env.image=bpy.data.images.load(str(R/'assets'/(['spruit_sunrise.hdr','kloofendal_48d_partly_cloudy_puresky.hdr'][cfg['id']%2])));l.new(env.outputs['Color'],n['Background'].inputs['Color']);n['Background'].inputs['Strength'].default_value=rng.uniform(.6,1.4)
 ground=mat('ground',(.25,.28,.17));nodes=ground.node_tree.nodes;links=ground.node_tree.links
 tx=nodes.new('ShaderNodeTexImage');tx.image=bpy.data.images.load(str(R/'assets'/(['brown_mud_leaves_01.jpg','concrete_floor_02.jpg'][cfg['id']%2])))
 coord=nodes.new('ShaderNodeTexCoord');mapping=nodes.new('ShaderNodeVectorMath');mapping.operation='SCALE';mapping.inputs[3].default_value=12;links.new(coord.outputs['Generated'],mapping.inputs[0]);links.new(mapping.outputs[0],tx.inputs['Vector']);links.new(tx.outputs['Color'],nodes['Principled BSDF'].inputs['Base Color'])
 cube((0,0,-.12),(60,60,.2),900,ground)
 obstacle=mat('occluder',(.17+rng.random()*.25,.16+rng.random()*.2,.12))
 kind=cfg['scenario']
 if kind=='wall':cube((0,-.7,1.6),(2.4,.5,3.2),100,obstacle)
 elif kind=='tree':
  bpy.ops.mesh.primitive_cylinder_add(vertices=20,radius=.75,depth=5,location=(0,-.5,2.5));o=bpy.context.object;o['track_id']=100;o.data.materials.append(obstacle)
  for x in [-1,0,1]:
   bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=1.6,location=(x,-.3,5));o=bpy.context.object;o['track_id']=100;o.data.materials.append(mat('leaves',(.08,.2+rng.random()*.15,.04)))
 elif kind=='rock':
  for x in [-.8,0,.8]:
   bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=1.1,location=(x,-.6,.7));o=bpy.context.object;o.scale=(1,.8,1);o['track_id']=100;o.data.materials.append(obstacle)
 elif kind=='fence':
  for x in np.linspace(-2,2,7):cube((float(x),-.5,.9),(.19,.22,1.8),100,obstacle)
  for z in [.5,1.2]:cube((0,-.5,z),(4.5,.18,.15),100,obstacle)
 roots=[];assets=[]
 for k in range(1,5):
  o,angle,info=load_animal(cfg['asset'],k,rng);roots.append((o,angle));assets.append(info)
 angle=rng.uniform(-.12,.12);radius=rng.uniform(11.5,13.0)
 bpy.ops.object.camera_add(location=(math.sin(angle)*radius,-math.cos(angle)*radius,rng.uniform(4.5,6.5)))
 camera=bpy.context.object;camera.rotation_euler=(Vector((0,1,1))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.lens=40;camera.data.sensor_width=36;camera.data.sensor_fit='HORIZONTAL';s.camera=camera
 bpy.context.view_layer.update()
 return s,roots,assets,devices

def pose(s,roots,cfg,frame):
 s.frame_set(frame*2)
 t=frame/31
 for k,(o,angle) in enumerate(roots):
  direction=1 if k%2==0 else -1
  x=direction*(-3.8+7.6*t)+(.2 if k>1 else 0)
  y=[.8,2.0,-.5,3.1][k]
  if cfg['scenario']=='exit' and k==0:x=-3+12*t
  o.location=(x,y,0);o.rotation_euler.z=angle+(0 if direction==1 else math.pi)
 bpy.context.view_layer.update()

def trees(s):
 dg=bpy.context.evaluated_depsgraph_get();verts=[];faces=[];owners=[];group=collections.defaultdict(lambda:([],[]))
 for obj in s.objects:
  if obj.type!='MESH' or obj.hide_render:continue
  e=obj.evaluated_get(dg);mesh=e.to_mesh();points=[e.matrix_world@v.co for v in mesh.vertices];identity=int(obj.get('track_id',0));offset=len(verts);verts.extend(points)
  for p in mesh.polygons:faces.append(tuple(offset+i for i in p.vertices));owners.append(identity)
  if 1<=identity<=4:
   v,f=group[identity];base=len(v);v.extend(points);f.extend(tuple(base+i for i in p.vertices) for p in mesh.polygons)
  e.to_mesh_clear()
 return BVHTree.FromPolygons(verts,faces),owners,{k:(BVHTree.FromPolygons(v,f),v) for k,(v,f) in group.items()}

def raygrid(s):
 focal=W*s.camera.data.lens/s.camera.data.sensor_width;rot=s.camera.matrix_world.to_quaternion()
 rays=[[(rot@Vector(((x+.5-W*1.5)/focal,(H*1.5-y-.5)/focal,-1))).normalized() for x in range(W*3)] for y in range(H*3)]
 return rays,focal,s.camera.matrix_world.translation.copy()

def geometry(s,rays,focal,origin,frame):
 tree,owners,own=trees(s);ids=np.zeros((H,W),np.uint16)
 for y in range(H):
  for x in range(W):
   hit,_,face,_=tree.ray_cast(origin,rays[y+H][x+W])
   if hit is not None:ids[y,x]=owners[face]
 rows=[];masks={};inv=s.camera.matrix_world.inverted()
 for track,(bt,pts) in own.items():
  q=[inv@p for p in pts];assert all(p.z<0 for p in q)
  xx=[W*1.5+focal*p.x/-p.z for p in q];yy=[H*1.5-focal*p.y/-p.z for p in q]
  full=np.zeros((H*3,W*3),np.uint8)
  for y in range(max(0,math.floor(min(yy))-1),min(H*3,math.ceil(max(yy))+1)):
   for x in range(max(0,math.floor(min(xx))-1),min(W*3,math.ceil(max(xx))+1)):
    if bt.ray_cast(origin,rays[y][x])[0] is not None:full[y,x]=1
  amodal=full[H:2*H,W:2*W];visible=ids==track;v=int(visible.sum());a=int(amodal.sum());af=int(full.sum());assert af>0
  assert not np.any(visible&(amodal==0));assert not(full[0].any() or full[-1].any() or full[:,0].any() or full[:,-1].any())
  visibility=v/a if a else 0.;state=3 if a==0 else 2 if v==0 else 1 if visibility<.85 else 0
  ys,xs=np.where(visible);bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if v else None
  rows.append({'frame':frame,'track_id':track,'state':state,'visibility':visibility,'truncation':1-a/af,'visible_pixels':v,'in_frame_pixels':a,'full_pixels':af,'box':bbox})
  masks['amodal_'+str(track)]=amodal
 dg=bpy.context.evaluated_depsgraph_get();rng=random.Random(20261019+frame)
 for _ in range(16):
  x,y=rng.randrange(W),rng.randrange(H);hit,_,_,_,obj,_=s.ray_cast(dg,origin,rays[y+H][x+W]);assert (int(obj.get('track_id',0)) if hit else 0)==int(ids[y,x])
 return {'instance_id':ids,**masks},rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode');p.add_argument('--shard',type=int);p.add_argument('--receipt');a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);scenes=json.loads((R/'scenes.json').read_text());start=time.time();results=[]
 selected=scenes[:1] if a.mode=='smoke' else [x for x in scenes if x['id']%4==a.shard]
 for cfg in selected:
  out=R/('smoke_data' if a.mode=='smoke' else 'synthetic')/('%03d'%cfg['id']);(out/'rgb').mkdir(parents=True,exist_ok=False);(out/'labels').mkdir()
  s,roots,assets,devices=setup(cfg);pose(s,roots,cfg,0);rays,focal,origin=raygrid(s);allrows=[]
  frames=[0,10,16,31] if a.mode=='smoke' else range(32)
  for f in frames:
   pose(s,roots,cfg,f);arrays,rows=geometry(s,rays,focal,origin,f);allrows+=rows;np.savez_compressed(out/'labels'/('%03d.npz'%f),**arrays)
   s.render.filepath=str(out/'rgb'/('%03d.png'%f));bpy.ops.render.render(write_still=True)
  record={**cfg,'records':allrows,'assets':assets,'devices':devices,'label_size':[W,H],'rgb_size':[384,256]};(out/'annotations.json').write_text(json.dumps(record));results.append({'scene':cfg['id'],'states':dict(collections.Counter(r['state'] for r in allrows))});print('SCENE_DONE',cfg['id'],round(time.time()-start,1),flush=True)
 pathlib.Path(a.receipt).write_text(json.dumps({'scenes':results,'elapsed_s':time.time()-start,'training_steps':0},indent=2))
if __name__=='__main__':main()
