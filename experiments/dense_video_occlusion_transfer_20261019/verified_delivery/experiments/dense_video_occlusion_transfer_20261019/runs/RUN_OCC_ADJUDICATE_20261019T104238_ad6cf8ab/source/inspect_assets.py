import bpy,json,pathlib,sys
root=pathlib.Path('/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2')
out=[]
for path in sorted((root/'assets').rglob('*.blend')):
 bpy.ops.wm.read_factory_settings(use_empty=True)
 with bpy.data.libraries.load(str(path),link=False) as (src,dst):dst.objects=src.objects;dst.actions=src.actions
 for o in dst.objects:
  if o is not None:bpy.context.collection.objects.link(o)
 bpy.context.view_layer.update()
 out.append({'asset':path.stem,'path':str(path.relative_to(root/'assets')),'meshes':[(o.name,len(o.data.vertices)) for o in bpy.context.scene.objects if o.type=='MESH'],'armatures':[(o.name,len(o.data.bones),o.animation_data.action.name if o.animation_data and o.animation_data.action else None) for o in bpy.context.scene.objects if o.type=='ARMATURE'],'actions':[(a.name,list(a.frame_range)) for a in bpy.data.actions]})
(root/'ASSET_INSPECTION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
