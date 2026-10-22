"""Read the registered Pig rig; executed by pinned Blender in a finite CPU job."""
import hashlib
import json
from pathlib import Path
import sys

import bpy


asset = Path('/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2/assets/farm/Farm Animals by @Quaternius/Blends/Pig.blend')
out = Path(sys.argv[sys.argv.index('--')+1])
bpy.ops.wm.read_factory_settings(use_empty=True)
with bpy.data.libraries.load(str(asset), link=False) as (src, dst):
    dst.objects = src.objects
    dst.actions = src.actions
for obj in dst.objects:
    if obj is not None:
        bpy.context.collection.objects.link(obj)
bpy.context.view_layer.update()
result = {'asset': str(asset), 'sha256': hashlib.sha256(asset.read_bytes()).hexdigest(),
          'blender': bpy.app.version_string, 'objects': [],
          'actions': [{'name': a.name, 'frame_range': list(a.frame_range)} for a in bpy.data.actions]}
for obj in bpy.context.scene.objects:
    row = {'name': obj.name, 'type': obj.type, 'location': list(obj.location),
           'rotation': list(obj.rotation_euler), 'scale': list(obj.scale),
           'matrix_world': [list(r) for r in obj.matrix_world]}
    if obj.type == 'MESH':
        row.update(vertices=len(obj.data.vertices), bounds=[list(p) for p in obj.bound_box],
                   materials=[m.name if m else None for m in obj.data.materials],
                   groups=[g.name for g in obj.vertex_groups])
    if obj.type == 'ARMATURE':
        row['bones'] = [{'name': b.name, 'head': list(b.head_local), 'tail': list(b.tail_local),
                         'parent': b.parent.name if b.parent else None,
                         'deform': b.use_deform,
                         'constraints': [{'name': c.name, 'type': c.type} for c in obj.pose.bones[b.name].constraints]}
                        for b in obj.data.bones]
    result['objects'].append(row)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print('INSPECTED', str(out), result['sha256'], flush=True)
