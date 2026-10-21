"""Finite Blender/Cycles generator with geometry-derived, target-specific labels.

Run with the pinned Blender binary, not an ordinary Python interpreter.
"""
import argparse
import collections
import json
import math
import pathlib
import random
import sys
import time

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=['smoke', 'pilot'], required=True)
    return p.parse_args(sys.argv[sys.argv.index('--') + 1:])


def material(name, color, roughness=0.65):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    return m


def tag(obj, track, mat):
    obj['track_id'] = track
    obj.pass_index = track
    obj.data.materials.append(mat)
    if obj.type == 'MESH':
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    return obj


def ellipsoid(name, xyz, scale, track, mat, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=xyz)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.parent = parent
    return tag(obj, track, mat)


def box(name, xyz, scale, track, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=xyz)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    return tag(obj, track, mat)


def quadruped(track, coat, dark):
    root = bpy.data.objects.new('track_%d_root' % track, None)
    bpy.context.collection.objects.link(root)
    ellipsoid('body', (0, 0, 1.0), (0.66, .30, .36), track, coat, root)
    ellipsoid('neck', (.43, 0, 1.18), (.22, .23, .30), track, coat, root)
    ellipsoid('head', (.66, 0, 1.38), (.27, .21, .25), track, coat, root)
    ellipsoid('muzzle', (.86, 0, 1.30), (.21, .16, .13), track, coat, root)
    for side in [-1, 1]:
        ear = ellipsoid('ear', (.56, side*.17, 1.63), (.10, .055, .20), track, coat, root)
        ear.rotation_euler[0] = side*.22
        ellipsoid('eye', (.75, side*.187, 1.44), (.035, .019, .035), track, dark, root)
    legs = []
    for ix, x in enumerate([-.43, .43]):
        for iy, y in enumerate([-.20, .20]):
            leg = ellipsoid('leg', (x, y, .48), (.10, .09, .45), track, coat, root)
            legs.append((leg, ix+iy))
    tail = ellipsoid('tail', (-.70, 0, 1.10), (.22, .065, .065), track, coat, root)
    tail.rotation_euler[1] = -.5
    return root, legs


def setup_scene(kind, seed):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = seed
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4
    scene.render.resolution_x = 512
    scene.render.resolution_y = 320
    scene.render.resolution_percentage = 100
    scene.render.fps = 12
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    selected = []
    for device in prefs.devices:
        device.use = device.type == 'OPTIX'
        if device.use:
            selected.append({'name':device.name, 'type':device.type, 'id':device.id})
    if len(selected) != 1:
        raise RuntimeError('Expected exactly one Slurm-accessible OptiX GPU: %r' % selected)
    scene.cycles.device = 'GPU'
    scene.world = bpy.data.worlds.new('world')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.38, .48, .63, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .65
    ground = material('ground', (.21, .26, .25))
    wall_mat = material('occluder', (.12, .30, .40))
    dark = material('eyes', (.012, .016, .019))
    coat_a = material('coat_a', (.42, .24, .12))
    coat_b = material('coat_b', (.45, .26, .14))
    box('floor', (0, 0, -.13), (40, 40, .2), 900, ground)
    wall = box('wall', (0, -.8, 1.4), (2.8, .45, 2.8), 100, wall_mat)
    if kind == 'rear_control':
        wall.location.y = 4.0
    elif kind == 'partial_occlusion':
        wall.scale.z = 1.0
        wall.location.z = .5
    elif kind in ['similar_crossing', 'out_of_view']:
        wall.location.x = -12
    a, a_legs = quadruped(1, coat_a, dark)
    b, b_legs = quadruped(2, coat_b, dark)
    bpy.ops.object.light_add(type='AREA', location=(-3, -4, 8))
    light = bpy.context.object
    light.data.energy = 1800
    light.data.shape = 'DISK'
    light.data.size = 6
    light.rotation_euler = (Vector((0, 1, .8))-light.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.ops.object.camera_add(location=(0, -12, 5.7))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 1, .8))-camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = 40
    camera.data.sensor_width = 36
    camera.data.sensor_fit = 'HORIZONTAL'
    scene.camera = camera
    scene['scenario'] = kind
    return scene, a, b, a_legs, b_legs, selected


def update_pose(kind, frame, a, b, a_legs, b_legs):
    t = frame/23
    x = -3.4 + 6.8*t
    a.location = (x, 1.0, 0)
    b.location = (2.8-3*t, 2.8, 0)
    a.rotation_euler.z = 0
    b.rotation_euler.z = math.pi
    if kind == 'similar_crossing':
        b.location = (3.4-6.8*t, -.25, 0)
    elif kind in ['out_of_view', 'mixed_occlusion_exit']:
        a.location.x = -2 + 10.5*t
    for legs, offset in [(a_legs, 0), (b_legs, .5)]:
        for leg, parity in legs:
            leg.rotation_euler.y = .28*math.sin((t*3+offset)*2*math.pi + parity*math.pi)
    bpy.context.view_layer.update()


def geometry_bvhs(scene):
    vertices, faces, owners = [], [], []
    grouped = collections.defaultdict(lambda: ([], []))
    for obj in scene.objects:
        if obj.type != 'MESH' or obj.hide_render:
            continue
        points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
        offset = len(vertices)
        vertices.extend(points)
        track = int(obj.get('track_id', 0))
        for poly in obj.data.polygons:
            faces.append(tuple(offset+i for i in poly.vertices))
            owners.append(track)
        if track in [1, 2]:
            vs, fs = grouped[track]
            base = len(vs)
            vs.extend(points)
            fs.extend(tuple(base+i for i in poly.vertices) for poly in obj.data.polygons)
    all_tree = BVHTree.FromPolygons(vertices, faces)
    individual = {k:(BVHTree.FromPolygons(v, f), v) for k,(v,f) in grouped.items()}
    return all_tree, owners, individual


def camera_rays(scene):
    width, height = 512, 320
    focal = width*scene.camera.data.lens/scene.camera.data.sensor_width
    rotation = scene.camera.matrix_world.to_quaternion()
    rays = []
    for y in range(height*2):
        rays.append([(rotation @ Vector(((x+.5-width)/focal, (height-y-.5)/focal, -1))).normalized()
                     for x in range(width*2)])
    return rays, focal, scene.camera.matrix_world.translation.copy()


def projection_bounds(points, camera, focal):
    inv = camera.matrix_world.inverted()
    coords = [inv @ p for p in points]
    if any(p.z >= -.01 for p in coords):
        raise RuntimeError('Asset crosses camera plane; unsupported in this pilot')
    xs = [512 + focal*p.x/-p.z for p in coords]
    ys = [320 - focal*p.y/-p.z for p in coords]
    return max(0,math.floor(min(xs))-2), min(1024,math.ceil(max(xs))+2), max(0,math.floor(min(ys))-2), min(640,math.ceil(max(ys))+2)


def labels(scene, rays, focal, origin, frame):
    tree, owners, individual = geometry_bvhs(scene)
    visible = np.zeros((320,512), np.uint16)
    distance = np.full((320,512), np.nan, np.float32)
    for y in range(320):
        for x in range(512):
            hit, _, face, dist = tree.ray_cast(origin, rays[y+160][x+256])
            if hit is not None:
                visible[y,x] = owners[face]
                distance[y,x] = dist
    masks, records = {}, []
    for track, (own_tree, points) in individual.items():
        full = np.zeros((640,1024), np.uint8)
        x0,x1,y0,y1 = projection_bounds(points, scene.camera, focal)
        for y in range(y0,y1):
            for x in range(x0,x1):
                if own_tree.ray_cast(origin, rays[y][x])[0] is not None:
                    full[y,x] = 1
        within = full[160:480,256:768]
        is_visible = visible == track
        n_full, n_in, n_vis = int(full.sum()), int(within.sum()), int(is_visible.sum())
        assert not np.any(is_visible & (within == 0)), 'Visible mask outside own isolated projection'
        censored = bool(full[0].any() or full[-1].any() or full[:,0].any() or full[:,-1].any())
        assert not censored, 'Overscan fails to cover full projection'
        n_hidden = n_in-n_vis
        occlusion = n_hidden/n_in if n_in else None
        truncation = 1-n_in/n_full if n_full else None
        if n_in == 0:
            state = 'out_of_view'
        elif n_vis == 0:
            state = 'fully_occluded'
        elif occlusion > .01:
            state = 'partially_occluded'
        else:
            state = 'visible'
        hidden = (within > 0) & ~is_visible
        counts = {str(int(k)):int(v) for k,v in zip(*np.unique(visible[hidden], return_counts=True))}
        assert sum(counts.values()) == n_hidden
        assert not counts.get('0',0), 'Unexplained/background occlusion pixels'
        ys,xs = np.where(is_visible)
        bbox = [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if n_vis else None
        records.append({'frame':frame,'track_id':track,'asset_id':'procedural_quadruped_%d'%track,'state':state,'full_projected_pixels':n_full,'in_frame_pixels':n_in,'visible_pixels':n_vis,'occlusion_fraction':occlusion,'truncation_fraction':truncation,'overscan_censored':censored,'occluder_pixels':counts,'visible_bbox_xyxy':bbox})
        masks['amodal_%d'%track] = within
        masks['overscan_%d'%track] = full
    rng = random.Random(20261019+frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    checks = []
    for _ in range(48):
        x,y = rng.randrange(512), rng.randrange(320)
        hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, origin, rays[y+160][x+256])
        actual = int(obj.get('track_id',0)) if hit else 0
        assert actual == int(visible[y,x]), 'Combined BVH / scene.ray_cast mismatch'
        if hit:
            assert abs((location-origin).length-float(distance[y,x])) < .002
        checks.append({'x':x,'y':y,'track_id':actual})
    return {'instance_id':visible,'range_to_surface_m':distance,**masks}, records, checks


def main():
    args = arguments()
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    scenarios = ['front_occlusion','rear_control','similar_crossing','partial_occlusion','out_of_view','mixed_occlusion_exit']
    if args.mode == 'smoke':
        scenarios = scenarios[:1]
    summary = {'mode':args.mode,'blender_version':bpy.app.version_string,'training_steps':0,'scenes':[],'label_source':'Blender world-space mesh ray intersections at pixel centers; no detector inputs'}
    start = time.time()
    for kind in scenarios:
        scene_dir = output/kind
        (scene_dir/'rgb').mkdir(parents=True)
        (scene_dir/'labels').mkdir()
        scene,a,b,al,bl,devices = setup_scene(kind,20261019)
        scene.frame_set(0)
        bpy.context.view_layer.update()
        rays,focal,origin = camera_rays(scene)
        frames = [0,12] if args.mode == 'smoke' else list(range(24))
        all_records, geometry_checks = [], []
        for frame in frames:
            scene.frame_set(frame)
            update_pose(kind,frame,a,b,al,bl)
            arrays,records,checks = labels(scene,rays,focal,origin,frame)
            np.savez_compressed(scene_dir/'labels'/('%04d.npz'%frame),**arrays)
            all_records.extend(records)
            geometry_checks.append({'frame':frame,'checks':checks})
            scene.render.filepath = str(scene_dir/'rgb'/('%04d.png'%frame))
            bpy.ops.render.render(write_still=True)
            print('FRAME_DONE '+json.dumps({'scene':kind,'frame':frame,'elapsed_s':round(time.time()-start,2)}),flush=True)
        payload = {'scene_id':kind,'seed':20261019,'fps':12,'resolution':[512,320],'frames':frames,'camera_to_world':[list(row) for row in scene.camera.matrix_world],'focal_pixels':focal,'principal_point':[256,160],'coordinate_convention':'image top-left; pixel centers; Blender camera forward -Z and up +Y','tracks':{'1':'procedural_quadruped_1','2':'procedural_quadruped_2','100':'wall','900':'floor'},'records':all_records,'independent_scene_raycast_checks':geometry_checks,'gpu_devices':devices}
        (scene_dir/'annotations.json').write_text(json.dumps(payload,indent=2))
        bpy.ops.wm.save_as_mainfile(filepath=str(scene_dir/'scene.blend'))
        summary['scenes'].append({'scene':kind,'frames':len(frames),'states':dict(collections.Counter(r['state'] for r in all_records)),'geometry_checks':len(frames)*48})
    summary['elapsed_seconds'] = time.time()-start
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    print('RENDER_COMPLETE '+json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
