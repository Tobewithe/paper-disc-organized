"""Blender 4.2: finite articulated pig-pen scenes and native-resolution object IDs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def material(name, color, roughness=.7):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    return m


def textured_skin(name, color, seed):
    m = material(name, color, .62)
    nodes, links = m.node_tree.nodes, m.node_tree.links
    noise = nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 5 + seed % 5
    noise.inputs['Detail'].default_value = 2
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = .25
    ramp.color_ramp.elements[1].position = .8
    ramp.color_ramp.elements[0].color = (*[c*.55 for c in color], 1)
    ramp.color_ramp.elements[1].color = (*color, 1)
    links.new(noise.outputs['Fac'], ramp.inputs[0])
    links.new(ramp.outputs['Color'], nodes['Principled BSDF'].inputs['Base Color'])
    bump = nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = .08
    bump.inputs['Distance'].default_value = .015
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], nodes['Principled BSDF'].inputs['Normal'])
    return m


def cube(name, xyz, dimensions, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=xyz)
    o = bpy.context.object; o.name = name; o.scale = dimensions
    o.data.materials.append(mat); o.pass_index = 0
    return o


def cylinder(name, a, b, radius, mat):
    a, b = Vector(a), Vector(b)
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=(b-a).length, location=(a+b)/2)
    o = bpy.context.object; o.name = name
    o.rotation_euler = (b-a).to_track_quat('Z', 'Y').to_euler()
    o.data.materials.append(mat); o.pass_index = 0
    return o


def evaluated_points(objects):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in objects:
        if obj.type == 'MESH':
            e = obj.evaluated_get(dg); mesh = e.to_mesh()
            points.extend([list(e.matrix_world @ v.co) for v in mesh.vertices])
            e.to_mesh_clear()
    return np.asarray(points, np.float64)


def global_bone_rotation(arm, name, radians, axis='X'):
    bone = arm.pose.bones[name]
    # Express an armature-space rotation in this bone's rest local axes.
    rest = bone.bone.matrix_local.to_quaternion()
    q = Quaternion(Vector({'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}[axis]), radians)
    bone.rotation_mode = 'QUATERNION'
    bone.rotation_quaternion = rest.inverted() @ q @ rest


def load_pig(cfg, recipe, i):
    old = set(bpy.data.objects)
    with bpy.data.libraries.load(cfg['asset'], link=False) as (src, dst):
        dst.objects = src.objects
        dst.actions = src.actions
    for obj in dst.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
    loaded = [o for o in bpy.data.objects if o not in old]
    for obj in list(loaded):
        if obj.type not in {'MESH', 'ARMATURE', 'EMPTY'}:
            bpy.data.objects.remove(obj, do_unlink=True); loaded.remove(obj)
    root = bpy.data.objects.new(f"pig_{i+1:02d}", None)
    bpy.context.collection.objects.link(root)
    raw = bpy.data.objects.new(f'asset_space_{i+1:02d}', None)
    bpy.context.collection.objects.link(raw); raw.parent = root
    for obj in loaded:
        if obj.parent not in loaded:
            obj.parent = raw
        if obj.type == 'MESH':
            obj.pass_index = i+1
            obj['pig_id'] = i+1
            for p in obj.data.polygons:
                p.use_smooth = True
            sub = obj.modifiers.new('render_and_label_smoothing', 'SUBSURF')
            sub.levels = 1; sub.render_levels = 1
    arm = next(o for o in loaded if o.type == 'ARMATURE')
    pose = recipe['poses'][i]
    if pose == 'walking':
        action = next(a for a in dst.actions if a.name.split('.')[0] == 'Walk')
        arm.animation_data_create()
        for track in arm.animation_data.nla_tracks:
            track.mute = True
        arm.animation_data.action = action
        bpy.context.scene.frame_set(recipe['pose_frame'][i])
        bpy.context.view_layer.update()
        # Bake the sampled walk pose, retaining the rig's IK for this still.
        arm.animation_data_clear()
    else:
        arm.animation_data_clear()
        for b in arm.pose.bones:
            b.matrix_basis = Matrix.Identity(4)
            for c in b.constraints:
                c.mute = True
        if pose == 'prone':
            for side in ['L', 'R']:
                global_bone_rotation(arm, 'FrontUpLeg.'+side, 1.30)
                global_bone_rotation(arm, 'FrontLowLeg.'+side, 1.20)
                global_bone_rotation(arm, 'BackUpLeg.'+side, -1.48)
                global_bone_rotation(arm, 'BackLowLeg.'+side, -1.15)
            global_bone_rotation(arm, 'Neck', .08)
        elif pose.startswith('lateral'):
            for side in ['L', 'R']:
                global_bone_rotation(arm, 'FrontUpLeg.'+side, .45 if side == 'L' else .60)
                global_bone_rotation(arm, 'FrontLowLeg.'+side, .45)
                global_bone_rotation(arm, 'BackUpLeg.'+side, -.55)
                global_bone_rotation(arm, 'BackLowLeg.'+side, -.4)
            global_bone_rotation(arm, 'Neck', .12)
            global_bone_rotation(arm, 'Head', .06)
        elif pose == 'feeding':
            global_bone_rotation(arm, 'Neck', .26)
            global_bone_rotation(arm, 'Head', .18)
    # Centre the source in its native XY axes; source is Y-long and Z-up.
    points = evaluated_points(loaded)
    lo, hi = points.min(0), points.max(0)
    raw.location = (-float((lo[0]+hi[0])/2), -float((lo[1]+hi[1])/2), 0)
    factor = recipe['body_lengths_m'][i] / (hi[1]-lo[1])
    rng = random.Random(recipe['seed'] + 73*i)
    root.scale = (factor*rng.uniform(.92, 1.10), factor, factor*rng.uniform(.94, 1.06))
    if pose.startswith('lateral'):
        root.rotation_euler.y = (1 if pose == 'lateral_left' else -1)*rng.uniform(1.40, 1.55)
    # Ground the evaluated articulated mesh, never the unposed rest bbox.
    p = evaluated_points(loaded)
    root.location.z -= float(p[:, 2].min())
    p = evaluated_points(loaded)
    assert abs(float(p[:, 2].min())) < cfg['floor_contact_tolerance_m']
    return root, loaded, {'pig_id': i+1, 'pose': pose, 'body_length_parameter_m': recipe['body_lengths_m'][i],
                          'pose_source': 'sampled_Walk' if pose == 'walking' else 'procedural_rest_bone_pose',
                          'death_animation_used': False}


def hull(points):
    pts = sorted(set(map(tuple, points)))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1])


def separated(a, b, gap=.012):
    for poly in [a, b]:
        edges = np.roll(poly, -1, axis=0)-poly
        for e in edges:
            axis = np.array([-e[1], e[0]]) / max(np.linalg.norm(e), 1e-12)
            pa, pb = a@axis, b@axis
            if pa.max()+gap <= pb.min() or pb.max()+gap <= pa.min():
                return True
    return False


def place_pigs(cfg, recipe):
    rng = random.Random(recipe['seed']); px, py = recipe['pen_dimensions_m']
    pigs, footprints, rejection_counts = [], [], []
    prepared = []
    for i in range(recipe['requested_pigs']):
        root, objects, meta = load_pig(cfg, recipe, i)
        root.rotation_euler.z = 0
        base = hull(evaluated_points(objects)[:, :2])
        area = abs(float(np.dot(base[:, 0], np.roll(base[:, 1], -1))-np.dot(base[:, 1], np.roll(base[:, 0], -1))))/2
        prepared.append((area, i, root, objects, meta, base))
    # Large resting footprints go first; a small early pig must not fragment all free space.
    prepared.sort(key=lambda z: (-z[0], z[1]))
    for area, i, root, objects, meta, base in prepared:
        accepted = False
        for attempt in range(12000):
            if recipe['layout'] == 'rest_rows':
                yaw = rng.choice([0, math.pi]) + rng.uniform(-.10, .10)
                if attempt < 1200:
                    x = rng.choice(np.linspace(-px/2+.65, px/2-.65, 8).tolist()) + rng.uniform(-.06, .06)
                    y = rng.choice(np.linspace(-py/2+1.45, py/2-1.0, 3).tolist()) + rng.uniform(-.10, .10)
                else:
                    x, y = rng.uniform(-px/2+.4, px/2-.4), rng.uniform(-py/2+.8, py/2-.3)
            else:
                yaw = rng.uniform(-math.pi, math.pi)
                x, y = rng.uniform(-px/2+.35, px/2-.35), rng.uniform(-py/2+.45, py/2-.3)
                if recipe['layout'] == 'feeder_cluster' and i < max(2, recipe['requested_pigs']//3) and attempt < 1200:
                    yaw = rng.uniform(-.2, .2)
                    x, y = rng.uniform(-2.6, 2.6), rng.uniform(-1.9, -.9)
                elif recipe['layout'] == 'fence_strip' and i < recipe['requested_pigs']//2 and attempt < 1200:
                    yaw = rng.choice([0, math.pi])+rng.uniform(-.25, .25)
                    x = rng.choice([-1, 1])*rng.uniform(2.4, 3.1)
            rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            polygon = base@rot.T + [x, y]
            lo, hi = polygon.min(0), polygon.max(0)
            # Reserve the feeder band and the drinker; no animal-geometry overlap.
            if lo[0] < -px/2+.10 or hi[0] > px/2-.10 or lo[1] < -py/2+.65 or hi[1] > py/2-.12:
                continue
            drinker = np.asarray([[px/2-.58, .05], [px/2-.10, .05], [px/2-.10, .75], [px/2-.58, .75]])
            if not separated(polygon, drinker):
                continue
            if all(separated(polygon, other) for other in footprints):
                root.rotation_euler.z = yaw
                root.location.x, root.location.y = x, y
                # A yaw leaves grounding unchanged.
                points = evaluated_points(objects)
                assert abs(float(points[:, 2].min())) < cfg['floor_contact_tolerance_m']
                meta.update(xyz=list(root.location), yaw_radians=yaw, scale=list(root.scale),
                            evaluated_bounds_m=[points.min(0).tolist(), points.max(0).tolist()],
                            footprint_xy=polygon.tolist(), contact_min_z_m=float(points[:, 2].min()),
                            placement_rejections=attempt)
                footprints.append(polygon); pigs.append((objects, meta)); rejection_counts.append(attempt)
                accepted = True
                break
        if not accepted:
            raise RuntimeError(f"Cannot place requested pig {i+1}/{recipe['requested_pigs']} without intersection; {recipe['geometry_id']}; footprint={np.ptp(base, axis=0).tolist()}")
    pigs.sort(key=lambda z: z[1]['pig_id'])
    return pigs


def build_pen(recipe):
    px, py = recipe['pen_dimensions_m']
    gray = material('concrete', (.28, .29, .28), .9)
    nodes, links = gray.node_tree.nodes, gray.node_tree.links
    noise = nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 7
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (.12, .13, .12, 1)
    ramp.color_ramp.elements[1].color = (.44, .43, .38, 1)
    if recipe['floor_type'] == 'bedded':
        ramp.color_ramp.elements[0].color = (.18, .12, .055, 1)
        ramp.color_ramp.elements[1].color = (.50, .39, .20, 1)
        noise.inputs['Scale'].default_value = 45
    elif recipe['floor_type'] == 'stained_concrete':
        ramp.color_ramp.elements[0].color = (.08, .07, .04, 1)
    links.new(noise.outputs['Fac'], ramp.inputs[0]); links.new(ramp.outputs['Color'], nodes['Principled BSDF'].inputs['Base Color'])
    cube('floor', (0, 0, -.075), (30, 30, .15), gray)
    metal = material('galvanized_fence', (.28, .31, .32), .44)
    metal.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = .65
    wall = material('pen_wall', (.47, .46, .40), .86)
    if recipe['floor_type'] == 'slatted':
        dark = material('slat_grooves', (.065, .07, .065), .95)
        for x in np.arange(-px/2, px/2, .22):
            cube('slat_groove', (float(x), 0, .0002), (.025, py, .0003), dark)
    for y in [-py/2, py/2]:
        if recipe['fence_type'] == 'low_wall':
            cube('low_wall', (0, y, .30), (px+.1, .12, .60), wall)
        for x in np.linspace(-px/2, px/2, 10):
            cylinder('fence_post', (float(x), y, 0), (float(x), y, 1.1), .035, metal)
        for z in [.38, .75, 1.05]:
            cylinder('horizontal_rail', (-px/2, y, z), (px/2, y, z), .027, metal)
        if recipe['fence_type'] == 'vertical_bars':
            for x in np.arange(-px/2, px/2, .25):
                cylinder('vertical_bar', (float(x), y, .05), (float(x), y, 1.05), .017, metal)
    for x in [-px/2, px/2]:
        cube('side_wall', (x, 0, .18), (.10, py, .36), wall)
        for y in np.linspace(-py/2, py/2, 8):
            cylinder('side_post', (x, float(y), 0), (x, float(y), 1.1), .035, metal)
        for z in [.70, 1.05]:
            cylinder('side_rail', (x, -py/2, z), (x, py/2, z), .027, metal)
    # Open-topped trough; no closed box pretending to be a feeder.
    fy = -py/2+.28
    cube('feeder_base', (0, fy, .20), (5.8, .46, .10), metal)
    for dy in [-.23, .23]:
        cube('feeder_lip', (0, fy+dy, .35), (5.8, .045, .30), metal)
    food = material('feed', (.28, .20, .095), .95)
    cube('feed_surface', (0, fy, .27), (5.7, .36, .05), food)
    for x in np.linspace(-2.9, 2.9, 8):
        cylinder('feeder_separator', (float(x), fy-.22, .42), (float(x), fy+.22, .42), .014, metal)
    cylinder('drinker_pipe', (px/2-.25, .4, .18), (px/2-.25, .4, 1.0), .025, metal)
    cube('drinker_bowl', (px/2-.28, .4, .22), (.40, .40, .12), metal)


def setup(cfg, recipe):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = cfg['render_samples']; scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = False; scene.cycles.seed = recipe['seed']
    scene.render.threads_mode = 'FIXED'; scene.render.threads = 4
    scene.render.resolution_x, scene.render.resolution_y = cfg['resolution']
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'; scene.render.image_settings.color_mode = 'RGB'
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = 'AgX'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'; prefs.get_devices()
    selected = []
    for device in prefs.devices:
        device.use = device.type == 'OPTIX'
        if device.use:
            selected.append(device.name)
    assert len(selected) == 1, selected
    scene.cycles.device = 'GPU'
    scene.world = bpy.data.worlds.new('indoor_ambient'); scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.5, .52, .55, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .30
    build_pen(recipe)
    pigs = place_pigs(cfg, recipe)
    px, py = recipe['pen_dimensions_m']
    elev, az = math.radians(recipe['camera_elevation_degrees']), math.radians(recipe['camera_azimuth_degrees'])
    bpy.ops.object.camera_add()
    camera = bpy.context.object; camera.data.lens = recipe['lens_mm']
    camera.data.sensor_width = 36; camera.data.sensor_fit = 'HORIZONTAL'
    target = Vector((0, 0, .32))
    # Fit all pen corners, then permit a small deterministic crop/zoom range.
    distance = 10.0
    camera.location = target+Vector((math.cos(az)*math.cos(elev), math.sin(az)*math.cos(elev), math.sin(elev)))*distance
    camera.rotation_euler = (target-camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera; bpy.context.view_layer.update()
    from bpy_extras.object_utils import world_to_camera_view
    corners = [Vector((x, y, z)) for x in [-px/2, px/2] for y in [-py/2, py/2] for z in [0, 1]]
    for _ in range(12):
        uv = [world_to_camera_view(scene, camera, p) for p in corners]
        scale = max(max(abs(p.x-.5)*2, abs(p.y-.5)*2) for p in uv)
        distance *= scale/.95
        camera.location = target+Vector((math.cos(az)*math.cos(elev), math.sin(az)*math.cos(elev), math.sin(elev)))*distance
        bpy.context.view_layer.update()
        if abs(scale-.95) < .008:
            break
    # V4 fastpilot: retain pen context while matching larger real-pig views.
    framing = recipe.get('framing', 'full_pen')
    if framing != 'full_pen':
        centres = [Vector(p[1]['xyz']) for p in pigs]
        if framing == 'contact_close' and len(centres) >= 2:
            pair = min(((a, b) for a in range(len(centres)) for b in range(a+1, len(centres))),
                       key=lambda ab: (centres[ab[0]]-centres[ab[1]]).length)
            target = (centres[pair[0]]+centres[pair[1]])/2
            target.z = .35
        elif centres:
            target = sum(centres, Vector())/len(centres); target.z = .35
    distance *= recipe.get('framing_distance_factor', 1.0)
    camera.location = target+Vector((math.cos(az)*math.cos(elev), math.sin(az)*math.cos(elev), math.sin(elev)))*distance
    camera.rotation_euler = (target-camera.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()
    scene.view_layers[0].use_pass_object_index = True
    scene.use_nodes = True; nodes = scene.node_tree.nodes; nodes.clear()
    render = nodes.new('CompositorNodeRLayers')
    output = nodes.new('CompositorNodeOutputFile')
    output.format.file_format = 'OPEN_EXR'; output.format.color_mode = 'RGB'; output.format.color_depth = '32'
    scene.node_tree.links.new(render.outputs['IndexOB'], output.inputs[0])
    bpy.context.scene.frame_set(1); bpy.context.view_layer.update()
    return scene, pigs, output, selected


def set_appearance(scene, pigs, recipe, style):
    rng = random.Random(style['seed'])
    palette = [(.76, .48, .42), (.68, .55, .48), (.82, .63, .56), (.32, .28, .25), (.58, .39, .31)]
    for objects, meta in pigs:
        color = palette[rng.randrange(len(palette))]
        color = tuple(min(.95, c*rng.uniform(.88, 1.1)) for c in color)
        skin = textured_skin(f"skin_{meta['pig_id']}_{style['id']}", color, style['seed']+meta['pig_id'])
        for obj in objects:
            if obj.type != 'MESH':
                continue
            for j, mat in enumerate(obj.data.materials):
                if mat is None:
                    continue
                # Retain dark eyes/hooves; replace only the original skin slot.
                base = mat.node_tree.nodes.get('Principled BSDF') if mat.use_nodes else None
                color0 = base.inputs['Base Color'].default_value if base else mat.diffuse_color
                if max(color0[:3]) > .15 and ('skin_' in mat.name or j == 0):
                    obj.data.materials[j] = skin
    for obj in list(scene.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    kind = style['light_type']
    if kind == 'side_window':
        positions = [(-4, -1, 5), (3, 2, 6)]
        energies = [1800, 450]
    elif kind == 'dim_indoor':
        positions = [(-2, 0, 4.5), (2, 1, 4.5)]
        energies = [500, 600]
    else:
        positions = [(-2, -1, 5), (2, 1, 5)]
        energies = [1200, 1400]
    for xyz, energy in zip(positions, energies):
        bpy.ops.object.light_add(type='AREA', location=xyz)
        obj = bpy.context.object; obj.data.energy = energy*rng.uniform(.85, 1.15)
        obj.data.size = rng.uniform(3, 5)
        obj.data.color = (1, rng.uniform(.87, 1), rng.uniform(.77, 1))
        obj.rotation_euler = (Vector((0, 0, .4))-obj.location).to_track_quat('-Z', 'Y').to_euler()
    scene.view_settings.exposure = rng.uniform(-.25, .25)
    bpy.context.view_layer.update()


def read_ids(path, width, height):
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.empty(width*height*4, np.float32)
    image.pixels.foreach_get(pixels)
    values = pixels.reshape(height, width, 4)[::-1, :, 0].copy()
    bpy.data.images.remove(image)
    assert np.isfinite(values).all()
    assert np.max(np.abs(values-np.rint(values))) < 1e-4
    return np.rint(values).astype(np.uint16)


def ray_check(scene, ids, cfg, seed):
    vertices, triangles, owners = [], [], []
    dg = bpy.context.evaluated_depsgraph_get()
    for obj in scene.objects:
        if obj.type != 'MESH' or obj.hide_render:
            continue
        e = obj.evaluated_get(dg); mesh = e.to_mesh(); mesh.calc_loop_triangles()
        offset = len(vertices)
        vertices.extend([e.matrix_world @ v.co for v in mesh.vertices])
        triangles.extend([tuple(offset+j for j in t.vertices) for t in mesh.loop_triangles])
        owners.extend([obj.pass_index]*len(mesh.loop_triangles)); e.to_mesh_clear()
    tree = BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
    width, height = cfg['resolution']; focal = width*scene.camera.data.lens/36
    origin = scene.camera.matrix_world.translation
    rot = scene.camera.matrix_world.to_quaternion()
    rng = np.random.default_rng(seed)
    locs = rng.choice(width*height, cfg['ray_check_samples_per_geometry'], replace=False)
    mismatches = []
    for ix in locs:
        y, x = divmod(int(ix), width)
        direction = rot @ Vector(((x+.5-width/2)/focal, (height/2-y-.5)/focal, -1))
        hit, _, face, _ = tree.ray_cast(origin, direction.normalized())
        expected = owners[face] if hit is not None else 0
        if expected != int(ids[y, x]):
            mismatches.append([x, y, expected, int(ids[y, x])])
    fraction = len(mismatches)/len(locs)
    assert fraction <= cfg['maximum_ray_id_mismatch_fraction'], ('ray_ID_disagreement', fraction, mismatches[:12])
    return {'samples': len(locs), 'mismatches': len(mismatches), 'mismatch_fraction': fraction,
            'first_mismatches': mismatches[:12], 'coordinate_contract': 'native pixel centres, horizontal sensor fit'}


def render_one(cfg, recipe, output_root):
    start = time.monotonic()
    folder = output_root/recipe['geometry_id']; folder.mkdir(parents=True, exist_ok=False)
    scene, pigs, compositor, devices = setup(cfg, recipe)
    reference_ids = None
    for style in recipe['appearances']:
        set_appearance(scene, pigs, recipe, style)
        compositor.base_path = str(folder)
        compositor.file_slots[0].path = f"id_{style['id']}_"
        scene.render.filepath = str(folder/f"rgb_{style['id']}.png")
        bpy.ops.render.render(write_still=True)
        exr = next(folder.glob(f"id_{style['id']}_*.exr"))
        ids = read_ids(exr, *cfg['resolution'])
        assert int(ids.max()) <= recipe['requested_pigs']
        if reference_ids is None:
            reference_ids = ids
            check = ray_check(scene, ids, cfg, recipe['seed'])
            np.savez_compressed(folder/'labels.npz', instance_id=ids)
        else:
            assert np.array_equal(reference_ids, ids), 'appearance changed instance geometry'
        exr.unlink()
    counts = {str(i): int((reference_ids == i).sum()) for i in range(1, recipe['requested_pigs']+1)}
    meta = {'recipe': recipe, 'blender_version': bpy.app.version_string, 'gpu': devices,
            'pigs': [meta for _, meta in pigs], 'visible_pixels_per_pig': counts,
            'ray_check': check, 'paired_instance_ids_equal': True,
            'camera_matrix_world': [list(r) for r in scene.camera.matrix_world],
            'camera_lens_mm': scene.camera.data.lens, 'camera_sensor_width_mm': 36,
            'all_pig_xy_footprints_disjoint': True, 'source_asset_sha256': cfg['asset_sha256'],
            'elapsed_seconds': time.monotonic()-start,
            'limits': ['single stylized Pig mesh', 'rest poses procedural and require visual QA', 'not a proven real-domain improvement']}
    dump(folder/'scene.json', meta)
    print('SCENE_COMPLETE', recipe['geometry_id'], dict(Counter(recipe['poses'])), meta['elapsed_seconds'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True); parser.add_argument('--run', required=True)
    parser.add_argument('--stage', choices=['smoke', 'formal'], required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    root, run = Path(args.root), Path(args.run)
    cfg = json.loads((root/'CONFIG.json').read_text())
    assert hashlib.sha256(Path(cfg['asset']).read_bytes()).hexdigest() == cfg['asset_sha256']
    plan = json.loads((root/'SCENES.json').read_text())
    recipes = plan['smoke' if args.stage == 'smoke' else 'formal']
    output = run/'scenes'; output.mkdir(parents=True, exist_ok=True)
    for recipe in recipes:
        render_one(cfg, recipe, output)
    print('RENDER_COMPLETE', len(recipes), flush=True)


if __name__ == '__main__':
    main()
