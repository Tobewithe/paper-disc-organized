"""Blender: deforming surface correspondences, explicit occluders, paired videos.

RGB 512x320; raster labels 128x80 use an explicit 4x coordinate map.
Point labels are continuous, full-precision ray tests, not raster lookups.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

import bpy
import bmesh
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np

OLD = Path("/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2")
RGB_W, RGB_H, W, H = 512, 320, 128, 80
T, FPS, POINTS = 48, 12, 8


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def legacy():
    path = OLD / "shared/dense_video_occlusion/generate.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "9c5f3a3b9f594f6abf89bb9bf3278ce8062d4379af9b87b71d70d0d2148f1622")
    spec = importlib.util.spec_from_file_location("original_asset_loader", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup(cfg, variant):
    old = legacy()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.samples = 8
    s.cycles.use_denoising = True
    s.cycles.use_adaptive_sampling = False
    s.cycles.seed = cfg["seed"]
    s.cycles.device = "GPU"
    s.render.resolution_x, s.render.resolution_y = RGB_W, RGB_H
    s.render.resolution_percentage = 100
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGB"
    s.render.threads_mode = "FIXED"
    s.render.threads = 4
    s.render.use_persistent_data = True
    s.render.use_motion_blur = False
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    selected = []
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
        if d.use:
            selected.append(d.name)
    assert len(selected) == 1, selected
    rng = random.Random(cfg["seed"])
    s.world = bpy.data.worlds.new("world")
    s.world.use_nodes = True
    s.world.node_tree.nodes["Background"].inputs["Color"].default_value = (.6, .7, .9, 1)
    s.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .6
    bpy.ops.object.light_add(type="AREA", location=(-3, -4, 8))
    light = bpy.context.object
    light.data.energy, light.data.shape, light.data.size = 1600, "DISK", 7
    light.rotation_euler = (Vector((0, 0, 0)) - light.location).to_track_quat("-Z", "Y").to_euler()
    bg_color = cfg["background_color"]
    old.cube((0, 0, -.15), (50, 50, .2), 900, old.mat("ground", bg_color))
    roots, assets = [], []
    for k in range(3):
        root, angle, info = old.load_animal(cfg["asset"], k + 1, rng)
        roots.append((root, angle))
        assets.append(info)
    # Blender may flip an n-gon's triangulation while its vertices deform.
    # Triangulate the undeformed mesh once, before evaluating the armature.
    # Stable physical triangles then remain stable for the entire sequence.
    for obj in s.objects:
        if obj.type == "MESH" and 1 <= int(obj.get("track_id", 0)) <= 3:
            obj.data = obj.data.copy()
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
            bm.to_mesh(obj.data)
            bm.free()
            obj.data.update()
    # Appearance intervention is the only change in variant 1.
    color = cfg["occluder_color"] if variant != 1 else [1-x for x in cfg["occluder_color"]]
    blockers = []
    for j, x in enumerate(cfg["barrier_x"]):
        b = old.cube((x, -1.1, 1.6), (cfg["barrier_width"], .28, 3.2), 101+j,
                     old.mat("barrier_" + str(j), color))
        blockers.append(b)
    # Variant 2 is either a camera-only or occluder-motion-only intervention.
    camera_x = cfg["camera_x"]
    if variant == 2 and cfg["intervention"] == "camera":
        camera_x += .9
    bpy.ops.object.camera_add(location=(camera_x, -13, cfg["camera_height"]))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 1, .9))-camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 43
    camera.data.sensor_width = 36
    camera.data.sensor_fit = "HORIZONTAL"
    s.camera = camera
    return s, roots, blockers, assets


def pose(s, roots, blockers, cfg, variant, f):
    s.frame_set(f*2 + cfg["animation_offset"])
    t = f/(T-1)
    for k, (root, angle) in enumerate(roots):
        direction = 1 if (k+cfg["direction"]) % 2 == 0 else -1
        x = direction*(-3.7 + 7.4*t + cfg["accel"]*math.sin(2*math.pi*t))
        y = [.7, 2.7, 4.7][k] + .1*math.sin(2*math.pi*t+k)
        if cfg["event_type"] == "exit_return" and k == 0:
            x = direction*(-3.1 + 10*math.sin(math.pi*t))
        if cfg["event_type"] == "censored" and k == 0:
            x = direction*(-3.5+3.5*min(t/.55, 1.0))
        if cfg["event_type"] == "turn" and k == 0:
            angle += math.pi*math.sin(math.pi*t)
        root.location = (x, y, 0)
        root.rotation_euler.z = angle + (math.pi if direction < 0 else 0)
    if variant == 2 and cfg["intervention"] == "occluder_motion":
        for j, b in enumerate(blockers):
            b.location.x = cfg["barrier_x"][j] + .9*math.sin(2*math.pi*t+.6)
    bpy.context.view_layer.update()


def scene_mesh(s):
    dg = bpy.context.evaluated_depsgraph_get()
    points, triangles, owners = [], [], []
    objects, own = {}, collections.defaultdict(lambda: ([], []))
    face_refs = []
    for obj in s.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        e = obj.evaluated_get(dg)
        m = e.to_mesh()
        m.calc_loop_triangles()
        v = [e.matrix_world@p.co for p in m.vertices]
        tri = [tuple(p.vertices) for p in m.loop_triangles]
        identity = int(obj.get("track_id", 0))
        objects[obj.name] = dict(vertices=v, triangles=tri, owner=identity,
                                 topology=hashlib.sha256(np.asarray(tri, np.int32).tobytes()).hexdigest())
        start = len(points)
        points.extend(v)
        triangles.extend(tuple(start+i for i in face) for face in tri)
        owners.extend([identity]*len(tri))
        face_refs.extend((obj.name, j) for j in range(len(tri)))
        if 1 <= identity <= 3:
            ov, ot = own[identity]
            offset = len(ov)
            ov.extend(v)
            ot.extend(tuple(offset+i for i in face) for face in tri)
        e.to_mesh_clear()
    tree = BVHTree.FromPolygons(points, triangles, all_triangles=True)
    own_trees = {k: BVHTree.FromPolygons(v, tri, all_triangles=True)
                 for k, (v, tri) in own.items()}
    return tree, np.asarray(owners), objects, face_refs, own_trees, own


def projection(s, point, w=RGB_W, h=RGB_H):
    p = s.camera.matrix_world.inverted()@point
    focal = w*s.camera.data.lens/s.camera.data.sensor_width
    return np.array([w/2+focal*p.x/-p.z, h/2-focal*p.y/-p.z]), -p.z


def ray(s, x, y, w=W, h=H):
    focal = w*s.camera.data.lens/s.camera.data.sensor_width
    return (s.camera.matrix_world.to_quaternion()@Vector(
        ((x-w/2)/focal, (h/2-y)/focal, -1))).normalized()


def barycentric(p, a, b, c):
    v0, v1, v2 = b-a, c-a, p-a
    d00, d01, d11 = v0.dot(v0), v0.dot(v1), v1.dot(v1)
    d20, d21 = v2.dot(v0), v2.dot(v1)
    denom = d00*d11-d01*d01
    if abs(denom) < 1e-15:
        return None
    v, w = (d11*d20-d01*d21)/denom, (d00*d21-d01*d20)/denom
    return [1-v-w, v, w]


def initial_queries(s, mesh, seed):
    tree, owners, objects, refs, _, _ = mesh
    origin = s.camera.matrix_world.translation
    # Candidate sampling is exclusively from frame 0. No future-event filtering.
    candidates = collections.defaultdict(list)
    for y in range(3, H-3, 2):
        for x in range(3, W-3, 2):
            hit, _, face, _ = tree.ray_cast(origin, ray(s, x+.5, y+.5))
            if hit is None or not 1 <= owners[face] <= 3:
                continue
            name, idx = refs[face]
            obj = objects[name]
            tri = obj["triangles"][idx]
            bc = barycentric(hit, *(obj["vertices"][i] for i in tri))
            if bc is not None:
                candidates[int(owners[face])].append(dict(
                    mesh=name, triangle=idx, vertex_indices=list(tri), barycentric=bc,
                    owner=int(owners[face]), topology_sha256=obj["topology"],
                    surface_id=f"{name}:{idx}:{x}:{y}"))
    rng = random.Random(seed+700)
    out = []
    for owner in range(1, 4):
        rng.shuffle(candidates[owner])
        assert len(candidates[owner]) >= POINTS, (owner, len(candidates[owner]))
        out.extend(candidates[owner][:POINTS])
    return out


def labels(s, mesh, queries, frame):
    tree, owners, objects, refs, own_trees, own = mesh
    origin = s.camera.matrix_world.translation
    ids, depth = np.zeros((H, W), np.uint16), np.zeros((H, W), np.float32)
    # Keep ray directions fixed in this call; camera is static within each clip.
    for y in range(H):
        for x in range(W):
            loc, _, idx, distance = tree.ray_cast(origin, ray(s, x+.5, y+.5))
            if loc is not None:
                ids[y, x] = owners[idx]
                depth[y, x] = distance
    arrays, ratios, inframe, trunc = {"owner": ids, "ray_depth": depth}, [], [], []
    for owner in range(1, 4):
        coords = [projection(s, v, W, H)[0] for v in own[owner][0]]
        a = np.asarray(coords)
        lx, ly = np.floor(a.min(0)-1).astype(int)
        ux, uy = np.ceil(a.max(0)+1).astype(int)
        assert ux-lx < W*4 and uy-ly < H*4, "Unbounded full projection"
        amodal, full_count = np.zeros((H, W), np.uint8), 0
        for y in range(ly, uy+1):
            for x in range(lx, ux+1):
                if own_trees[owner].ray_cast(origin, ray(s, x+.5, y+.5))[0] is not None:
                    full_count += 1
                    if 0 <= x < W and 0 <= y < H:
                        amodal[y, x] = 1
        visible = ids == owner
        assert not np.any(visible & ~amodal.astype(bool))
        area = int(amodal.sum())
        arrays[f"amodal_{owner}"] = amodal
        ratios.append(float(visible.sum()/area) if area else -1.0)
        inframe.append(area > 0)
        trunc.append(1-area/full_count if full_count else 1.)
    xy, state, occ, valid, purity, world, reproj = [], [], [], [], [], [], []
    for q in queries:
        obj = objects[q["mesh"]]
        assert obj["topology"] == q["topology_sha256"], "Deformed topology changed"
        p = sum((obj["vertices"][i]*float(w) for i, w in
                 zip(q["vertex_indices"], q["barycentric"])), Vector())
        uv, z = projection(s, p)
        reference = world_to_camera_view(s, s.camera, p)
        error = np.linalg.norm(uv-np.array([reference.x*RGB_W, (1-reference.y)*RGB_H]))
        assert error <= .5, error
        reproj.append(error)
        world.append(list(p))
        xy.append(uv/[RGB_W, RGB_H])
        valid.append(True)
        if z <= 0 or not (0 <= uv[0] < RGB_W and 0 <= uv[1] < RGB_H):
            state.append(3)
            occ.append(0)
            purity.append(0.)
            continue
        direct = p-origin
        dist = direct.length
        direction = direct.normalized()
        own_hit = own_trees[q["owner"]].ray_cast(origin, direction)[0]
        hit, _, idx, hit_dist = tree.ray_cast(origin, direction)
        assert hit is not None and own_hit is not None
        owner = int(owners[idx])
        if (own_hit-origin).length < dist-.002:
            state.append(2)  # self hidden, even if another object is also in front
        elif hit_dist < dist-.002 and owner != q["owner"]:
            state.append(1)
        else:
            state.append(0)
        occ.append(owner)
        gx, gy = int(uv[0]/4), int(uv[1]/4)
        patch = ids[max(0, gy-2):min(H, gy+3), max(0, gx-2):min(W, gx+3)]
        purity.append(float((patch == q["owner"]).mean()) if state[-1] == 0 else 0.)
    # Independent Blender scene.ray_cast validates ownership, excluding ray ties.
    rng = random.Random(20261020+frame)
    for _ in range(24):
        x, y = rng.randrange(W), rng.randrange(H)
        found, _, _, _, obj, _ = s.ray_cast(
            bpy.context.evaluated_depsgraph_get(), origin, ray(s, x+.5, y+.5))
        expected = int(obj.get("track_id", 0)) if found else 0
        assert expected == int(ids[y, x]), (frame, x, y, expected, int(ids[y, x]))
    track = dict(xy=xy, state=state, occluder=occ, valid=valid, purity=purity,
                 world=world, instance_ratio=ratios, instance_inframe=inframe,
                 instance_truncation=trunc, reprojection_error=reproj)
    return arrays, track


def events(states, active, owners):
    out = []
    for q in range(states.shape[1]):
        if not active[q]:
            continue
        f = 1
        while f < T:
            if states[f, q] == 0:
                f += 1
                continue
            start = f
            while f < T and states[f, q] != 0:
                f += 1
            reasons = sorted(set(states[start:f, q].tolist()))
            out.append(dict(query=q, owner=owners[q], start=start, end=f,
                            gap=f-start, reappear=f if f < T else None,
                            right_censored=f == T, reasons=reasons,
                            physical_external_only=reasons == [1]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--frames", type=int, default=48)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    assert args.frames == 48, "Full event clips only"
    configs = json.loads(Path(args.manifest).read_text())
    selected = set(args.ids.split(","))
    for cfg in configs:
        if cfg["id"] not in selected:
            continue
        family = Path(args.output)/cfg["id"]
        family.mkdir(parents=True, exist_ok=False)
        dump(family/"config.json", cfg)
        queries, topology_meta = None, None
        for variant in range(3):
            out = family/str(variant)
            (out/"rgb").mkdir(parents=True)
            (out/"labels").mkdir()
            s, roots, blockers, assets = setup(cfg, variant)
            tracks, cameras = [], []
            for f in range(T):
                pose(s, roots, blockers, cfg, variant, f)
                mesh = scene_mesh(s)
                if queries is None:
                    queries = initial_queries(s, mesh, cfg["seed"])
                    dump(family/"surfaces.json", queries)
                arrays, track = labels(s, mesh, queries, f)
                tracks.append(track)
                cameras.append([list(row) for row in s.camera.matrix_world])
                np.savez_compressed(out/"labels"/f"{f:03d}.npz", **arrays)
                s.render.filepath = str(out/"rgb"/f"{f:03d}.png")
                bpy.ops.render.render(write_still=True)
            fields = {k: np.asarray([x[k] for x in tracks]) for k in tracks[0]}
            fields["active"] = fields["state"][0] == 0
            fields["owners"] = np.asarray([q["owner"] for q in queries])
            fields["timestamps"] = np.arange(T)/FPS
            np.savez_compressed(out/"tracks.npz", **fields)
            dump(out/"events.json", events(fields["state"], fields["active"], fields["owners"].tolist()))
            dump(out/"metadata.json", dict(assets=assets, camera_matrices=cameras,
                rgb_wh=[RGB_W, RGB_H], label_wh=[W, H], label_to_rgb_scale=4,
                label_pixel_centers=True, point_coordinates="normalized_edge_origin_xy",
                frames=T, fps=FPS, variant=variant, intervention=(
                    "base" if variant == 0 else "occluder_appearance" if variant == 1 else cfg["intervention"])))
        dump(family/"COMPLETE.json", dict(frames=T*3, finished_at=time.time()))


if __name__ == "__main__":
    main()
