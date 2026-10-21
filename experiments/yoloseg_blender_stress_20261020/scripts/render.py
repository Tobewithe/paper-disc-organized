"""A bounded paired-scene smoke. Run only with the pinned Blender binary."""
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
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("asset_loader", ROOT/"sources/asset_loader.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
W, H, FRAMES = 640, 384, 48


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def setup(cfg):
    # The reused setup selects HDRI/ground by id parity. Keep both fixed within
    # a control/pressure pair rather than inadvertently changing appearance.
    s, roots, assets, devices = old.setup({**cfg, "id": cfg["id"]//2})
    if cfg["scenario"] == "density" and cfg["variant"] == "pressure":
        rng = random.Random(cfg["seed"]+19)
        for k in range(5, 9):
            root, angle, info = old.load_animal(cfg["asset"], k, rng)
            roots.append((root, angle)); assets.append(info)
    if cfg["scenario"] in ("wall", "fence") and cfg["variant"] == "control":
        for o in s.objects:
            if int(o.get("track_id", 0)) == 100:
                o.location.y = 9
    for obj in s.objects:
        if obj.type == "MESH" and 1 <= int(obj.get("track_id", 0)) <= 8:
            obj.data = obj.data.copy()
            bm = bmesh.new(); bm.from_mesh(obj.data)
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
            bm.to_mesh(obj.data); bm.free(); obj.data.update()
    s.render.resolution_x = W; s.render.resolution_y = H
    s.render.fps = 12; s.cycles.samples = 16
    s.render.use_motion_blur = False
    s.render.film_transparent = False
    return s, roots, assets, devices


def pose(s, roots, cfg, frame):
    s.frame_set(frame*2)
    t = frame/(FRAMES-1)
    for k, (o, angle) in enumerate(roots):
        direction = 1 if k % 2 == 0 else -1
        x = direction*(-3.8+7.6*t) + (.2 if k > 1 else 0)
        ys = [.8, 2, -.5, 3.1, 1.4, 2.6, 3.8, 5.0]
        if cfg["scenario"] == "crossing":
            ys = [.8, 3, 5.2, 7.4] if cfg["variant"] == "control" else [.8, 1.7, 2.6, 3.5]
        o.location = (x, ys[k], 0)
        o.rotation_euler.z = angle + (0 if direction == 1 else math.pi)
    bpy.context.view_layer.update()


def meshes(s):
    dg = bpy.context.evaluated_depsgraph_get()
    vertices, faces, owners = [], [], []
    groups = collections.defaultdict(lambda: ([], [])); objects = {}
    for obj in s.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        e = obj.evaluated_get(dg); mesh = e.to_mesh(); mesh.calc_loop_triangles()
        points = [e.matrix_world@v.co for v in mesh.vertices]
        triangles = [tuple(t.vertices) for t in mesh.loop_triangles]
        owner = int(obj.get("track_id", 0)); offset = len(vertices)
        vertices.extend(points); faces.extend(tuple(offset+i for i in t) for t in triangles)
        owners.extend([owner]*len(triangles))
        if 1 <= owner <= 8:
            vs, ts = groups[owner]; offset = len(vs)
            vs.extend(points); ts.extend(tuple(offset+i for i in t) for t in triangles)
            objects[obj.name] = (owner, points, triangles,
                hashlib.sha256(np.asarray(triangles, np.int32).tobytes()).hexdigest())
        e.to_mesh_clear()
    whole = BVHTree.FromPolygons(vertices, faces, all_triangles=True)
    own = {k: BVHTree.FromPolygons(vs, ts, all_triangles=True) for k, (vs, ts) in groups.items()}
    return whole, owners, own, groups, objects


def project(s, p):
    q = s.camera.matrix_world.inverted()@p
    assert q.z < -.01, "object crosses camera plane"
    focal = W*s.camera.data.lens/s.camera.data.sensor_width
    return np.array([W/2+focal*q.x/-q.z, H/2-focal*q.y/-q.z])


def ray(s, x, y):
    focal = W*s.camera.data.lens/s.camera.data.sensor_width
    return (s.camera.matrix_world.to_quaternion()@Vector(((x-W/2)/focal, (H/2-y)/focal, -1))).normalized()


def queries(objects, seed):
    groups = collections.defaultdict(list)
    for name, (owner, vs, ts, digest) in sorted(objects.items()):
        for index in range(len(ts)):
            groups[owner].append((owner, name, index, digest))
    rng = random.Random(seed+101)
    out = []
    for owner, rows in sorted(groups.items()):
        rng.shuffle(rows); out.extend(rows[:128])
    return out


def geometry(s, mesh, points, frame, rays):
    tree, owners, own, groups, objects = mesh
    origin = s.camera.matrix_world.translation
    ids = np.zeros((H, W), np.uint16)
    for y in range(H):
        for x in range(W):
            hit, _, face, _ = tree.ray_cast(origin, rays[y][x])
            if hit is not None:
                ids[y, x] = owners[face]
    masks = {}; records = []
    for track, (verts, _) in sorted(groups.items()):
        coords = np.asarray([project(s, p) for p in verts])
        lo = np.floor(coords.min(0)-1).astype(int); hi = np.ceil(coords.max(0)+1).astype(int)
        assert max(hi-lo) < W*4, "unbounded independent projection"
        amodal = np.zeros((H, W), np.uint8); total = 0
        for y in range(int(lo[1]), int(hi[1])+1):
            for x in range(int(lo[0]), int(hi[0])+1):
                direction = rays[y][x] if 0 <= x < W and 0 <= y < H else ray(s, x+.5, y+.5)
                if own[track].ray_cast(origin, direction)[0] is not None:
                    total += 1
                    if 0 <= x < W and 0 <= y < H:
                        amodal[y, x] = 1
        visible = ids == track; assert not np.any(visible & ~amodal.astype(bool))
        v, a = int(visible.sum()), int(amodal.sum())
        yy, xx = np.where(visible)
        occ, counts = np.unique(ids[amodal.astype(bool) & ~visible], return_counts=True)
        records.append(dict(track_id=track, visible_pixels=v, in_frame_pixels=a,
            full_pixels=total, visibility=v/a if a else None,
            truncation=1-a/total if total else None,
            bbox=[int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)] if v else None,
            state="out_of_view" if not a else "fully_occluded" if not v else "visible" if v == a else "partially_occluded",
            occluder_pixels={str(k):int(c) for k, c in zip(occ, counts)}))
        masks[f"amodal_{track}"] = amodal
    # Uniform and foreground checks prevent a predominantly-background audit.
    rng = random.Random(20261020+frame); fg = np.argwhere((ids > 0) & (ids <= 8))
    chosen = [(rng.randrange(H), rng.randrange(W)) for _ in range(64)]
    chosen += [tuple(fg[rng.randrange(len(fg))]) for _ in range(64)] if len(fg) else [
        (rng.randrange(H), rng.randrange(W)) for _ in range(64)]
    dg = bpy.context.evaluated_depsgraph_get()
    for y, x in chosen:
        hit, _, _, _, obj, _ = s.ray_cast(dg, origin, rays[y][x])
        expected = int(obj.get("track_id", 0)) if hit else 0
        assert expected == int(ids[y, x]), ("ray_identity", frame, int(x), int(y), expected, int(ids[y,x]))
    xy, valid, owner_list = [], [], []
    for owner, name, index, digest in points:
        _, verts, triangles, current = objects[name]; assert current == digest, "surface topology changed"
        p = sum((verts[i] for i in triangles[index]), Vector())/3
        uv = project(s, p); reference = world_to_camera_view(s, s.camera, p)
        assert np.linalg.norm(uv-np.array([reference.x*W, (1-reference.y)*H])) <= .5
        direct = p-origin; dist = direct.length
        hit, _, idx, hit_dist = tree.ray_cast(origin, direct.normalized())
        visible = hit is not None and owners[idx] == owner and abs(hit_dist-dist) <= .002
        x, y = math.floor(uv[0]), math.floor(uv[1])
        pure = 1 <= x < W-1 and 1 <= y < H-1 and np.all(ids[y-1:y+2, x-1:x+2] == owner)
        xy.append(uv); valid.append(bool(visible and pure)); owner_list.append(owner)
    return dict(instance_id=ids, **masks, point_xy=np.asarray(xy, np.float32),
        point_valid=np.asarray(valid, bool), point_owner=np.asarray(owner_list, np.int16)), records


def main():
    output = Path(sys.argv[sys.argv.index("--")+1]); output.mkdir(exist_ok=True)
    (output/"images").mkdir(); (output/"labels").mkdir()
    configs = json.loads((ROOT/"SCENES.json").read_text())
    start = time.time(); all_frames = []
    for cfg in configs:
        scene, roots, assets, devices = setup(cfg)
        pose(scene, roots, cfg, 0)
        points = queries(meshes(scene)[-1], cfg["seed"])
        rays = [[ray(scene, x+.5, y+.5) for x in range(W)] for y in range(H)]
        scene_out = output/f"scene_{cfg['id']:02d}"; scene_out.mkdir()
        save(scene_out/"scene.json", dict(config=cfg, assets=assets, devices=devices, queries=points,
            camera_matrix=[list(r) for r in scene.camera.matrix_world], lens=scene.camera.data.lens))
        for frame in range(FRAMES):
            pose(scene, roots, cfg, frame)
            arrays, records = geometry(scene, meshes(scene), points, frame, rays)
            image_id = cfg["id"]*1000+frame+1
            np.savez_compressed(output/"labels"/f"{image_id:012d}.npz", **arrays)
            scene.render.filepath = str(output/"images"/f"{image_id:012d}.png")
            bpy.ops.render.render(write_still=True)
            record = dict(image_id=image_id, scene_id=cfg["id"], frame=frame, scenario=cfg["scenario"],
                variant=cfg["variant"], asset=cfg["asset"], records=records,
                rgb_sha256=hashlib.sha256(Path(scene.render.filepath).read_bytes()).hexdigest())
            all_frames.append(record)
            with (output/"FRAMES.jsonl").open("a") as f:
                f.write(json.dumps(record)+"\n")
            save(output/"PROGRESS.json", dict(stage="render", completed_frames=len(all_frames),
                 planned_frames=384, scene_id=cfg["id"], frame=frame, elapsed_s=time.time()-start))
            print("FRAME_DONE", cfg["id"], frame, len(all_frames), round(time.time()-start,1), flush=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(scene_out/"scene.blend"))
    assert len(all_frames) == 384
    save(output/"RENDER_SUMMARY.json", dict(frames=len(all_frames), scenes=len(configs),
        independent_ray_checks=len(all_frames)*128, width=W, height=H,
        elapsed_s=time.time()-start, training_steps=0, blender_version=bpy.app.version_string))


if __name__ == "__main__":
    main()
