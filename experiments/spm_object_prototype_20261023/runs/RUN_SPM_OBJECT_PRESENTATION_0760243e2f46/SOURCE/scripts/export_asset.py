"""Blender-only export. Canonical mesh, weights and true evaluated skin transforms."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


def rotate(arm, name, angle):
    bone = arm.pose.bones[name]
    rest = bone.bone.matrix_local.to_quaternion()
    bone.rotation_mode = 'QUATERNION'
    bone.rotation_quaternion = rest.inverted() @ Quaternion(Vector((1, 0, 0)), angle) @ rest


def reset_pose(arm):
    arm.animation_data_clear()
    for bone in arm.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
        for constraint in bone.constraints:
            constraint.mute = True


def part_name(name):
    if name == 'Head': return 0
    if name == 'Neck': return 1
    if name in ['Body', 'Torso', 'Shoulders', 'root']: return 2
    if name in ['Back', 'Hips']: return 3
    if name.startswith('Front'): return 4 if name.endswith('.L') else 5
    if name.startswith('Back'): return 6 if name.endswith('.L') else 7
    raise ValueError(('unmapped_bone', name))


def export(cfg, folder):
    assert hashlib.sha256(Path(cfg['asset']).read_bytes()).hexdigest() == cfg['asset_sha256']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(cfg['asset'], link=False) as (src, dst):
        dst.objects = src.objects
        dst.actions = src.actions
    for obj in dst.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
    mesh_obj = next(o for o in dst.objects if o is not None and o.type == 'MESH')
    arm = next(o for o in dst.objects if o is not None and o.type == 'ARMATURE')
    modifiers = [m for m in mesh_obj.modifiers if m.type == 'ARMATURE']
    assert len(modifiers) == 1
    modifiers[0].use_deform_preserve_volume = False
    assert not any(m.type != 'ARMATURE' and m.show_viewport for m in mesh_obj.modifiers)
    reset_pose(arm)
    bpy.context.view_layer.update()
    verts = np.array([list(mesh_obj.matrix_world @ v.co) for v in mesh_obj.data.vertices], np.float64)
    length = float(np.ptp(verts[:, 1]))
    center = (verts.max(0)+verts.min(0))/2
    norm = np.eye(4)
    norm[:3, :3] /= length
    norm[:3, 3] = -center/length
    canonical = (verts-center)/length
    mesh_obj.data.calc_loop_triangles()
    faces = np.array([list(t.vertices) for t in mesh_obj.data.loop_triangles], np.int32)
    names = [b.name for b in arm.data.bones]
    weights = np.zeros((len(verts), len(names)), np.float64)
    groups = {g.index: g.name for g in mesh_obj.vertex_groups}
    for vertex in mesh_obj.data.vertices:
        for item in vertex.groups:
            name = groups[item.group]
            if name in names:
                weights[vertex.index, names.index(name)] = item.weight
    assert np.all(weights.sum(1) > 1e-8), ('unbound_vertices', weights.sum(1).min())
    weights /= weights.sum(1, keepdims=True)
    part_weights = np.zeros((len(verts), 8), np.float64)
    for j, name in enumerate(names):
        part_weights[:, part_name(name)] += weights[:, j]
    mesh_world = np.array(mesh_obj.matrix_world, np.float64)
    arm_world = np.array(arm.matrix_world, np.float64)
    inv_rest = [np.linalg.inv(np.array(b.matrix_local, np.float64)) for b in arm.data.bones]
    poses, targets, parity = [], [], []
    for name in cfg['poses']:
        reset_pose(arm)
        if name == 'walking':
            # Read the real action, then bake its evaluated rotations/constraints into skin matrices.
            for b in arm.pose.bones:
                for c in b.constraints: c.mute = False
            action = next(a for a in dst.actions if a.name.split('.')[0] == 'Walk')
            arm.animation_data_create()
            for track in arm.animation_data.nla_tracks: track.mute = True
            arm.animation_data.action = action
            bpy.context.scene.frame_set(16)
        elif name == 'feeding':
            rotate(arm, 'Neck', .26); rotate(arm, 'Head', .18)
        elif name == 'prone':
            for side in ['L', 'R']:
                for bone, angle in [('FrontUpLeg.', 1.30), ('FrontLowLeg.', 1.20), ('BackUpLeg.', -1.48), ('BackLowLeg.', -1.15)]:
                    rotate(arm, bone+side, angle)
        elif name.startswith('lateral'):
            for side in ['L', 'R']:
                for bone, angle in [('FrontUpLeg.', .5), ('FrontLowLeg.', .45), ('BackUpLeg.', -.55), ('BackLowLeg.', -.4)]:
                    rotate(arm, bone+side, angle)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        eval_arm = arm.evaluated_get(dg)
        transforms = np.array([norm @ arm_world @ np.array(eval_arm.pose.bones[n].matrix) @ inv_rest[j] @ np.linalg.inv(arm_world) @ np.linalg.inv(norm)
                               for j, n in enumerate(names)])
        evaluated = mesh_obj.evaluated_get(dg)
        data = evaluated.to_mesh()
        actual = np.array([list(evaluated.matrix_world @ v.co) for v in data.vertices], np.float64)
        evaluated.to_mesh_clear()
        actual = (actual-center)/length
        if name.startswith('lateral'):
            angle = (1 if name == 'lateral_left' else -1)*1.48
            rot = np.array(Matrix.Rotation(angle, 4, 'Y'))
            transforms = np.einsum('ij,bjk->bik', rot, transforms)
            actual = actual @ rot[:3, :3].T
        homogeneous = np.c_[canonical, np.ones(len(canonical))]
        skin = np.einsum('vb,bij,vj->vi', weights, transforms, homogeneous)[:, :3]
        error = float(np.max(np.abs(skin-actual)))
        assert error < 2e-5, ('skinning_parity', name, error)
        poses.append(transforms[:, :3].astype(np.float32))
        targets.append(actual.astype(np.float32))
        parity.append({'pose': name, 'max_abs_error': error})
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder/'asset.npz', canonical=canonical.astype(np.float32), faces=faces,
                        weights=weights.astype(np.float32), part_weights=part_weights.astype(np.float32),
                        poses=np.array(poses), posed_vertices=np.array(targets))
    report = {'asset': cfg['asset'], 'asset_sha256': cfg['asset_sha256'], 'blender': bpy.app.version_string,
              'vertices': len(verts), 'triangles': len(faces), 'bone_names': names, 'poses': cfg['poses'],
              'part_names': ['head', 'neck', 'front_trunk', 'rear_trunk', 'front_left', 'front_right', 'rear_left', 'rear_right'],
              'skinning_parity': parity, 'coordinate_system': 'asset Z up, Y body length, centered and divided by rest body length',
              'skinning_representation': 'evaluated per-bone affine transforms; not a learned joint-angle FK model',
              'rig_and_part_groups_are_given_prior': True}
    (folder/'ASSET_REPORT.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    cfg = json.loads((Path(args.root)/'CONFIG_V0.json').read_text(encoding='utf-8'))
    export(cfg, Path(args.output))
