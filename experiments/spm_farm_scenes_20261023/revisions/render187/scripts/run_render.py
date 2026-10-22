"""One-shot Slurm renderer supervisor, dataset checks and traceable delivery."""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as cm


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def manifest(run):
    dump(run/'ARTIFACT_MANIFEST.json', {p.relative_to(run).as_posix(): {'sha256': sha(p), 'size_bytes': p.stat().st_size}
                                      for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json'})


def postprocess(root, run, stage):
    cfg = json.loads((root/'CONFIG.json').read_text())
    width, height = cfg['resolution']
    coco = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'pig'}]}
    entries, poses, splits, layouts, floors, cameras = [], Counter(), Counter(), Counter(), Counter(), Counter()
    vis_pigs, small_pigs, ray_mismatches, ray_samples = 0, 0, 0, 0
    scene_paths = sorted((run/'scenes').glob('*/scene.json'))
    previews, mask_previews = [], []
    count_hist, lengths, all_contact = [], [], []
    for path in scene_paths:
        meta = json.loads(path.read_text()); recipe = meta['recipe']; folder = path.parent
        ids = np.load(folder/'labels.npz')['instance_id']
        assert ids.shape == (height, width) and ids.dtype == np.uint16
        assert ids.max() <= recipe['requested_pigs']
        assert meta['paired_instance_ids_equal'] and meta['all_pig_xy_footprints_disjoint']
        for pig in meta['pigs']:
            assert abs(pig['contact_min_z_m']) <= cfg['floor_contact_tolerance_m']
            all_contact.append(pig['contact_min_z_m'])
        poses.update(recipe['poses']); splits.update([recipe['split']]); layouts.update([recipe['layout']])
        floors.update([recipe['floor_type']]); cameras.update([recipe['camera_family']])
        ray_mismatches += meta['ray_check']['mismatches']; ray_samples += meta['ray_check']['samples']
        masks = []
        for pig in meta['pigs']:
            target = np.asfortranarray(ids == pig['pig_id'], dtype=np.uint8)
            area = int(target.sum())
            if area < cfg['minimum_visible_pixels_for_annotation']:
                small_pigs += 1
                continue
            rle = cm.encode(target)
            assert np.array_equal(cm.decode(rle), target)
            box = cm.toBbox(rle).astype(float).tolist()
            masks.append((pig, area, box, {'size': rle['size'], 'counts': rle['counts'].decode()}))
            lengths.append(max(box[2]/width, box[3]/height))
        count_hist.append(len(masks)); vis_pigs += len(masks)
        for style in recipe['appearances']:
            a = style['id']; rgb = folder/f'rgb_{a}.png'
            image = cv2.imread(str(rgb)); assert image is not None and image.shape == (height, width, 3)
            raw_path = folder/f'raw_rgb_{a}.png'; rgb.rename(raw_path)
            if style['camera_quality'] == 'mild_blur':
                image = cv2.GaussianBlur(image, (3, 3), .55)
            if style['camera_quality'] == 'compressed':
                encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            assert cv2.imwrite(str(rgb), image)
            image_id = recipe['index']*2+a+1
            coco['images'].append({'id': image_id, 'file_name': rgb.relative_to(root).as_posix(),
                                   'width': width, 'height': height, 'split': recipe['split'],
                                   'geometry_id': recipe['geometry_id'], 'appearance_id': a})
            for pig, area, box, rle in masks:
                coco['annotations'].append({'id': len(coco['annotations'])+1, 'image_id': image_id,
                                             'category_id': 1, 'iscrowd': 0, 'segmentation': rle,
                                             'bbox': box, 'area': area, 'pig_id': pig['pig_id'], 'pose': pig['pose']})
            entries.append({'source_id': f"synthetic/{recipe['geometry_id']}/{a}", 'domain': 'synthetic',
                            'group': recipe['geometry_id'], 'split': recipe['split'], 'appearance_id': a,
                            'rgb_path': str(rgb), 'label_path': str(folder/'labels.npz'),
                            'rgb_sha256': sha(rgb), 'label_sha256': sha(folder/'labels.npz'),
                            'raw_rgb_sha256': sha(raw_path), 'geometry_metadata_sha256': sha(path),
                            'instance_count': len(masks), 'pose_counts': dict(Counter(recipe['poses'])),
                            'requested_instance_count': recipe['requested_pigs']})
            if a == 0:
                previews.append((recipe['geometry_id'], recipe['composition'], image))
                overlay = image.copy()
                for pig, _, _, _ in masks:
                    mask = (ids == pig['pig_id']).astype(np.uint8)
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    color = tuple(int(x) for x in np.random.default_rng(pig['pig_id']).integers(70, 256, size=3))
                    overlay[mask > 0] = (.7*overlay[mask > 0]+.3*np.asarray(color)).astype(np.uint8)
                    cv2.drawContours(overlay, contours, -1, color, 1)
                mask_previews.append((recipe['geometry_id'], recipe['composition'], overlay))
    expected = 8 if stage == 'smoke' else cfg['formal_geometry_count']
    assert len(scene_paths) == expected and len(entries) == expected*2
    dump(run/'annotations.json', coco)
    dump(run/'DATA_MANIFEST.json', {'rows': entries, 'geometry_scenes': expected,
                                   'same_geometry_pair_is_not_independent': True,
                                   'source_asset_sha256': cfg['asset_sha256']})
    def sheet(rows, name):
        # Deterministic evenly spaced formal samples; all smoke scenes shown.
        indices = list(range(len(rows))) if len(rows) <= 8 else [round(i*(len(rows)-1)/7) for i in range(8)]
        canvas = Image.new('RGB', (960, 4*312), (240, 240, 240)); draw = ImageDraw.Draw(canvas)
        for j, ix in enumerate(indices):
            gid, composition, im = rows[ix]
            thumb = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).resize((480, 288))
            x, y = j % 2*480, j//2*312
            canvas.paste(thumb, (x, y+24)); draw.text((x+7, y+5), f'{gid}  {composition}', fill=(0, 0, 0))
        canvas.save(run/name, quality=93)
    sheet(previews, 'PREVIEW_RGB.jpg'); sheet(mask_previews, 'PREVIEW_MASKS.jpg')
    vals = sorted(lengths)
    summary = {'stage': stage, 'geometry_scenes': expected, 'RGB_images': len(entries),
               'split_geometry_counts': dict(splits), 'pose_instances_once_per_geometry': dict(poses),
               'requested_pigs_once_per_geometry': sum(poses.values()),
               'visible_instances_once_per_geometry': vis_pigs,
               'annotations_across_appearances': len(coco['annotations']),
               'fully_hidden_or_sub16pixel_instances_once_per_geometry': small_pigs,
               'layouts': dict(layouts), 'floors': dict(floors), 'camera_families': dict(cameras),
               'visible_pigs_per_geometry_min_max': [min(count_hist), max(count_hist)],
               'box_long_axis_fraction_quantiles': {str(p): vals[round((len(vals)-1)*p)] for p in [.1, .5, .9]},
               'ray_check_samples': ray_samples, 'ray_check_mismatches': ray_mismatches,
               'ray_check_mismatch_fraction': ray_mismatches/ray_samples,
               'max_abs_ground_contact_z_m': max(abs(z) for z in all_contact),
               'RLE_roundtrip_passed': True, 'appearance_pair_ID_equal': True,
               'labels_resolution': [width, height], 'data_manifest_sha256': sha(run/'DATA_MANIFEST.json'),
               'annotations_sha256': sha(run/'annotations.json'),
               'real_AP_gain': None, 'limits': [cfg['single_asset_limit'], 'pose and camera frequencies partly designed, not fully fitted',
                                               'same-asset scene holdout is not independent-asset validation']}
    dump(run/'VALIDATION.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['smoke', 'formal']); p.add_argument('--root', required=True)
    args = p.parse_args(); root = Path(args.root)
    freeze = json.loads((root/'FREEZE_MANIFEST.json').read_text())
    for name, digest in freeze['files'].items():
        assert sha(root/name) == digest, name
    lock = root/('LOCK_RENDER_'+args.stage)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    run = root/'runs'/('RUN_SPM_FARM_'+args.stage.upper()+'_'+uuid.uuid4().hex[:12]); run.mkdir(parents=True)
    meta = {'study_id': 'STUDY_SPM_FARM_SCENES_20261023', 'run_id': run.name, 'stage': args.stage,
            'started_at': utc(), 'execution_status': 'running', 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
            'executor': os.uname().nodename, 'pid': os.getpid(), 'command': sys.argv,
            'code_commit': freeze['git_head'], 'freeze_manifest_sha256': sha(root/'FREEZE_MANIFEST.json'),
            'protocol_sha256': sha(root/'PROTOCOL.md'), 'config_sha256': sha(root/'CONFIG.json'),
            'scientific_status': 'synthetic_dataset_engineering_not_method_utility'}
    dump(run/'run.json', meta)
    dump(run/'EVAL_CONTRACT.json', {'task': 'visible_instance_masks', 'native_mask_resolution': [640, 384],
                                  'runtime_config': json.loads((root/'CONFIG.json').read_text()),
                                  'GT_at_deployment': False, 'AP_evaluated': False})
    print('RUN', run.name, flush=True); start = time.monotonic()
    try:
        blender = '/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion/software/blender-4.2.1-linux-x64/blender'
        cmd = [blender, '-b', '--factory-startup', '-t', '4', '--python', str(root/'scripts/render_farm.py'),
               '--', '--root', str(root), '--run', str(run), '--stage', args.stage]
        dump(run/'RENDER_COMMAND.json', cmd)
        with open(run/'BLENDER.log', 'w') as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line); log.flush()
                if 'SCENE_COMPLETE' in line or 'Error' in line or 'Traceback' in line or 'RENDER_COMPLETE' in line:
                    print(line.rstrip(), flush=True)
            code = process.wait()
        assert code == 0, f'Blender failed exit={code}; inspect BLENDER.log'
        summary = postprocess(root, run, args.stage)
        summary['elapsed_seconds'] = time.monotonic()-start
        dump(run/'SUMMARY.json', summary)
        dump(run/'DECISION.json', {'status': 'pending_visual_QA' if args.stage == 'smoke' else 'dataset_generated_pending_receipt',
                                   'scope': 'geometry_and_label_engineering_only',
                                   'next_action': 'visually_review_all_smoke_scenes' if args.stage == 'smoke' else 'freeze_new_prototype_training_inputs'})
        meta.update(execution_status='completed', ended_at=utc(), exit_code=0)
        dump(run/'run.json', meta)
        dump(root/(args.stage.upper()+'_POINTER.json'), {'run_id': run.name, 'slurm_job_id': meta['slurm_job_id']})
        print('COMPLETE', json.dumps(summary), flush=True)
    except BaseException:
        (run/'ERROR.log').write_text(traceback.format_exc())
        meta.update(execution_status='failed', ended_at=utc(), exit_code=1)
        dump(run/'run.json', meta)
        raise
    finally:
        manifest(run); lock.unlink()


if __name__ == '__main__':
    main()
