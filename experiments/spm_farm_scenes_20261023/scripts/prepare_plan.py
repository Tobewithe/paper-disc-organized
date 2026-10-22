"""Prepare train-only reference statistics and deterministic grouped scene recipes."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'spm_prototype_training_20261022' / 'INPUT_MANIFEST.json'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def quantiles(xs):
    xs = sorted(xs)
    return {str(p): xs[round((len(xs)-1)*p)] for p in [0, .1, .25, .5, .75, .9, 1]}


def scene_recipe(index, spec, split, smoke=False):
    composition, layout, density, camera = spec
    seed = 230100 + index
    rng = random.Random(seed)
    count = [3, 8, 14, 20][density]
    poses = []
    for i in range(count):
        if composition == 'active':
            pose = rng.choice(['standing', 'walking', 'feeding'])
        elif composition == 'resting':
            pose = rng.choice(['prone', 'lateral_left', 'lateral_right'])
        elif composition == 'mixed':
            pose = ['standing', 'prone', 'lateral_left', 'walking', 'lateral_right', 'feeding'][i % 6]
        else:
            pose = 'feeding' if i < max(2, count//3) else rng.choice(['prone', 'lateral_left', 'standing'])
        poses.append(pose)
    return {'geometry_id': f"{'SMOKE' if smoke else 'FARM'}_{index:04d}", 'index': index,
            'seed': seed, 'split': split, 'composition': composition, 'layout': layout,
            'density_bin': density, 'requested_pigs': count, 'poses': poses,
            'camera_family': camera, 'camera_elevation_degrees': rng.uniform(68, 83) if camera == 'overhead' else rng.uniform(38, 58),
            'camera_azimuth_degrees': rng.choice([15, 75, 165, 245, 310]) + rng.uniform(-10, 10),
            'lens_mm': rng.uniform(30, 42),
            'pen_dimensions_m': [7.6 + max(0, count-14)*.3, 6.2 + max(0, count-14)*.15],
            'body_lengths_m': [rng.uniform(1.3, 1.8) for _ in range(count)],
            'pose_frame': [rng.randint(4, 60) for _ in range(count)],
            'floor_type': ['concrete', 'slatted', 'bedded', 'stained_concrete'][index % 4],
            'fence_type': ['tubular', 'vertical_bars', 'low_wall'][index % 3],
            'appearances': [{'id': a, 'seed': seed + 100000*(a+1),
                              'light_type': ['overhead_soft', 'side_window', 'dim_indoor'][(index+a) % 3],
                              'camera_quality': ['clean', 'mild_blur', 'compressed'][(index+2*a) % 3]}
                             for a in range(2)]}


def main():
    data = json.loads(OLD.read_text(encoding='utf-8'))
    rows = [r for r in data['rows'] if r['domain'] == 'real' and r['split'] == 'train']
    areas, lengths, centres = [], [], []
    for row in rows:
        for ann in row['annotations']:
            x, y, w, h = ann['bbox']
            areas.append(float(ann['area']) / (row['width']*row['height']))
            lengths.append(max(w/row['width'], h/row['height']))
            centres.append([(x+w/2)/row['width'], (y+h/2)/row['height']])
    profile = {'source_manifest': str(OLD.relative_to(ROOT.parent.parent)),
               'source_manifest_sha256': hashlib.sha256(OLD.read_bytes()).hexdigest(),
               'train_images': len(rows), 'train_videos': len({r['video_id'] for r in rows}),
               'source_ids': [r['source_id'] for r in rows], 'dev_or_test_used': False,
               'pigs_per_image_quantiles': quantiles([len(r['annotations']) for r in rows]),
               'visible_area_fraction_quantiles': quantiles(areas),
               'box_long_axis_fraction_quantiles': quantiles(lengths),
               'image_aspect_ratio_quantiles': quantiles([r['width']/r['height'] for r in rows]),
               'mean_normalized_box_centre': [statistics.mean(z[i] for z in centres) for i in range(2)],
               'measured': ['image_dimensions', 'visible_mask_area', 'box_size', 'pig_count', 'video_source'],
               'designed_not_measured': ['standing_lying_frequencies', 'physical_camera_intrinsics', 'pen_dimensions', 'light_parameters'],
               'physical_pen_independence_verified': False}
    dump(ROOT / 'REAL_TRAIN_PROFILE.json', profile)
    specs = list(itertools.product(['active', 'resting', 'mixed', 'feeding_rest'],
                                  ['scatter', 'rest_rows', 'feeder_cluster', 'fence_strip'], range(4),
                                  ['overhead', 'oblique']))
    dev = set(sorted(range(len(specs)), key=lambda i: hashlib.sha256(f'farmdev0/{i}'.encode()).hexdigest())[:16])
    scenes = [scene_recipe(i, spec, 'dev' if i in dev else 'train') for i, spec in enumerate(specs)]
    smoke_specs = [specs[i] for i in [0, 11, 48, 59, 68, 79, 116, 127]]
    smoke = [scene_recipe(9000+i, spec, 'engineering', smoke=True) for i, spec in enumerate(smoke_specs)]
    dump(ROOT / 'SCENES.json', {'formal': scenes, 'smoke': smoke})
    cfg = {'study_id': 'STUDY_SPM_FARM_SCENES_20261023', 'seed': 0,
           'resolution': [640, 384], 'render_samples': 16, 'ray_check_samples_per_geometry': 4096,
           'minimum_visible_pixels_for_annotation': 16, 'floor_contact_tolerance_m': .006,
           'maximum_ray_id_mismatch_fraction': .005, 'poses_are_skeletal_not_death_action': True,
           'asset': '/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2/assets/farm/Farm Animals by @Quaternius/Blends/Pig.blend',
           'asset_sha256': '8aec801123f31ca475000c738287dcd075dda6be4c5650c03b5657d801a847ef',
           'floor_texture': '/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2/assets/concrete_floor_02.jpg',
           'formal_geometry_count': 128, 'appearance_count': 2,
           'single_asset_limit': 'one low-poly Pig asset; scaling and skeletal variation are not independent animal assets',
           'real_profile_sha256': hashlib.sha256((ROOT/'REAL_TRAIN_PROFILE.json').read_bytes()).hexdigest(),
           'scenes_sha256': hashlib.sha256((ROOT/'SCENES.json').read_bytes()).hexdigest()}
    dump(ROOT / 'CONFIG.json', cfg)
    print(json.dumps({'train_images': len(rows), 'pigs_per_image': profile['pigs_per_image_quantiles'],
                      'box_long_axis': profile['box_long_axis_fraction_quantiles'],
                      'formal_geometry': len(scenes), 'formal_images': len(scenes)*2,
                      'pose_counts': dict(Counter(p for s in scenes for p in s['poses']))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
