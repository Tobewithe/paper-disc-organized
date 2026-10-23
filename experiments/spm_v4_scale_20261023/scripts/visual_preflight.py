"""CPU render review; consumes completed geometry files, never changes training inputs."""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import dump, read, sha
from task_context import task_context
from task_visualize import method, overlay, savefig, representative_scenes


def main(root):
    with task_context(root, 'visual_preflight', freeze='VISUAL_PREFLIGHT_FREEZE.json') as (run, meta):
        out = run/'figures'; out.mkdir(); method(out)
        recipes = read(root/'SCENES.json')['formal']; completed = {}
        for shard in range(3):
            for file in (root/'render_plans'/f'shard_{shard}'/'runs').glob('*/scenes/*/scene.json'):
                completed[file.parent.name] = file
        # Fixed recipe bins cover all four layouts/floors/densities and both cameras; no AP or quality selection.
        selected_recipes = representative_scenes(recipes)
        chosen = [completed[r['geometry_id']] for r in selected_recipes]
        fig, axes = plt.subplots(8, 3, figsize=(12, 17)); records = []
        for i, path in enumerate(chosen):
            scene = read(path); recipe = scene['recipe']; folder = path.parent
            images = [cv2.imread(str(folder/f'rgb_{a}.png')) for a in [0, 1]]
            assert all(image is not None for image in images)
            ids = np.load(folder/'labels.npz')['instance_id']
            masks = [ids == k for k in np.unique(ids) if k]
            panels = [images[0][:, :, ::-1], overlay(images[0][:, :, ::-1], masks), images[1][:, :, ::-1]]
            for j, panel in enumerate(panels):
                axes[i, j].imshow(panel); axes[i, j].axis('off')
            axes[i, 0].set_title(f"{recipe['geometry_id']} | {recipe['layout']} | {recipe['floor_type']} | {recipe['requested_pigs']} pigs | {recipe['camera_family']}", fontsize=8)
            if i == 0:
                axes[i, 1].set_title('Rendered visible instance IDs', fontsize=10)
                axes[i, 2].set_title('Same geometry, second appearance', fontsize=10)
            records.append({'geometry_id': recipe['geometry_id'], 'scene_file': str(path), 'scene_sha256': sha(path),
                            'composition': recipe['composition'], 'camera_family': recipe['camera_family'], 'layout': recipe['layout'],
                            'floor_type': recipe['floor_type'], 'density_bin': recipe['density_bin'],
                            'poses': dict(Counter(recipe['poses'])), 'requested_pigs': recipe['requested_pigs'],
                            'visible_instance_ids': len(masks), 'ray_check': scene['ray_check'],
                            'scope': 'visual_preflight_of_completed_geometry_not_full_dataset_acceptance'})
        fig.suptitle('Actual scale render preflight | fixed layout / floor / density / camera coverage', fontsize=12)
        fig.tight_layout(); savefig(fig, out, '01_scene_label_appearance_preflight')
        dump(run/'SUMMARY.json', {'review_scenes': records, 'implementation_diagram': 'figures/00_method_and_supervision.png',
                                  'training_steps': 0, 'training_inputs_modified': False, 'method_utility_not_measured': True})
        dump(run/'DECISION.json', {'status': 'render_previews_ready_for_visual_review',
                                   'next_action': 'continue_existing_baseline_data_and_method_dependencies'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    main(Path(parser.parse_args().root))
