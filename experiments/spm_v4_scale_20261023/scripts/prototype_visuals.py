import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import dump, read, sha
from data import Store
from task_context import task_context
from task_visualize import method, memory_figures, savefig


def main(root):
    parent = root/'runs'/read(root/'PROTOTYPE_FINAL_POINTER.json')['run_id']
    assert read(parent/'run.json')['execution_status'] == 'completed'
    memory = torch.load(parent/'memory.pt', map_location='cpu', weights_only=False)
    assert memory['prototype_total_steps'] == 20000
    with task_context(root, 'prototype_visuals', freeze='PROTOTYPE_VISUAL_FREEZE.json') as (run, meta):
        out = run/'figures'; out.mkdir(); store = Store(read(root/'ALL_DATASET.json')['rows'], max_cache=32)
        method(out); memory_figures(root, out, store)
        traces = []
        for stage in ['prototype_warm', 'prototype_final']:
            source = root/'runs'/read(root/(stage.upper()+'_POINTER.json'))['run_id']
            traces += read(source/'TRAINING.json')
        assert [r['step'] for r in traces] == list(range(1, 20001))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for key in ['loss', 'instance_mask_alignment', 'foreground', 'contact']:
            values = np.convolve([r[key] for r in traces], np.ones(100)/100, mode='valid')
            axes[0].plot(np.arange(len(values))+100, values, label=key)
        axes[0].set(ylabel='Loss, 100-step mean', title='20,000 actual task-supervised updates'); axes[0].legend(frameon=False, fontsize=8)
        for key in ['correspondence', 'instance_pull', 'contact_push']:
            values = np.convolve([r[key] for r in traces], np.ones(100)/100, mode='valid')
            axes[1].plot(np.arange(len(values))+100, values, label=key)
        axes[1].set_title('Cross-view and instance supervision'); axes[1].legend(frameon=False, fontsize=8)
        axes[2].plot([r['step'] for r in traces], [r['gradient_norm'] for r in traces], lw=.5)
        axes[2].set(ylabel='Gradient norm before clipping', title='Real optimization signal')
        for axis in axes:
            axis.axvline(5000, ls=':', c='#555'); axis.axvline(15000, ls=':', c='#555'); axis.set_xlabel('Global projector step')
        fig.suptitle('Real/cutout 5k -> mixed 10k -> real 5k | loss is not segmentation AP', fontsize=12)
        fig.tight_layout(); savefig(fig, out, '01_actual_task_prototype_learning')
        summary = {'parent_prototype_run_id': parent.name, 'memory_sha256': sha(parent/'memory.pt'),
                   'trained_global_steps': 20000, 'K': memory['K'], 'dimension': memory['dimension'],
                   'upstream_checkpoint_sha256': memory['upstream_checkpoint_sha256'],
                   'source_tokens': sum(c['support_tokens'] for c in memory['clusters']),
                   'new_training_steps_in_this_run': 0,
                   'figures': [p.name for p in sorted(out.glob('*.png'))], 'Mask_AP_gain': None}
        dump(run/'SUMMARY.json', summary); dump(run/'DECISION.json', {'status': 'actual_trained_prototypes_visualized',
                         'next_action': 'continue_matched_four_arm_segmentation_training_and_utility_evaluation'})
        meta.update(optimizer_steps=0, source_prototype_run_id=parent.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    torch.set_num_threads(2); main(Path(parser.parse_args().root))
