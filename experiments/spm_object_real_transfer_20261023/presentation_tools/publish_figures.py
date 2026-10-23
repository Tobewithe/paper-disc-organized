"""Scientific plots and pre-fixed real examples; no training or model selection."""
import argparse
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as cm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rt import context, dump, read, sha


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white'); plt.close(fig)


def overlay(rgb, mask, gt=None):
    out = rgb.copy().astype(np.float32)
    m = mask.astype(bool)
    out[m] = .55*out[m]+.45*np.array([244, 148, 48])
    if gt is not None:
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (50, 240, 160), 1)
    return np.clip(out, 0, 255).astype(np.uint8)


def crop(a, box):
    x0, y0, x1, y1 = box
    return a[max(0, int(np.floor(y0))):min(640, int(np.ceil(y1))),
             max(0, int(np.floor(x0))):min(640, int(np.ceil(x1)))]


def execute(root):
    torch.set_num_threads(4)
    with context(root, 'presentation') as (out, meta, cfg):
        source = root/'runs'/read(root/'FORMAL_POINTER.json')['run_id']
        summary = read(source/'SUMMARY.json'); cache = read(source/'CACHE_MANIFEST.json')
        bank_pointer = read(root/'BANK_POINTER.json')
        bank = torch.load(bank_pointer['file'], map_location='cpu', weights_only=False)
        native = read(source/'native_predictions.json')
        s = read(source/'structured_predictions.json'); q = read(source/'scrambled_control_predictions.json')
        fig, ax = plt.subplots(figsize=(8, 4))
        for tag, label in [('structured', 'Aligned object S'), ('scrambled_control', 'Layout-scrambled Q')]:
            logs = read(source/(tag+'_TRAINING.json'))['losses']
            ax.plot([r['step'] for r in logs], [r['loss'] for r in logs], label=label)
        ax.set(xlabel='Actual optimizer steps per arm', ylabel='Training objective', title='Fixed 4000-step training; final checkpoint')
        ax.legend(); ax.grid(alpha=.2); save(fig, out/'01_training.png')
        tags = ['native', 'scrambled_control', 'structured', 'structured_off', 'structured_scrambled']
        labels = ['Native N', 'Control Q', 'Object S', 'S: object off', 'S: layout scrambled']
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, key, title in zip(axes, ['mask_ap', 'mask_ap75', 'mask_small_ap'], ['Mask AP50-95', 'Mask AP75', 'Small Mask AP']):
            values = [summary['metrics'][t][key] for t in tags]
            bars = ax.bar(range(len(tags)), values, color=['#8996a5', '#c5a25a', '#3287a8', '#b8cfdd', '#86a5af'])
            ax.set_xticks(range(len(tags)), labels, rotation=35, ha='right'); ax.set_title(title+' (pp)')
            ax.set_ylim(max(0, min(values)-5), max(values)+2)
            for bar, value in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, value+.1, f'{value:.2f}', ha='center', fontsize=9)
        fig.suptitle('432 real development images; all methods share boxes, scores and detection identities')
        fig.tight_layout(); save(fig, out/'02_real_segmentation.png')
        fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(4)
        dtags = tags[1:]
        ax.bar(x-.18, [summary['damage'][t]['failure_to_success'] for t in dtags], .36, label='Native Mask75 failures repaired', color='#3c9d8d')
        ax.bar(x+.18, [summary['damage'][t]['success_to_failure'] for t in dtags], .36, label='Native Mask75 successes damaged', color='#cc736d')
        ax.set_xticks(x, labels[1:], rotation=15); ax.set_ylabel('GT instances (official matching)'); ax.legend(fontsize=8)
        ax.set_title('Visible-instance repair and damage; not a GT-assisted inference method')
        save(fig, out/'03_repair_damage.png')
        fig, axes = plt.subplots(2, 6, figsize=(13, 4))
        candidate_ids = np.linspace(0, len(bank['maps'])-1, 12, dtype=int)
        for ax, k in zip(axes.flat, candidate_ids):
            value = bank['maps'][k].float().numpy()
            color = np.moveaxis((value[4:7]+1)/2, 0, -1)
            color *= value[0, :, :, None]
            ax.imshow(np.clip(color, 0, 1)); ax.set_title(f"{bank['identities'][k]['sample_id']}\n{bank['identities'][k]['rotation_degrees']} deg", fontsize=8); ax.axis('off')
        fig.suptitle('Frozen learned-body candidate maps; colors project learned features, not semantic IDs')
        save(fig, out/'04_learned_body_candidates.png')
        offsets = np.cumsum([0]+[r['rows'] for r in cache['dev']])
        fixed = np.linspace(0, len(cache['dev'])-1, 6, dtype=int)
        cases = []
        for number, i in enumerate(fixed):
            info = cache['dev'][int(i)]
            sample = torch.load(info['file'], map_location='cpu', weights_only=False)
            assert len(sample['eligible']), f'Fixed frame without eligible output: {i}'
            ordinal = int(sample['eligible'][0]); idx = int(offsets[i])+ordinal
            box = sample['det'][ordinal, :4].numpy(); row = sample['row']
            rgb = cv2.imread(row['image'])[:, :, ::-1].copy()
            masks = [cm.decode(p['segmentation']).astype(bool) for p in (native[idx], s[idx], q[idx])]
            x0, y0, x1, y1 = box
            gt_boxes = np.array(row['bboxes'])*640
            lo = gt_boxes[:, :2]-gt_boxes[:, 2:]/2; hi = gt_boxes[:, :2]+gt_boxes[:, 2:]/2
            inter = np.maximum(0, np.minimum(box[2:], hi)-np.maximum(box[:2], lo)).prod(1)
            overlap = inter/np.maximum(1e-8, (x1-x0)*(y1-y0)+gt_boxes[:, 2:].prod(1)-inter)
            g = int(overlap.argmax()); gt = cm.decode(row['rles'][g]) if overlap[g] >= .5 else np.zeros((640, 640), np.uint8)
            k = int(sample['bank_index'][0]); value = bank['maps'][k].float().numpy()
            fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
            panels = [crop(rgb, box), crop(overlay(rgb, masks[0], gt), box),
                      value[0], crop(overlay(rgb, masks[1], gt), box), crop(overlay(rgb, masks[2], gt), box)]
            titles = ['RGB / predicted box', 'Native visible mask', 'Aligned complete-body prior', 'Object S visible mask', 'Control Q visible mask']
            for ax, panel, title in zip(axes, panels, titles):
                ax.imshow(panel, cmap='magma' if panel.ndim == 2 else None, vmin=0 if panel.ndim == 2 else None,
                          vmax=1 if panel.ndim == 2 else None)
                ax.set_title(title, fontsize=9); ax.axis('off')
            fig.suptitle(f"Fixed frame {i}, image {row['image_id']}, output {int(sample['head_ids'][ordinal])}; orange=prediction, green=GT outline", fontsize=10)
            path = out/f'05_fixed_real_{number:02d}.png'; save(fig, path)
            cases.append(dict(development_ordinal=int(i), image_id=row['image_id'], detection_ordinal=ordinal,
                              native_output_detection_id=int(sample['head_ids'][ordinal]), candidate_index=k,
                              aligned_candidate=bank['identities'][k], figure=path.name, selection='six evenly spaced dev frames; first eligible output; no outcome selection'))
        dump(out/'SUMMARY.json', dict(source_run_id=source.name, figures=10, fixed_cases=cases,
                    bank_sha256=bank_pointer['bank_sha256'], actual_training_steps=8000,
                    figure_selection_does_not_use_outcomes=True, real_test_used=False,
                    scope='image-plane finite template alignment; complete prior distinguished from visible labels'))
        print('PRESENTATION_COMPLETE', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); execute(a.root.resolve())
