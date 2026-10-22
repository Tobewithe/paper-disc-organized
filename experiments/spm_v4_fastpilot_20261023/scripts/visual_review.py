"""CPU-only figure review: meaningful fallback crops and grouped descriptive scores."""
import argparse
import contextlib
import io
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as cm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import dump, read, run_context, sha
from data import Store, contact_target
from visualize import overlay, savefig


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); args = p.parse_args(); root = Path(args.root)
    with run_context(root, 'visual_review') as (run, meta):
        meta['visual_review_source_sha256'] = sha(Path(__file__))
        (run/'VISUAL_REVIEW_SOURCE.py').write_bytes(Path(__file__).read_bytes())
        out = run/'figures'; out.mkdir(); store = Store(read(root/'DATASET.json')['rows'])
        by_source = {r['source_id']: r for r in store.rows}
        viz = root/'runs'/read(root/'VISUALIZE_POINTER.json')['run_id']
        ordered = read(viz/'figures/CASE_SELECTION.json')['all_cases']; selected = ordered[:3]+list(reversed(ordered[-3:]))
        predictions = {}; grouped = {}
        for mode in ['baseline', 'continued', 'spm']:
            folder = root/'runs'/read(root/(mode.upper()+'_POINTER.json'))['run_id']; step = read(folder/'SUMMARY.json')['selected']['step']
            predictions[mode] = read(folder/'evaluation'/f'step_{step:04d}_predictions.json')
            grouped[mode] = {}
            for item in predictions[mode]:
                if item['score'] >= .25:
                    grouped[mode].setdefault(item['image_id'], []).append(item)
        fig, axes = plt.subplots(6, 4, figsize=(12, 15)); crop_record = []
        for i, case in enumerate(selected):
            row = by_source[case['source_id']]; sample = store.get(row); rgb = sample['rgb'][:, :, ::-1]
            gt = [cm.decode(e) for e in row['rles']]
            panels = [rgb, overlay(rgb, gt)]+[overlay(rgb, [cm.decode(p['segmentation']) for p in grouped[m].get(row['image_id'], [])]) for m in ['continued', 'spm']]
            fg, contact, boundary, valid = contact_target(sample)
            if contact.any():
                target = contact; kind = 'contact'
            elif boundary.any():
                target = boundary; kind = 'pig-background edge (no C)'
            else:
                target = fg; kind = 'foreground (no C)'
            y, x = np.unravel_index(cv2.GaussianBlur(target, (11, 11), 0).argmax(), target.shape)
            x0, y0 = min(480, max(0, int(x*8)-80)), min(480, max(0, int(y*8)-80))
            crop_record.append({'source_id': row['source_id'], 'kind': kind, 'xywh': [x0, y0, 160, 160]})
            for j, panel in enumerate(panels):
                axes[i, j].imshow(panel[y0:y0+160, x0:x0+160]); axes[i, j].axis('off')
                if i == 0:
                    axes[i, j].set_title(['RGB crop', 'Visible GT', 'B* continued', 'F SPM'][j], fontsize=10)
            axes[i, 0].text(0, -.05, f"{row['source_id']} | {kind}", transform=axes[i, 0].transAxes, fontsize=8)
        fig.suptitle('Contact / boundary detail | 3 improved + 3 degraded cases | display score >= .25', fontsize=12)
        fig.tight_layout(); fig.savefig(out/'08_contact_and_boundary_zoom.png', dpi=160, bbox_inches='tight'); plt.close(fig)
        groups = sorted({r['group'] for r in store.rows if r['domain'] == 'real' and r['split'] == 'dev'})
        result = []
        for group in groups:
            ids = [r['image_id'] for r in store.rows if r['domain'] == 'real' and r['split'] == 'dev' and r['group'] == group]
            values = {}
            for mode in predictions:
                with contextlib.redirect_stdout(io.StringIO()):
                    gt = COCO(str(root/'DEV_COCO.json'))
                    dt = gt.loadRes([{k: v for k, v in item.items() if k != 'bbox'} for item in predictions[mode]])
                    evaluator = COCOeval(gt, dt, 'segm'); evaluator.params.imgIds = ids
                    evaluator.params.maxDets = [1, 10, 100]; evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
                values[mode] = float(evaluator.stats[0]*100)
            result.append({'inferred_filename_family': group, 'images': len(ids), **values,
                           'spm_minus_continued_pp': values['spm']-values['continued'],
                           'spm_minus_baseline_pp': values['spm']-values['baseline']})
        dump(run/'GROUPED_DEV_RESULTS.json', result); dump(run/'CROP_SELECTION.json', crop_record)
        fig, ax = plt.subplots(figsize=(10, 4)); pos = np.arange(len(result))
        ax.barh(pos-.17, [r['spm_minus_continued_pp'] for r in result], height=.3, label='F - continued', color='#0072B2')
        ax.barh(pos+.17, [r['spm_minus_baseline_pp'] for r in result], height=.3, label='F - B* real-only', color='#E69F00')
        ax.axvline(0, c='#777', lw=1); ax.set_yticks(pos, [f"Family {i+1} (n={r['images']})" for i, r in enumerate(result)])
        ax.set(xlabel='Development Mask AP difference (pp)', title='Descriptive family breakdown | inferred groups, no physical-pen claim')
        ax.legend(frameon=False); fig.tight_layout(); savefig(fig, out, '11_grouped_development_effect')
        dump(run/'SUMMARY.json', {'figures_reviewed': 12, 'crop_fallback_explicit': True, 'grouped_results': result,
                                  'training_steps': 0, 'scope': 'descriptive_development_breakdown_not_independent_validation'})
        dump(run/'DECISION.json', {'status': 'visual_review_completed', 'next_action': 'deliver_existing_winning_checkpoint'})


if __name__ == '__main__':
    main()
