from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as cm
import torch

from common import dump, read
from proto_data import contact_target
from task_model import load, memory_module

COLORS = {'baseline': '#666666', 'continued': '#0072B2', 'adapter': '#E69F00',
          'random_memory': '#CC79A7', 'task_spm': '#009E73',
          'real': '#0072B2', 'synthetic': '#CC79A7', 'cutout': '#E69F00'}
LABELS = {'baseline': 'B* real-only', 'continued': 'B continued', 'adapter': 'G adapter',
          'random_memory': 'R random bank', 'task_spm': 'F task prototypes'}


def savefig(fig, out, name):
    fig.savefig(out/(name+'.png'), dpi=160, bbox_inches='tight', facecolor='white')
    fig.savefig(out/(name+'.svg'), bbox_inches='tight', facecolor='white'); plt.close(fig)


def overlay(rgb, masks):
    image = rgb.copy(); palette = np.asarray(plt.get_cmap('tab20').colors)*255
    for i, mask in enumerate(masks):
        mask = mask.astype(bool); color = palette[i % len(palette)]
        image[mask] = (.55*image[mask]+.45*color).astype(np.uint8)
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image, contours, -1, tuple(int(x) for x in color), 1)
    return image


def method(out):
    fig, ax = plt.subplots(figsize=(16, 6)); ax.set(xlim=(0, 16), ylim=(0, 6)); ax.axis('off')
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.08', facecolor=color, edgecolor='#506070'))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=10)
    def arrow(start, end):
        ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '->', 'color': '#506070', 'lw': 1.6})
    box(.2, 3.3, 2.3, 1.1, 'Frozen B* encoder\nP2 + upsampled P3\n512 x 160 x 160', '#e5edf5')
    box(3.1, 3.3, 2.2, 1.1, 'Train-only mean/std\nFrozen projector\n512 -> 256 -> 128', '#e5edf5')
    box(6, 3.3, 2.2, 1.1, '32 frozen centers\nFG/BG cosine contrast\ntanh + learned mix', '#d9f0e5')
    box(8.9, 3.3, 2, 1.1, '32 extra mask bases\nspatial center / RMS\n160 x 160', '#d9f0e5')
    box(8.9, 1.1, 2, 1.1, '32 original bases\nNative Proto26', '#e5edf5')
    box(11.5, 3.3, 2, 1.1, '64 spatial bases\n32 native + 32 extra\nconcatenate channels', '#d9f0e5')
    box(11.5, .9, 2, 1.2, '64 RGB coefficients\n32 native + 32 new\nnew rows start at 0', '#fff0d9')
    box(14.1, 2.3, 1.6, 1.2, 'Native O2O\nprocess_mask\nvisible instance', '#e5edf5')
    arrow((2.5, 3.85), (3.1, 3.85)); arrow((5.3, 3.85), (6, 3.85)); arrow((8.2, 3.85), (8.9, 3.85))
    arrow((10.9, 3.85), (11.5, 3.85)); arrow((10.9, 1.65), (11.5, 3.5))
    arrow((13.5, 3.85), (14.1, 3.1)); arrow((13.5, 1.5), (14.1, 2.65))
    ax.text(7.7, 5.25, 'V4.1 | instance-supervised memory + direct high-resolution mask readout', ha='center', fontsize=15, weight='bold')
    ax.text(4.2, 2.3, 'Pretraining: cross-view instance assignment\n+ contact separation + non-collapse\n5k real/cut -> 10k mixed -> 5k real', ha='center', fontsize=11)
    ax.text(12.5, .65, 'Predicted from head features', ha='center', fontsize=8)
    ax.text(8, .35, 'B / G / R / F: same 8,000 steps and 8 real + 2 cutout + 2 synthetic per batch. GT enters training/evaluation only.', ha='center', fontsize=10)
    savefig(fig, out, '00_method_and_supervision')


def training(root, out, sources, metrics, costs, decision):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for mode, source in sources.items():
        run = root/'runs'/source['run_id']; trace = read(run/'TRAINING.json'); ev = read(run/'EVALUATIONS.json')
        if mode != 'baseline':
            smooth = np.convolve([r['loss'] for r in trace], np.ones(50)/50, mode='valid')
            axes[0].plot(np.arange(len(smooth))+50, smooth, c=COLORS[mode], label=LABELS[mode])
            axes[1].plot([r['step'] for r in ev], [r['mask_ap'] for r in ev], '-o', ms=4, c=COLORS[mode], label=LABELS[mode])
    axes[0].set(xlabel='Matched method optimizer step', ylabel='Native loss (50-step mean)', title='Four-arm actual training')
    axes[1].axhline(metrics['baseline']['mask_ap'], c=COLORS['baseline'], ls=':', label='Selected B*')
    axes[1].set(xlabel='Matched method optimizer step', ylabel='Real-dev Mask AP (%)', title='All checkpoints retained')
    names = list(sources); bars = axes[2].bar([LABELS[k] for k in names], [metrics[k]['mask_ap'] for k in names], color=[COLORS[k] for k in names])
    axes[2].bar_label(bars, fmt='%.2f', padding=3); axes[2].tick_params(axis='x', rotation=25)
    axes[2].set(ylabel='Mask mAP50-95 (%)', title=f"F - B = {decision['measured_F_minus_continued_pp']:+.2f} pp | target +5 pp")
    axes[2].set_ylim(0, max(metrics[k]['mask_ap'] for k in names)*1.22)
    for a in axes[:2]: a.legend(frameon=False, fontsize=7)
    fig.suptitle('432 real development images | seed 0 | development selection, no independent test')
    fig.tight_layout(); savefig(fig, out, '01_training_and_primary_results')
    baseline = root/'runs'/sources['baseline']['run_id']; ev = read(baseline/'EVALUATIONS.json'); trace = read(baseline/'TRAINING.json')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    smooth = np.convolve([r['loss'] for r in trace], np.ones(50)/50, mode='valid')
    axes[0].plot((np.arange(len(smooth))+50)/106, smooth, c=COLORS['baseline'])
    axes[0].set(xlabel='Full real-data epoch (1272 unique frames)', ylabel='Native loss (50-step mean)', title='120-epoch B* training')
    for key, label in [('mask_ap', 'Mask AP'), ('mask_ap75', 'Mask AP75'), ('box_ap', 'Box AP')]:
        axes[1].plot([r['epoch'] for r in ev], [r[key] for r in ev], '-o', label=label)
    axes[1].legend(frameon=False); axes[1].set(xlabel='Full real-data epoch', ylabel='Real-dev AP (%)', title='Early and late checkpoints reported')
    fig.tight_layout(); savefig(fig, out, '02_full_baseline_learning')
    traces = []
    for stage in ['prototype_warm', 'prototype_final']:
        run = root/'runs'/read(root/(stage.upper()+'_POINTER.json'))['run_id']; traces += read(run/'TRAINING.json')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for key in ['loss', 'instance_mask_alignment', 'foreground', 'contact']:
        sm = np.convolve([r[key] for r in traces], np.ones(100)/100, mode='valid')
        axes[0].plot(np.arange(len(sm))+100, sm, label=key)
    axes[0].legend(frameon=False, fontsize=8); axes[0].set(ylabel='Loss (100-step mean)', title='20,000 actual projector steps')
    for key in ['correspondence', 'instance_pull', 'contact_push']:
        sm = np.convolve([r[key] for r in traces], np.ones(100)/100, mode='valid')
        axes[1].plot(np.arange(len(sm))+100, sm, label=key)
    axes[1].legend(frameon=False, fontsize=8); axes[1].set(title='Instance / correspondence supervision')
    axes[2].plot([r['step'] for r in traces], [r['gradient_norm'] for r in traces], lw=.5)
    axes[2].set(ylabel='Gradient norm before clipping', title='Real gradient signal')
    for a in axes:
        a.axvline(5000, ls=':', c='#555'); a.axvline(15000, ls=':', c='#555'); a.set_xlabel('Global projector optimizer step')
    fig.tight_layout(); savefig(fig, out, '03_prototype_training_phases')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    variants = ['task_spm']+['task_spm_'+v for v in ['off', 'permute_channels', 'shuffle_spatial', 'random_dictionary']]
    bars = axes[0].bar(['F normal', 'Off', 'Channel shift', 'Spatial shift', 'Random keys'], [metrics[k]['mask_ap'] for k in variants], color=['#009E73']+['#999999']*4)
    axes[0].bar_label(bars, fmt='%.2f', fontsize=8); axes[0].tick_params(axis='x', rotation=30)
    axes[0].set(ylabel='Real-dev Mask AP (%)', title='Same F checkpoint, no retraining')
    x = np.arange(len(names))
    axes[1].bar(x-.18, [metrics[k]['mask_ap75'] for k in names], .36, label='Mask AP75')
    axes[1].bar(x+.18, [metrics[k]['box_ap'] for k in names], .36, label='Box AP')
    axes[1].set_xticks(x, [LABELS[k] for k in names], rotation=25); axes[1].legend(frameon=False); axes[1].set_title('Quality and detection damage')
    axes[2].bar([LABELS[k] for k in names], [costs[k]['median_inference_and_mask_decode_ms'] for k in names], color=[COLORS[k] for k in names])
    axes[2].tick_params(axis='x', rotation=25); axes[2].set(ylabel='FP32 inference + decode median (ms)', title='Same GPU, 24 images, synchronized')
    fig.tight_layout(); savefig(fig, out, '04_use_ablation_quality_and_cost')


def representative_scenes(recipes):
    layouts = ['scatter', 'feeder_cluster', 'fence_strip', 'rest_rows']
    floors = ['concrete', 'slatted', 'stained_concrete', 'bedded']
    compositions = ['active', 'resting', 'feeding_rest', 'mixed']
    selected = []
    for i in range(8):
        wanted = {'layout': layouts[i % 4], 'floor_type': floors[(i//2) % 4],
                  'density_bin': i % 4, 'camera_family': 'overhead' if i % 2 == 0 else 'oblique',
                  'composition': compositions[i % 4]}
        selected.append(next(r for r in recipes if all(r[k] == value for k, value in wanted.items())))
    return selected


def data_figures(root, out, store):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2)); counts = Counter(r['domain']+'/'+r['split'] for r in store.rows)
    axes[0].barh(list(counts), list(counts.values()), color='#0072B2'); axes[0].set(xlabel='Cached RGB images', title='Actual full input coverage')
    for domain in ['real', 'synthetic', 'cutout']:
        values = [max(box[2:]) for r in store.rows if r['domain'] == domain and r['split'] == 'train' for box in r['bboxes']]
        axes[1].hist(values, bins=np.linspace(0, .9, 30), density=True, histtype='step', lw=2, label=domain, color=COLORS[domain])
    axes[1].legend(frameon=False); axes[1].set(xlabel='Box long axis / 640 canvas', title='Actual training instance scales')
    poses = Counter()
    for shard in range(3):
        child = root/'render_plans'/f'shard_{shard}'; run = child/'runs'/read(child/'FORMAL_POINTER.json')['run_id']
        poses.update(read(run/'SUMMARY.json')['pose_instances_once_per_geometry'])
    axes[2].barh(list(poses), list(poses.values()), color='#009E73'); axes[2].set(xlabel='Pig instances (one count / geometry)', title='2048 geometries, 4096 appearances')
    fig.tight_layout(); savefig(fig, out, '05_data_coverage_and_poses')
    recipes = read(root/'SCENES.json')['formal']; by_geometry = {r['group']: r for r in store.rows if r['domain'] == 'synthetic' and r['appearance_id'] == 0}
    selected = [(recipe, by_geometry[recipe['geometry_id']]) for recipe in representative_scenes(recipes)]
    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 2.6*len(selected)), squeeze=False)
    for i, (recipe, row) in enumerate(selected):
        rgb = store.get(row)['rgb'][:, :, ::-1]; paired = next(r for r in store.rows if r['domain'] == 'synthetic' and r['group'] == row['group'] and r['appearance_id'] == 1)
        for j, image in enumerate([rgb, overlay(rgb, [cm.decode(e) for e in row['rles']]), store.get(paired)['rgb'][:, :, ::-1]]):
            axes[i, j].imshow(image); axes[i, j].axis('off')
        axes[i, 0].set_title(f"{row['group']} | {recipe['layout']} | {recipe['floor_type']} | {recipe['requested_pigs']} pigs | {recipe['camera_family']}", fontsize=8)
        if i == 0:
            axes[i, 1].set_title('Exact visible instance masks'); axes[i, 2].set_title('Same geometry, second appearance')
    fig.tight_layout(); savefig(fig, out, '06_actual_synthetic_scene_and_label_pairs')


def memory_figures(root, out, store):
    run = root/'runs'/read(root/'PROTOTYPE_FINAL_POINTER.json')['run_id']
    memory = torch.load(run/'memory.pt', map_location='cpu', weights_only=False)
    embedded = torch.load(run/'embedding.pt', map_location='cpu', weights_only=False); clusters = memory['clusters']
    by_source = {r['source_id']: r for r in store.rows}
    for part, start in [('foreground', 0), ('background', 16)]:
        canvas = Image.new('RGB', (1280, 962), '#f3f5f7'); draw = ImageDraw.Draw(canvas)
        draw.font = ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'), 12)
        draw.text((14, 12), f'{part.upper()} | learned 16 centers | actual train exemplars | red cross = stride-4 query', fill='#14212c')
        for j, cluster in enumerate(clusters[start:start+16]):
            left, top = (j % 4)*320, 45+(j//4)*228
            draw.text((left+6, top), f"P{cluster['prototype']:02d} | support {cluster['support_tokens']}", fill='#14212c')
            for k, exemplar in enumerate(cluster['exemplars'][:3]):
                row = by_source[exemplar['source_id']]; rgb = store.get(row)['rgb'][:, :, ::-1]
                x, y = exemplar['x']*4+2, exemplar['y']*4+2
                padded = cv2.copyMakeBorder(rgb, 48, 48, 48, 48, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                crop = Image.fromarray(padded[y:y+96, x:x+96].copy()).resize((100, 100))
                mark = ImageDraw.Draw(crop); mark.line((44, 50, 56, 50), fill='red', width=1); mark.line((50, 44, 50, 56), fill='red', width=1)
                canvas.paste(crop, (left+5+k*104, top+24)); draw.text((left+5+k*104, top+128), f"{exemplar['domain'][:3]} {exemplar['stratum']}", fill='#253646')
                draw.text((left+5+k*104, top+144), f"cos {exemplar['cosine']:.3f}", fill='#253646')
            draw.text((left+6, top+170), 'real/syn/cut '+ '/'.join(str(cluster['domain_counts'].get(k, 0)) for k in ['real', 'synthetic', 'cutout']), fill='#253646')
            draw.text((left+6, top+188), 'C/B/I/BG '+ '/'.join(str(cluster['stratum_counts'].get(k, 0)) for k in ['C', 'B', 'I', 'BG']), fill='#253646')
        canvas.save(out/f'07_prototype_atlas_{part}.png')
    fig, ax = plt.subplots(figsize=(13, 4)); bottom = np.zeros(32)
    for domain in ['real', 'synthetic', 'cutout']:
        values = np.array([c['domain_counts'][domain] for c in clusters]); ax.bar(range(32), values, bottom=bottom, label=domain, color=COLORS[domain]); bottom += values
    ax.axvline(15.5, c='#555', ls=':'); ax.legend(frameon=False); ax.set(xlabel='Prototype ID: FG 0-15 / BG 16-31', ylabel='Supporting train tokens', title='Actual slot usage and training provenance')
    savefig(fig, out, '08_prototype_support')
    z = embedded['embedding'].numpy(); centers = memory['centers'].numpy(); selected = np.random.default_rng(0).choice(len(z), min(7000, len(z)), replace=False)
    sample = z[selected]; mean = sample.mean(0); _, _, vv = np.linalg.svd(sample-mean, full_matrices=False)
    xy, pc = (sample-mean)@vv[:2].T, (centers-mean)@vv[:2].T
    ends = np.asarray([e['stop'] for e in embedded['episodes']]); domains, strata = [], []
    for token in selected:
        ep = embedded['episodes'][int(np.searchsorted(ends, token, side='right'))]
        domains.append(ep['domain']); strata.append(ep['strata'][token-ep['start']])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, labels, colors in [(axes[0], domains, {d: COLORS[d] for d in ['real', 'synthetic', 'cutout']}),
                               (axes[1], strata, {'C': '#D55E00', 'B': '#E69F00', 'I': '#009E73', 'BG': '#999999'})]:
        labels = np.asarray(labels)
        for name, color in colors.items():
            idx = labels == name; ax.scatter(xy[idx, 0], xy[idx, 1], s=3, alpha=.25, c=color, label=name, rasterized=True)
        ax.scatter(pc[:, 0], pc[:, 1], marker='X', c='black', s=45)
        for i, point in enumerate(pc): ax.annotate(str(i), point, fontsize=7)
        ax.legend(frameon=False, markerscale=3); ax.set(xlabel='PC1', ylabel='PC2')
    axes[0].set_title('Actual learned train embedding by domain'); axes[1].set_title('Same embedding by supervised stratum')
    fig.tight_layout(); savefig(fig, out, '09_embedding_PCA')
    del embedded, z


def cases(root, run, out, store):
    grouped = {}
    for mode in ['continued', 'task_spm']:
        predictions = read(run/'evaluation'/(mode+'_predictions.json')); values = defaultdict(list)
        for prediction in predictions:
            if prediction['score'] >= .25: values[prediction['image_id']].append(prediction)
        grouped[mode] = values
    all_cases = []
    for row in store.rows:
        if row['domain'] != 'real' or row['split'] != 'dev': continue
        values = {}
        for mode in grouped:
            predictions = grouped[mode][row['image_id']]
            overlaps = cm.iou([p['segmentation'] for p in predictions], row['rles'], [0]*len(row['rles'])) if predictions else np.zeros((0, len(row['rles'])))
            values[mode] = float(overlaps.max(0).mean()) if predictions else 0.
        all_cases.append({'source_id': row['source_id'], 'image_id': row['image_id'], **values, 'delta': values['task_spm']-values['continued']})
    ordered = sorted(all_cases, key=lambda c: c['delta'], reverse=True); chosen = ordered[:3]+list(reversed(ordered[-3:]))
    dump(out/'CASE_SELECTION.json', {'metric': 'mean_best_GT_mask_IoU_at_score_0.25_for_display_only_not_AP', 'all_cases': ordered})
    by_source = {r['source_id']: r for r in store.rows}; modelrun = root/'runs'/read(root/'TASK_SPM_POINTER.json')['run_id']
    model = load(modelrun/'best.pt').eval(); module = memory_module(model); module.capture = True
    figure, axes = plt.subplots(6, 4, figsize=(13, 18)); zoomfig, zoomaxes = plt.subplots(6, 4, figsize=(12, 15))
    heatfig, heataxes = plt.subplots(6, 5, figsize=(16, 17)); logitfig, logitaxes = plt.subplots(6, 5, figsize=(16, 17)); records = []
    for i, case in enumerate(chosen):
        row = by_source[case['source_id']]; sample = store.get(row); rgb = sample['rgb'][:, :, ::-1]
        panels = [rgb, overlay(rgb, [cm.decode(r) for r in row['rles']])]
        for mode in ['continued', 'task_spm']:
            panels.append(overlay(rgb, [cm.decode(p['segmentation']) for p in grouped[mode][row['image_id']]]))
        fg, contact, boundary, valid = contact_target(sample)
        target, cropkind = (contact, 'contact') if contact.any() else ((boundary, 'visible boundary, no contact') if boundary.any() else (fg, 'foreground, no contact'))
        y, x = np.unravel_index(cv2.GaussianBlur(target, (11, 11), 0).argmax(), target.shape)
        x0, y0 = min(480, max(0, int(x*4)-80)), min(480, max(0, int(y*4)-80))
        for j, panel in enumerate(panels):
            axes[i, j].imshow(panel); axes[i, j].axis('off'); zoomaxes[i, j].imshow(panel[y0:y0+160, x0:x0+160]); zoomaxes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(['RGB', 'Visible GT', 'B continued >= .25', 'F prototypes >= .25'][j], fontsize=10)
                zoomaxes[i, j].set_title(['RGB crop', 'Visible GT', 'B continued', 'F prototypes'][j])
        axes[i, 0].text(0, -.035, f"{row['source_id']} | delta {case['delta']:+.4f}", transform=axes[i, 0].transAxes, fontsize=7)
        zoomaxes[i, 0].text(0, -.04, cropkind, transform=zoomaxes[i, 0].transAxes, fontsize=8)
        with torch.no_grad():
            result = model(store.batch([row], 0, augment=False)['img']); det, proto = result[0]
            proto = proto[0] if isinstance(proto, tuple) else proto; det = det[0]
            detection = int(det[:, 4].argmax()); coefficient = det[detection, 6:]; bases = proto[0]
            native = torch.einsum('c,chw->hw', coefficient[:32], bases[:32]).cpu().numpy()
            extra = torch.einsum('c,chw->hw', coefficient[32:], bases[32:]).cpu().numpy()
        maps = {k: v.numpy() for k, v in module.maps.items() if k != 'query'}
        np.savez_compressed(out/f"response_{row['image_id']}.npz", **maps, native_logits=native, extra_logits=extra,
                            coefficients=coefficient.cpu().numpy(), output_detection_id=detection)
        heat = [rgb, maps['foreground_similarity'][0], maps['background_similarity'][0], maps['top1'][0], np.sqrt((maps['extra_bases'][0]**2).mean(0))]
        for j, value in enumerate(heat):
            a = heataxes[i, j]; im = a.imshow(value, **({'cmap': 'tab20', 'vmin': 0, 'vmax': 31} if j == 3 else {'cmap': 'viridis'} if j else {})); a.axis('off')
            if j: heatfig.colorbar(im, ax=a, fraction=.035, pad=.02)
            if i == 0: a.set_title(['RGB', 'Best foreground cosine', 'Best background cosine', 'Top-1 prototype ID', 'Extra-basis RMS'][j], fontsize=9)
        total = native+extra; scale = max(float(np.abs(native).max()), float(np.abs(extra).max()), float(np.abs(total).max()), 1.)
        logitpanels = [rgb, native, extra, total, 1/(1+np.exp(-np.clip(total, -60, 60)))]
        for j, value in enumerate(logitpanels):
            a = logitaxes[i, j]; opts = {} if j == 0 else ({'cmap': 'viridis', 'vmin': 0, 'vmax': 1} if j == 4 else {'cmap': 'coolwarm', 'vmin': -scale, 'vmax': scale})
            im = a.imshow(value, **opts); a.axis('off')
            if j: logitfig.colorbar(im, ax=a, fraction=.035, pad=.02)
            if i == 0: a.set_title(['RGB', 'Original 32 logit sum', 'New 32 logit sum', 'Actual combined logit', 'Sigmoid, before box crop'][j], fontsize=9)
        records.append({'source_id': row['source_id'], 'crop_kind': cropkind, 'crop_xywh': [x0, y0, 160, 160],
                        'output_detection_id': detection, 'selection': 'highest_final_detection_score_for_logit_display',
                        'score': float(det[detection, 4]), 'native_logit_rms': float(np.sqrt((native**2).mean())),
                        'extra_logit_rms': float(np.sqrt((extra**2).mean()))})
    for fig, name, title in [(figure, '10_real_improvement_and_damage_cases', 'Highest 3 / lowest 3 descriptive IoU deltas, fixed display rule'),
                              (zoomfig, '11_contact_or_explicit_boundary_crops', 'Actual contact crops; boundary fallback labeled'),
                              (heatfig, '12_actual_prototype_responses', 'Frozen learned keys, 160 x 160 query, actual model response'),
                              (logitfig, '13_actual_mask_logit_contribution', 'Same F detection: native / prototype contribution, unthresholded, uncropped')]:
        fig.suptitle(title, fontsize=12); fig.tight_layout(); savefig(fig, out, name)
    dump(out/'RESPONSE_CASES.json', records); module.capture = False; del model; torch.cuda.empty_cache()


def family_figure(out, family):
    groups = family['families']; fig, ax = plt.subplots(figsize=(11, 4.5)); y = np.arange(len(groups))
    values = list(groups.values())
    for shift, mode, label, color in [(-.18, 'continued', 'F - matched B', '#0072B2'), (.18, 'baseline', 'F - B* real-only', '#E69F00')]:
        delta = [v['metrics']['task_spm']['mask_ap']-v['metrics'][mode]['mask_ap'] for v in values]
        ax.barh(y+shift, delta, height=.32, label=label, color=color)
    ax.set_yticks(y, [f"Family {i+1} (n={v['images']})" for i, v in enumerate(values)]); ax.axvline(0, c='#555', lw=1)
    ax.set(xlabel='Real-dev Mask AP difference (absolute pp)', title='All 4 inferred filename families | descriptive, no physical-pen claim')
    ax.legend(frameon=False); fig.tight_layout(); savefig(fig, out, '14_all_family_damage_breakdown')


def figures(root, run, store, sources, metrics, costs, family, decision):
    out = run/'figures'; out.mkdir(); plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    method(out); training(root, out, sources, metrics, costs, decision); data_figures(root, out, store)
    memory_figures(root, out, store); cases(root, run, out, store); family_figure(out, family)
