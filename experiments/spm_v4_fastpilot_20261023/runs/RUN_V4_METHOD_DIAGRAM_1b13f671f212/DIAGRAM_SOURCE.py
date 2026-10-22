from collections import Counter
import hashlib
from pathlib import Path
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import FancyBboxPatch
import torch
from pycocotools import mask as cm

from common import dump, read, run_context, sha
from data import contact_target
from evaluate import evaluate, predict
from model import load
from train import stage_checkpoint

COLORS = {'baseline': '#555555', 'continued': '#0072B2', 'adapter': '#E69F00', 'spm': '#009E73',
          'real': '#0072B2', 'synthetic': '#CC79A7', 'cutout': '#E69F00'}
LABELS = {'baseline': 'B* real-only', 'continued': 'B* continued', 'adapter': 'G adapter', 'spm': 'F SPM'}


def savefig(fig, out, name):
    fig.savefig(out/(name+'.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(out/(name+'.svg'), bbox_inches='tight', facecolor='white'); plt.close(fig)


def method_figure(out):
    fig, ax = plt.subplots(figsize=(15, 5)); ax.set(xlim=(0, 15), ylim=(0, 5)); ax.axis('off')
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', facecolor=color, edgecolor='#577081', lw=1.2))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=10)
    def arrow(a, b):
        ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '->', 'color': '#465a68', 'lw': 1.7})
    box(.2, 2.5, 2, 1, 'B* frozen encoder\nP3: 256 x 80 x 80', '#e7eef5')
    box(2.8, 2.5, 2.2, 1, 'Frozen normalization\n128D projector + L2', '#e7eef5')
    box(5.6, 3, 2.4, 1, 'FG cosine Top-4\n16 frozen keys / values', '#d9f0e5')
    box(5.6, 1.5, 2.4, 1, 'BG cosine Top-4\n16 frozen keys / values', '#f7e6d3')
    box(8.6, 2.5, 2, 1, 'Concatenate\ntrainable 256D readout', '#d5eaf7')
    box(11.2, 2.5, 1.8, 1, 'RMS cap x gate\nx learned alpha', '#d5eaf7')
    box(13.6, 2.5, 1.1, 1, 'Native\nYOLO head', '#d5eaf7')
    arrow((2.2, 3), (2.8, 3)); arrow((5, 3), (5.6, 3.5)); arrow((5, 3), (5.6, 2))
    arrow((8, 3.5), (8.6, 3)); arrow((8, 2), (8.6, 3)); arrow((10.6, 3), (11.2, 3)); arrow((13, 3), (13.6, 3))
    ax.plot([1.2, 1.2, 13.2, 13.2], [2.5, .65, .65, 3], color='#577081', lw=1.6)
    arrow((13.2, 3), (13.6, 3)); ax.text(7.3, .8, 'Residual skip: original P3 + learned correction', ha='center', fontsize=10)
    ax.text(7.5, 4.65, 'SPM-YOLO V4 | implemented retrieval and fusion path', ha='center', fontsize=15, weight='bold')
    ax.text(7.5, .1, 'Frozen bank / query coordinates; alpha starts at 0; native O2M supplies fusion gradients; native O2O output and mask decoder retained.', ha='center', fontsize=9)
    savefig(fig, out, '00_method_overview')


def overview(root, out, store, summaries):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for mode in LABELS:
        folder = root/'runs'/read(root/(mode.upper()+'_POINTER.json'))['run_id']
        trace = read(folder/'TRAINING.json'); smooth = np.convolve([r['loss'] for r in trace], np.ones(25)/25, mode='valid')
        axes[0].plot(np.arange(len(smooth))+25, smooth, color=COLORS[mode], label=LABELS[mode])
        ev = read(folder/'EVALUATIONS.json')
        axes[1].plot([r['step'] for r in ev], [r['mask_ap'] for r in ev], marker='o', color=COLORS[mode], label=LABELS[mode])
    axes[0].set(xlabel='Optimizer step within stage', ylabel='Native weighted loss (25-step mean)', title='Actual training curves')
    axes[1].set(xlabel='Optimizer step within stage', ylabel='Development Mask AP (%)', title='Identical selection rule')
    modes = list(LABELS); values = [summaries[m]['selected']['mask_ap'] for m in modes]
    bars = axes[2].bar([LABELS[m] for m in modes], values, color=[COLORS[m] for m in modes])
    axes[2].bar_label(bars, fmt='%.2f', padding=3); axes[2].set(ylabel='Mask AP (%)', title='Selected development checkpoints')
    axes[2].tick_params(axis='x', rotation=22); axes[2].set_ylim(0, max(values)*1.18+1)
    for ax in axes[:2]:
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('SPM-YOLO V4 | single seed, 54 real development images, 640 canvas', fontsize=13)
    fig.tight_layout(); savefig(fig, out, '01_training_and_results')
    prun = root/'runs'/read(root/'PROTOTYPE_POINTER.json')['run_id']
    trace = read(prun/'TRAINING.json'); fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for key in ['loss', 'fg', 'contact', 'consistency']:
        ax[0].plot([r['step'] for r in trace], [r[key] for r in trace], label=key)
    ax[0].legend(frameon=False); ax[0].set(xlabel='Projector optimization step', title='Independent memory training')
    ax[1].plot([r['gradient_norm'] for r in trace], color=COLORS['spm']); ax[1].set(xlabel='Projector optimization step', ylabel='Gradient norm', title='Actual gradient signal')
    fig.tight_layout(); savefig(fig, out, '02_prototype_training')
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    counts = Counter(r['domain']+'/'+r['split'] for r in store.rows)
    names = sorted(counts); ax[0].barh(names, [counts[k] for k in names], color='#0072B2'); ax[0].set(title='Distinct cached frames', xlabel='Image count')
    for domain in ['real', 'synthetic', 'cutout']:
        sizes = [max(b[2:]) for r in store.rows if r['domain'] == domain and r['split'] == 'train' for b in r['bboxes']]
        ax[1].hist(sizes, bins=np.linspace(0, .6, 25), density=True, histtype='step', lw=2, label=domain, color=COLORS[domain])
    ax[1].legend(frameon=False); ax[1].set(xlabel='Box long axis / common 640 canvas', ylabel='Density', title='Training object scale')
    render = root/'runs'/read(root/'FORMAL_POINTER.json')['run_id']; summary = read(render/'SUMMARY.json')
    poses = summary['pose_instances_once_per_geometry']; ax[2].barh(list(poses), list(poses.values()), color='#009E73')
    ax[2].set(title='Rendered poses (one count / geometry)', xlabel='Pig instances')
    fig.tight_layout(); savefig(fig, out, '03_data_distribution')


def memory_figures(root, store, out):
    prun = root/'runs'/read(root/'PROTOTYPE_POINTER.json')['run_id']
    memory = torch.load(prun/'memory.pt', map_location='cpu', weights_only=False)
    embedding = torch.load(prun/'embedding.pt', map_location='cpu', weights_only=False)
    clusters = memory['clusters']
    for part, start in [('fg', 0), ('bg', 16)]:
        canvas = Image.new('RGB', (1280, 4*234+50), '#f3f5f7'); draw = ImageDraw.Draw(canvas)
        font_path = Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
        draw.font = ImageFont.truetype(str(font_path), 12)
        draw.text((18, 14), f'{part.upper()} memory | actual top exemplars | 16 slots | cross marks queried P3 cell', fill='#14212c')
        for j, cluster in enumerate(clusters[start:start+16]):
            left, top = (j % 4)*320, 50+(j//4)*234
            draw.text((left+8, top+4), f"P{cluster['prototype']:02d}  n={cluster['support_tokens']}  groups={cluster['unique_groups']}", fill='#14212c')
            for k, exemplar in enumerate(cluster['exemplars'][:3]):
                row = store.rows[exemplar['ordinal']]; rgb = store.get(row)['rgb'][:, :, ::-1]
                x, y = exemplar['x']*8+4, exemplar['y']*8+4
                # Keep the marked token at the crop center, including edge cases.
                padded = cv2.copyMakeBorder(rgb, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                crop = Image.fromarray(padded[y+16:y+112, x+16:x+112].copy()).resize((100, 100))
                mark = ImageDraw.Draw(crop); mark.line((45, 50, 55, 50), fill='red', width=1); mark.line((50, 45, 50, 55), fill='red', width=1)
                canvas.paste(crop, (left+6+k*104, top+30))
                draw.text((left+6+k*104, top+133), f"{exemplar['domain'][:3]} {exemplar['stratum']}", fill='#253646')
                draw.text((left+6+k*104, top+149), f"cos {exemplar['cosine']:.3f}", fill='#253646')
            domains = cluster['domain_counts']
            draw.text((left+8, top+174), 'real/syn/cut: '+ '/'.join(str(domains.get(k, 0)) for k in ['real', 'synthetic', 'cutout']), fill='#253646')
            draw.text((left+8, top+192), 'C/B/I/BG: '+ '/'.join(str(cluster['stratum_counts'].get(k, 0)) for k in ['C', 'B', 'I', 'BG']), fill='#253646')
        canvas.save(out/f'04_prototype_atlas_{part}.png')
    fig, ax = plt.subplots(figsize=(13, 3.8)); bottom = np.zeros(32)
    for domain in ['real', 'synthetic', 'cutout']:
        counts = np.array([c['domain_counts'].get(domain, 0) for c in clusters])
        ax.bar(range(32), counts, bottom=bottom, color=COLORS[domain], label=domain); bottom += counts
    ax.axvline(15.5, color='#555', ls='--'); ax.legend(frameon=False); ax.set(xlabel='Prototype ID (0-15 foreground; 16-31 background)', ylabel='Supporting train tokens', title='Memory provenance and slot utilization')
    savefig(fig, out, '05_memory_support')
    z = embedding['embedding'].numpy(); centers = memory['centers'].numpy(); mean = z.mean(0)
    _, _, vv = np.linalg.svd(z[np.random.default_rng(0).choice(len(z), min(len(z), 6000), replace=False)]-mean, full_matrices=False)
    xy, pc = (z-mean)@vv[:2].T, (centers-mean)@vv[:2].T
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    chosen = np.random.default_rng(0).choice(len(z), min(7000, len(z)), replace=False)
    records = embedding['records']
    for domain in ['real', 'synthetic', 'cutout']:
        idx = [i for i in chosen if records[i]['domain'] == domain]
        ax[0].scatter(xy[idx, 0], xy[idx, 1], s=3, alpha=.25, color=COLORS[domain], label=domain, rasterized=True)
    for label, color in [('C', '#D55E00'), ('B', '#E69F00'), ('I', '#009E73'), ('BG', '#999999')]:
        idx = [i for i in chosen if records[i]['stratum'] == label]
        ax[1].scatter(xy[idx, 0], xy[idx, 1], s=3, alpha=.25, color=color, label=label, rasterized=True)
    for a in ax:
        a.scatter(pc[:, 0], pc[:, 1], marker='X', s=50, c='black')
        for k, point in enumerate(pc):
            a.annotate(str(k), point, fontsize=7)
        a.legend(frameon=False, markerscale=3); a.set(xlabel='PC1', ylabel='PC2')
    ax[0].set_title('Learned embedding by training domain'); ax[1].set_title('Same embedding by supervised stratum')
    fig.tight_layout(); savefig(fig, out, '06_embedding_PCA')
    return memory


def overlay(rgb, masks):
    image = rgb.copy()
    palette = np.asarray(plt.get_cmap('tab20').colors)*255
    for i, mask in enumerate(masks):
        m = mask.astype(bool); color = palette[i % len(palette)]
        image[m] = (.55*image[m]+.45*color).astype(np.uint8)
        contours = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image, contours, -1, tuple(int(x) for x in color), 1)
    return image


def comparison_cases(root, store, out, model):
    rows = [r for r in store.rows if r['domain'] == 'real' and r['split'] == 'dev']
    sources = {}
    for mode in ['continued', 'spm']:
        folder = root/'runs'/read(root/(mode.upper()+'_POINTER.json'))['run_id']
        selected = read(folder/'SUMMARY.json')['selected']['step']
        sources[mode] = read(folder/'evaluation'/f'step_{selected:04d}_predictions.json')
    grouped = {mode: {} for mode in sources}
    for mode, preds in sources.items():
        for p in preds:
            if p['score'] >= .25:
                grouped[mode].setdefault(p['image_id'], []).append(p)
    cases = []
    for row in rows:
        gt = np.asarray([cm.decode(e) for e in row['rles']], dtype=np.uint8)
        stats = {}
        for mode in sources:
            preds = grouped[mode].get(row['image_id'], [])
            masks = np.asarray([cm.decode(p['segmentation']) for p in preds], dtype=np.uint8).reshape(-1, 640, 640)
            overlaps = cm.iou([p['segmentation'] for p in preds], row['rles'], [0]*len(gt)) if len(preds) else np.zeros((0, len(gt)))
            stats[mode] = {'masks': masks, 'mean_best_gt_iou': float(overlaps.max(0).mean()) if len(preds) else 0.}
        cases.append({'row': row, 'gt': gt, 'stats': stats,
                      'delta': stats['spm']['mean_best_gt_iou']-stats['continued']['mean_best_gt_iou']})
    ordered = sorted(cases, key=lambda c: c['delta'], reverse=True)
    selected = ordered[:3]+list(reversed(ordered[-3:]))
    # This diagnostic does not replace official AP and can reward merged masks; label it explicitly.
    dump(out/'CASE_SELECTION.json', {'selection_metric': 'mean_GT_best_mask_IoU_at_display_score_0.25_not_AP',
                                    'all_cases': [{'source_id': c['row']['source_id'], 'delta': c['delta'],
                                                   **{m: c['stats'][m]['mean_best_gt_iou'] for m in sources}} for c in ordered]})
    fig, axes = plt.subplots(6, 4, figsize=(14, 18))
    zoomfig, zoomaxes = plt.subplots(6, 4, figsize=(12, 15))
    heatfig, heataxes = plt.subplots(6, 6, figsize=(18, 18))
    for i, case in enumerate(selected):
        row = case['row']; sample = store.get(row); rgb = sample['rgb'][:, :, ::-1]
        panels = [rgb, overlay(rgb, case['gt']), overlay(rgb, case['stats']['continued']['masks']), overlay(rgb, case['stats']['spm']['masks'])]
        fg, contact, boundary, valid = contact_target(sample)
        center_y, center_x = np.unravel_index(cv2.GaussianBlur(contact, (11, 11), 0).argmax(), contact.shape)
        if not contact.any():
            center_y, center_x = 40, 40
        x0, y0 = min(480, max(0, int(center_x*8)-80)), min(480, max(0, int(center_y*8)-80))
        for j, panel in enumerate(panels):
            axes[i, j].imshow(panel); axes[i, j].axis('off')
            zoomaxes[i, j].imshow(panel[y0:y0+160, x0:x0+160]); zoomaxes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(['RGB', 'Visible GT', 'B* continued (score >= .25)', 'F SPM (score >= .25)'][j], fontsize=10)
                zoomaxes[i, j].set_title(['RGB contact crop', 'Visible GT', 'B* continued', 'F SPM'][j], fontsize=10)
        axes[i, 0].text(0, -.03, f"{row['source_id']} | delta best-GT IoU {case['delta']:+.4f}", transform=axes[i, 0].transAxes, fontsize=8)
        zoomaxes[i, 0].text(0, -.03, row['source_id'], transform=zoomaxes[i, 0].transAxes, fontsize=8)
        model.spm.capture = True
        with torch.no_grad():
            predict(model, store.batch([row], 0, augment=False)['img'])
        maps = model.spm.maps; model.spm.capture = False
        np.savez_compressed(out/f"response_{row['image_id']}.npz", **{k: v.numpy() for k, v in maps.items() if k != 'query'})
        data = [rgb, maps['foreground_similarity'][0], maps['background_similarity'][0], maps['top1'][0],
                maps['gate'][0], maps['residual_rms'][0]/maps['feature_rms'][0].clamp_min(1e-8)]
        for j, value in enumerate(data):
            a = heataxes[i, j]
            if j == 0:
                a.imshow(value)
            else:
                opts = {'cmap': 'viridis', 'vmin': -1 if j in [1, 2] else 0, 'vmax': 1}
                if j == 3:
                    opts = {'cmap': 'nipy_spectral', 'vmin': 0, 'vmax': 31}
                elif j == 5:
                    opts = {'cmap': 'magma', 'vmin': 0, 'vmax': max(float(np.max(value.numpy())), .001)}
                im = a.imshow(value.numpy(), **opts); heatfig.colorbar(im, ax=a, fraction=.046, pad=.02)
            a.axis('off')
            if i == 0:
                a.set_title(['Input', 'FG max cosine', 'BG max cosine', 'Global top-1 ID', 'Actual gate', 'Effective delta / feature RMS'][j], fontsize=10)
    fig.suptitle('Top 3 improved and top 3 degraded diagnostic cases | all 54 cases retained', fontsize=13)
    fig.tight_layout(); fig.savefig(out/'07_improvement_and_damage.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    zoomfig.suptitle('Contact-centered crops | identical pixels and display threshold', fontsize=13)
    zoomfig.tight_layout(); zoomfig.savefig(out/'08_contact_zoom.png', dpi=160, bbox_inches='tight'); plt.close(zoomfig)
    heatfig.suptitle('Frozen prototype retrieval and learned gate | actual checkpoint responses', fontsize=13)
    heatfig.tight_layout(); heatfig.savefig(out/'09_retrieval_gate_residual.png', dpi=140, bbox_inches='tight'); plt.close(heatfig)


@torch.no_grad()
def latency(model, image):
    model.eval()
    for _ in range(5):
        model(image)
    torch.cuda.synchronize(); times = []
    for _ in range(20):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); model(image); b.record(); b.synchronize(); times.append(a.elapsed_time(b))
    return {'median_forward_ms': float(np.median(times)), 'p90_forward_ms': float(np.quantile(times, .9)),
            'batch': 1, 'precision': 'fp32', 'warmup': 5, 'repetitions': 20,
            'parameters': sum(p.numel() for p in model.parameters()), 'gpu': torch.cuda.get_device_name()}


def visualize(root, store):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'svg.fonttype': 'none'})
    with run_context(root, 'visualize') as (run, meta):
        out = run/'figures'; out.mkdir()
        meta['visualization_source_sha256'] = sha(Path(__file__))
        (run/'VISUALIZATION_SOURCE.py').write_bytes(Path(__file__).read_bytes())
        summaries = {mode: read(root/'runs'/read(root/(mode.upper()+'_POINTER.json'))['run_id']/'SUMMARY.json') for mode in LABELS}
        method_figure(out); overview(root, out, store, summaries); memory = memory_figures(root, store, out)
        timing = {}; image = store.batch([next(r for r in store.rows if r['domain'] == 'real')], 0, augment=False)['img']
        for mode in LABELS:
            model = load(stage_checkpoint(root, mode)); timing[mode] = latency(model, image)
            del model; torch.cuda.empty_cache()
        model = load(stage_checkpoint(root, 'spm')); model.eval()
        comparison_cases(root, store, out, model)
        interventions = {}
        original_alpha = model.spm.alpha.detach().clone()
        model.spm.alpha.data.zero_(); interventions['alpha_zero'] = evaluate(model, root, store, run/'interventions', 'alpha_zero')
        model.spm.alpha.data.copy_(original_alpha)
        original_values = model.spm.values.clone()
        model.spm.values.copy_(torch.roll(original_values, 1, 0))
        interventions['wrong_key_value'] = evaluate(model, root, store, run/'interventions', 'wrong_key_value')
        model.spm.values.copy_(original_values); model.spm.intervention = 'shuffle_spatial'
        interventions['shuffle_spatial'] = evaluate(model, root, store, run/'interventions', 'shuffle_spatial')
        model.spm.intervention = 'none'
        primary = summaries['spm']['selected']['mask_ap']-summaries['continued']['selected']['mask_ap']
        increment = summaries['spm']['selected']['mask_ap']-summaries['adapter']['selected']['mask_ap']
        rows = [{'arm': LABELS[m], **summaries[m]['selected'], **timing[m],
                 'delta_vs_continued_mask_ap': summaries[m]['selected']['mask_ap']-summaries['continued']['selected']['mask_ap']} for m in LABELS]
        dump(run/'RESULTS.json', {'arms': rows, 'spm_minus_continued_mask_ap_pp': primary, 'spm_minus_adapter_mask_ap_pp': increment,
                                  'memory_interventions': interventions, 'selection': 'development_selection_not_test',
                                  'actual_spm_alpha': float(original_alpha)})
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].bar([LABELS[m] for m in LABELS], [timing[m]['median_forward_ms'] for m in LABELS], color=[COLORS[m] for m in LABELS])
        ax[0].set(ylabel='FP32 forward latency (ms), batch 1', title='Same GPU, median of 20 repetitions'); ax[0].tick_params(axis='x', rotation=20)
        labels = ['F intact']+list(interventions)
        vals = [summaries['spm']['selected']['mask_ap']]+[x['mask_ap'] for x in interventions.values()]
        bars = ax[1].bar(labels, vals, color=['#009E73', '#777777', '#CC79A7', '#E69F00']); ax[1].bar_label(bars, fmt='%.2f')
        ax[1].set(ylabel='Development Mask AP (%)', title='Same F checkpoint, zero-training interventions'); ax[1].tick_params(axis='x', rotation=20)
        ax[1].set_ylim(0, max(vals)*1.18+1); fig.tight_layout(); savefig(fig, out, '10_cost_and_interventions')
        best = max(LABELS, key=lambda m: summaries[m]['selected']['mask_ap'])
        decision = {'status': 'development_positive' if primary > 0 else 'no_spm_development_gain',
                    'primary_delta_pp': primary, 'increment_over_adapter_pp': increment, 'best_measured_configuration': best,
                    'scientific_scope': 'single_seed_small_development_set_no_mechanism_or_final_test_claim',
                    'recommended_direction': 'prioritize_the_best_measured_configuration_and_verify_its_real_held_out_utility',
                    'next_action': 'preserve_and_deliver_all_weights_metrics_and_technical_visualizations', 'further_training_started': False}
        dump(run/'SUMMARY.json', {'results': rows, 'decision': decision, 'figures': len(list(out.glob('*.png')))})
        dump(run/'DECISION.json', decision)
        lines = ['# SPM-YOLO V4 technical visualization gallery', '',
                 f'Single seed; 54 real development images; 640 canvas. F - continued: {primary:+.4f} pp; F - G: {increment:+.4f} pp.', '',
                 'Checkpoint selection uses development AP. These are not independent test results.', '']
        for path in sorted(out.glob('*.png')):
            lines += ['## '+path.stem, '', f'![{path.stem}](figures/{path.name})', '']
        (run/'VISUAL_GUIDE.md').write_text('\n'.join(lines), encoding='utf-8')
        print('VISUALIZATION_COMPLETE', decision, flush=True)
        del model; torch.cuda.empty_cache()
