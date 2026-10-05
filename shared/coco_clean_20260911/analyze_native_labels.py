"""S020 paired label-pipeline contrasts, with image-cluster uncertainty."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

ARMS = ['COCO_CV_LINEAR160', 'NATIVE_INDEPENDENT160', 'NATIVE_OVERLAP160',
        'NATIVE_OVERLAP640', 'NATIVE_OVERLAP160_NATIVEBOX', 'NATIVE_OVERLAP640_NATIVEBOX']
CONTRASTS = {
    'downsample_linear_minus_nearest': ('COCO_CV_LINEAR160', 'S019_A160'),
    'native_polygon_minus_coco_linear': ('NATIVE_INDEPENDENT160', 'COCO_CV_LINEAR160'),
    'overlap_minus_independent': ('NATIVE_OVERLAP160', 'NATIVE_INDEPENDENT160'),
    'native160_minus_old160': ('NATIVE_OVERLAP160', 'S019_A160'),
    'native640_minus_old640': ('NATIVE_OVERLAP640', 'S019_C640'),
    'grid_native_rawbox': ('NATIVE_OVERLAP640', 'NATIVE_OVERLAP160'),
    'grid_native_nativebox': ('NATIVE_OVERLAP640_NATIVEBOX', 'NATIVE_OVERLAP160_NATIVEBOX'),
    'native_box_and_area160': ('NATIVE_OVERLAP160_NATIVEBOX', 'NATIVE_OVERLAP160'),
    'native_box_and_area640': ('NATIVE_OVERLAP640_NATIVEBOX', 'NATIVE_OVERLAP640'),
    'native160_nativebox_minus_old160': ('NATIVE_OVERLAP160_NATIVEBOX', 'S019_A160')}

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def read(p):
    with open(p, newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); out = a.out
    receipt = json.loads((out/'COMPLETE.json').read_text())
    assert receipt['status'] == 'COMPLETE'
    for name, value in receipt['hashes'].items(): assert sha(out/name) == value, name
    rows = read(out/'metrics.csv'); labels = read(out/'labels.csv'); witness = read(out/'witness.csv')
    protocol = json.loads((out/'protocol.json').read_text())
    aids = protocol['target_ids']; lookup = {(int(r['annotation_id']), r['arm']): r for r in rows}
    assert len(rows) == len(lookup) == len(aids)*len(ARMS)
    lab = {int(r['annotation_id']): r for r in labels}
    assert len(lab) == len(labels) == len(aids)
    paired = []
    for aid in aids:
        rr = {arm: lookup[(aid, arm)] for arm in ARMS}; ref = rr[ARMS[0]]
        assert len({r['image_id'] for r in rr.values()}) == 1
        vals = {arm: float(r['coco_iou']) for arm, r in rr.items()}
        vals.update(S019_A160=float(ref['s019_A_iou']), S019_C640=float(ref['s019_C_iou']), original=float(ref['original_iou']))
        for r in rr.values():
            assert float(r['s019_A_iou']) == vals['S019_A160'] and float(r['s019_C_iou']) == vals['S019_C640']
        paired.append(dict(annotation_id=aid, image_id=int(ref['image_id']), density=ref['density'],
            area=float(ref['area']), parts=int(ref['parts']), **vals,
            **{name: vals[plus]-vals[minus] for name, (plus, minus) in CONTRASTS.items()}))
    with open(out/'PAIRED_TARGETS.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    ids = sorted({r['image_id'] for r in paired}); index = {v: i for i, v in enumerate(ids)}
    boot = np.random.default_rng(20260912).multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=2000)
    def estimate(rr, field):
        num = np.zeros(len(ids)); den = np.zeros(len(ids))
        for r in rr: num[index[r['image_id']]] += r[field]; den[index[r['image_id']]] += 1
        bd = boot@den; bn = boot@num; ok = bd > 0
        return dict(n=len(rr), mean_pp=100*float(num.sum()/den.sum()) if rr else None,
            ci95_pp=(100*np.quantile(bn[ok]/bd[ok], [.025, .975])).tolist() if ok.any() else None)
    contrasts = []; subgroups = []; summary = []; label_summary = []
    for density in ['high', 'other']:
        selected = [r for r in paired if r['density'] == density]
        for name in CONTRASTS: contrasts.append(dict(density=density, contrast=name, **estimate(selected, name)))
        for arm in ['original', 'S019_A160', 'S019_C640'] + ARMS:
            summary.append(dict(density=density, arm=arm, n=len(selected), mean_iou=100*float(np.mean([r[arm] for r in selected])),
                recovered75=sum(r[arm]>=.75 for r in selected)))
        for dimension, threshold in [('area', 1024), ('parts', 2)]:
            for above in [False, True]:
                sub = [r for r in selected if (r[dimension] >= threshold) == above]
                subgroups.append(dict(density=density, dimension=dimension, threshold=threshold, above_or_equal=above,
                    effects={name: estimate(sub, name) for name in ['native160_minus_old160', 'grid_native_nativebox']}))
        ll = [r for r in labels if r['density'] == density]
        entry = dict(density=density, n=len(ll), multipart=sum(int(r['parts'])>1 for r in ll),
            targets_changed_by_overlap=sum(int(r['native_overlap_loss_pixels'])>0 for r in ll),
            bbox_difference_over_one_input_pixel=sum(float(r['nativebox_max_abs_input_px'])>1 for r in ll))
        for field in ['old_nearest_pixels', 'coco_linear_pixels', 'native_independent_pixels', 'native_overlap_pixels', 'actual_loader_pixels']:
            entry[field+'_zero_count'] = sum(int(r[field]) == 0 for r in ll)
        for field in ['old_vs_linear_iou', 'old_vs_native_independent_iou', 'old_vs_native_overlap_iou',
                      'native_independent_vs_overlap_iou', 'nativebox_max_abs_input_px', 'native_overlap_loss_pixels']:
            values = np.array([float(r[field]) for r in ll])
            entry[field] = dict(mean=float(values.mean()), median=float(np.median(values)), max=float(values.max()))
        label_summary.append(entry)
    doc = dict(summary=summary, contrasts=contrasts, subgroups=subgroups, labels=label_summary,
        actual_loader=dict(images=len(witness), instance_rows=sum(int(r['native_instances']) for r in witness),
            annotation_id_sets_all_match=all(r['all_annotation_ids_retained']=='True' for r in witness),
            dataset_outputs_all_replayed=all(r['actual_ds_replayed']=='True' for r in witness),
            image_pixels_differ=sum(int(r['geometry_pixel_channels_different'])>0 for r in witness)),
        optimizer=dict(total=len(rows), iteration_limit=sum(int(r['iterations'])>=protocol['solver']['max_iter'] for r in rows),
            empty_support=sum(r['status']=='empty_support_unchanged' for r in rows),
            loss_increases=sum(float(r['loss_after'])>float(r['loss_before'])+1e-6 for r in rows)),
        bootstrap='2,000 shared image-cluster resamples; target-weighted, pointwise percentile intervals, exploratory fixed failure cohort.',
        scope='Native Format controlled substitutions on common cached geometry; actual val loader audited separately. No historical pretraining or augmented-train replay, no reassignment, no network training, no method AP.',
        receipt_files_verified=len(receipt['hashes']), script_sha256=sha(__file__), metrics_sha256=sha(out/'metrics.csv'))
    (out/'LABEL_ANALYSIS.json').write_text(json.dumps(doc, indent=2), encoding='utf-8')
    print(json.dumps(doc, indent=2))

if __name__ == '__main__': main()
