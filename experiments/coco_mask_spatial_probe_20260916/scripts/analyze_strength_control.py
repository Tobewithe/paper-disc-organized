"""Fixed-strength controls, with all endpoints recomputed on the same device.

Image bootstrap is for fixed Mask75 matches, not for COCO AP. These comparisons
were specified after the previous validation results and remain exploratory.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    for key in ('source', 'previous', 'annotations', 'raw', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--coordinate-control', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    result = json.loads((a.source/'RESULTS.json').read_text())
    matched = {k: set(v) for k, v in json.loads((a.source/'MATCHED_GT75.json').read_text()).items()}
    old = json.loads((a.previous/'RESULTS.json').read_text())
    old_matched = {k: set(v) for k, v in json.loads((a.previous/'MATCHED_GT75.json').read_text()).items()}
    ids = json.loads((a.source/'SPLIT.json').read_text())['val_ids']
    ix = {v: i for i, v in enumerate(ids)}
    anns = {v['id']: v for v in json.loads(a.annotations.read_text())['annotations']
            if v['image_id'] in ix and not v.get('iscrowd', 0) and not v.get('ignore', 0)}
    raw = {v['annotation_id']: v for line in a.raw.read_text().splitlines() if (v := json.loads(line))}
    assert len(anns) == 36335 and set(raw) == set(anns)
    assert len(ids) == 5000 and set(result) == set(matched)
    base = matched['baseline']
    for name, v in result.items():
        assert v['repaired75'] == len(matched[name]-base)
        assert v['damaged75'] == len(base-matched[name])
    groups = {'all': set(anns), 'raw_box75_mask_unavailable': {
        k for k, v in raw.items() if v['geometry_state'] == 'box_good_mask_unavailable'}}
    for name, lo, hi in [('small', 0, 1024), ('medium', 1024, 9216), ('large', 9216, float('inf'))]:
        groups[name] = {k for k, v in anns.items() if lo <= v['area'] < hi}
    rng = np.random.default_rng(20260916)
    draws = rng.multinomial(len(ids), [1/len(ids)]*len(ids), size=2000).astype(np.float32)
    pairs = [('local4_s0', 'baseline')]
    if a.coordinate_control:
        pairs += [('local4_center_s0', 'local4_s0'), ('local4_center_s0', 'baseline')]
    else:
        for obj in ('guard', 'safe'):
            half, full = f'local4_{obj}_half_s0', f'local4_{obj}_s0'
            pairs += [(half, 'local4_s0'), (full, 'local4_s0'), (half, full)]
    contrasts = {}
    for left, right in pairs:
        entry = {}
        for name, group in groups.items():
            counts = np.zeros(len(ids), np.float32)
            changes = np.zeros(len(ids), np.float32)
            for k in group:
                j = ix[anns[k]['image_id']]
                counts[j] += 1
                changes[j] += int(k in matched[left])-int(k in matched[right])
            samples = 100*(draws@changes)/(draws@counts)
            entry[name] = dict(gt=len(group), net=int(changes.sum()),
                recall_delta_pp=float(100*changes.sum()/len(group)), ci95=np.quantile(samples, [.025, .975]).tolist())
        contrasts[left+'__minus__'+right] = entry
    reference_repairs = matched['local4_s0']-base
    reference_damage = base-matched['local4_s0']
    transitions = {}
    for name, good in matched.items():
        if name in ('baseline', 'local4_s0'):
            continue
        rep, dam = good-base, base-good
        transitions[name] = dict(retained_repairs=len(rep&reference_repairs),
            lost_repairs=len(reference_repairs-rep), added_repairs=len(rep-reference_repairs),
            rescued_damage=len(reference_damage-dam), added_damage=len(dam-reference_damage),
            original_repairs=len(reference_repairs), original_damage=len(reference_damage))
    environment = {}
    for name in set(result)&set(old):
        environment[name] = dict(AP_delta_pp=100*(result[name]['metrics']['AP']-old[name]['metrics']['AP']),
            matched_gained=len(matched[name]-old_matched[name]), matched_lost=len(old_matched[name]-matched[name]))
    analysis = dict(contrasts=contrasts, transitions=transitions, cross_environment_sensitivity=environment,
        note='Fresh same-device controls. Post-hoc strength diagnostic, not validation-based deployment selection. CIs only for fixed Mask75 recall, not AP or seed variation.')
    (a.out/'ANALYSIS.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    lines = ['# '+('局部修正坐标对照' if a.coordinate_control else '同幅度局部修正对照'), '', '| 模型 | alpha | AP | ΔAP | AP75 | APS | APM | APL | 修复 | 误伤 | 净增 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, v in result.items():
        m = v['metrics']
        lines.append(f"| {name} | {v['alpha']} | {100*m['AP']:.4f} | {100*(m['AP']-result['baseline']['metrics']['AP']):+.4f} | {100*m['AP75']:.4f} | {100*m['APS']:.4f} | {100*m['APM']:.4f} | {100*m['APL']:.4f} | {v['repaired75']} | {v['damaged75']} | {v['repaired75']-v['damaged75']} |")
    lines += ['', '固定原局部修正的修复/误伤GT集合，身份变化与分层配对区间见 ANALYSIS.json。',
              '所有模型在同一进程重新推理与评分；之前的结果只用于复现敏感性回查。后验诊断不用于重新挑选部署强度。']
    (a.out/'TABLE.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (a.out/'COMPLETE.json').write_text(json.dumps(dict(images=len(ids), gt=len(anns), variants=list(result))), encoding='utf-8')
    print(json.dumps(dict(all_contrasts={k:v['all'] for k,v in contrasts.items()}, transitions=transitions,
                         cross_environment_sensitivity=environment)), flush=True)


if __name__ == '__main__':
    main()
