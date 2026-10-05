"""Memory-bounded evaluation of completed corrected-suite checkpoints."""
from corrected_suite import *


def load_corrected(keys):
    models = {}
    for key in keys:
        mode, seed = key.removeprefix('new_').rsplit('_s', 1)
        model = CorrectedHead(mode).to(DEVICE)
        checkpoint = OUT / 'corrected_training' / f'{mode}_s{seed}' / 'epoch015.pt'
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=False)['model'])
        models[key] = model.eval()
    return models


def arm_group(mode):
    keys = [f'new_{mode}_s{s}' for s in range(3)]
    models = load_corrected(keys)
    arms = {key: (key, mode in ('full', 'full_noorth'), False) for key in keys}
    if mode in ('bsr', 'ada', 'scalar', 'orth'):
        arms.update({f'new_{mode}_center_s{s}': (f'new_{mode}_s{s}', True, False) for s in range(3)})
    if mode in ('full', 'full_noorth'):
        arms.update({f'new_{mode}_nocenter_s{s}': (f'new_{mode}_s{s}', False, False) for s in range(3)})
    return models, arms


def main():
    sel = json.loads((CACHE / 'selection.json').read_text())
    normalizer = torch.load(LABELS / 'normalizer.pt', map_location=DEVICE, weights_only=False)
    groups = ['bsr', 'ada', 'scalar', 'contrast', 'adapter', 'orth', 'full_noorth', 'full']
    parts = []
    for mode in groups:
        dest = OUT / 'corrected_grouped' / mode
        models, arms = arm_group(mode)
        if not (dest / 'COMPLETE.json').exists():
            reevaluate(models, arms, dest, sel['transfer'], normalizer)
        parts.append(dest)
        del models
        torch.cuda.empty_cache()

    # Baselines are copied from the already completed aligned-decoder evaluation.
    existing = OUT / 'existing_aligned'
    summary = [pd.read_csv(p / 'summary.csv') for p in parts]
    gtrows = [pd.read_csv(p / 'gt_recovery.csv') for p in parts]
    pairrows = [pd.read_csv(p / 'pair_recovery.csv') for p in parts]
    old_summary = pd.read_csv(existing / 'summary.csv')
    old_gt = pd.read_csv(existing / 'gt_recovery.csv')
    old_pair = pd.read_csv(existing / 'pair_recovery.csv')
    keep = old_summary.arm.str.match(r'^(original|s032_s[0-2])$')
    keep_arms = set(old_summary.loc[keep, 'arm'])
    final = OUT / 'corrected_aligned'
    final.mkdir(exist_ok=True)
    pd.concat([old_summary.loc[keep], *summary], ignore_index=True).to_csv(final / 'summary.csv', index=False)
    pd.concat([old_gt[old_gt.arm.isin(keep_arms)], *gtrows], ignore_index=True).to_csv(final / 'gt_recovery.csv', index=False)
    pd.concat([old_pair[old_pair.arm.isin(keep_arms)], *pairrows], ignore_index=True).to_csv(final / 'pair_recovery.csv', index=False)
    write(final / 'COMPLETE.json', dict(images=len(sel['transfer']), arms=sum(len(x) for x in summary) + int(keep.sum()),
          execution='memory-bounded method groups; identical decoder and evaluator', parts=[str(p) for p in parts]))
    state('corrected_grouped_complete', arms=sum(len(x) for x in summary) + int(keep.sum()))


if __name__ == '__main__':
    main()
