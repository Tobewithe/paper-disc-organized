"""Run the S080 audit from completed corrected-suite checkpoints."""
from corrected_suite import *


def main():
    normalizer = torch.load(LABELS / 'normalizer.pt', map_location=DEVICE, weights_only=False)
    models = {'original': None}
    old = load_old()
    for seed in range(3):
        models[f's032_s{seed}'] = old[f's032_s{seed}']
    del old
    for seed in range(3):
        for mode in ['bsr', 'ada', 'scalar', 'contrast', 'adapter', 'orth', 'full_noorth', 'full']:
            model = CorrectedHead(mode).to(DEVICE)
            checkpoint = OUT / 'corrected_training' / f'{mode}_s{seed}' / 'epoch015.pt'
            model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=False)['model'])
            models[f'new_{mode}_s{seed}'] = model.eval()
    s080(models, normalizer)
    state('s080_corrected_complete', arms=len(models))


if __name__ == '__main__':
    main()
