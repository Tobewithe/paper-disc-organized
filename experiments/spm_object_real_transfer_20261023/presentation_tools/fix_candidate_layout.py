"""Layout-only revision: separate the atlas heading from individual labels."""
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rt import dump, read, sha

p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
a = p.parse_args(); root = a.root.resolve(); torch.set_num_threads(2)
folder = root/'runs'/read(root/'PRESENTATION_POINTER.json')['run_id']
bank = torch.load(read(root/'BANK_POINTER.json')['file'], map_location='cpu', weights_only=False)
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
ids = np.linspace(0, len(bank['maps'])-1, 12, dtype=int)
for ax, k in zip(axes.flat, ids):
    value = bank['maps'][k].float().numpy()
    color = np.moveaxis((value[4:7]+1)/2, 0, -1)*value[0, :, :, None]
    ax.imshow(np.clip(color, 0, 1))
    ax.set_title(f"{bank['identities'][k]['sample_id']}\n{bank['identities'][k]['rotation_degrees']} deg", fontsize=8, pad=5)
    ax.axis('off')
fig.suptitle('Frozen learned-body candidates: feature colors are not semantic IDs', fontsize=12, y=.99)
fig.subplots_adjust(top=.83, bottom=.04, hspace=.45, wspace=.13)
target = root/'presentation_revision/04_learned_body_candidates_r1.png'
target.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(target, dpi=160, bbox_inches='tight', facecolor='white'); plt.close(fig)
dump(target.parent/'FIGURE_LAYOUT_R1.json', dict(source_sha256=sha(__file__), layout_only=True,
    original=folder.name+'/04_learned_body_candidates.png', original_sha256=sha(folder/'04_learned_body_candidates.png'),
    revised_sha256=sha(target), bank_sha256=read(root/'BANK_POINTER.json')['bank_sha256'],
    candidate_indices=ids.tolist(), scientific_values_changed=False))
print('LAYOUT_REVISED', sha(target), flush=True)
