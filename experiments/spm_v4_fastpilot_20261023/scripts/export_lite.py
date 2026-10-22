"""Export the winning training weights without an inference-time memory branch."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch
from ultralytics.nn.tasks import SegmentationModel

from common import dump, read, run_context, sha
from data import Store
from evaluate import evaluate
from model import configure, load, save
from train import stage_checkpoint
from visualize import latency


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); args = p.parse_args(); root = Path(args.root)
    torch.set_num_threads(4)
    with run_context(root, 'export_lite') as (run, meta):
        source = stage_checkpoint(root, 'spm'); source_data = torch.load(source, map_location='cpu', weights_only=False)
        model = SegmentationModel(deepcopy(source_data['yaml']), nc=1, verbose=False).to('cuda').float()
        model.names = {0: 'pig'}
        model.load_state_dict({k: v for k, v in source_data['state_dict'].items() if not k.startswith('spm.')}, strict=True)
        configure(model, 400); model.eval()
        save(model, run/'best.pt', 'spm_training_only', source_data['step'])
        reloaded = load(run/'best.pt').eval()
        store = Store(read(root/'DATASET.json')['rows'])
        reference = load(source).eval(); reference.spm.alpha.data.zero_()
        errors = []
        for row in [r for r in store.rows if r['domain'] == 'real' and r['split'] == 'dev'][:5]:
            image = store.batch([row], 0, augment=False)['img']
            with torch.no_grad():
                actual, expected = reloaded(image)[0], reference(image)[0]
                err = max(float((a-b).abs().max()) for a, b in zip(actual, expected))
            errors.append({'source_id': row['source_id'], 'max_abs_difference': err})
        assert max(e['max_abs_difference'] for e in errors) == 0.
        metric = evaluate(reloaded, root, store, run/'evaluation', 'native_head_export')
        timings = latency(reloaded, image)
        meta.update(source_checkpoint_sha256=sha(source), export_source_sha256=sha(Path(__file__)))
        dump(run/'SUMMARY.json', {'metrics': metric, 'timing': timings, 'source_run_id': source.parent.name,
                                  'source_checkpoint_sha256': sha(source), 'export_checkpoint_sha256': sha(run/'best.pt'),
                                  'reload_equivalence_to_alpha_zero': errors, 'additional_training_steps': 0,
                                  'interpretation': 'SPM-training checkpoint with native-only deployment; not evidence of active retrieval utility'})
        dump(run/'DECISION.json', {'status': 'exported_and_evaluated', 'recommended_deployment': 'native_head_without_retrieval',
                                   'reason': 'intact_vs_alpha_zero_development_AP_difference_below_0.002_pp',
                                   'next_action': 'deliver_winning_weights_and_original_spm_research_checkpoint'})
        (run/'EXPORT_SOURCE.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
