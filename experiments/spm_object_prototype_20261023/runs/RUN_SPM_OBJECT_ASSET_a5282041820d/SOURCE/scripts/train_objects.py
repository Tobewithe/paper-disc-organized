import argparse
import copy
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Objects
from object_model import ObjectModel, objective
from runtime import context, dump, read, sha


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def train(root, device):
    with context(root, 'train', authorization=True) as (run, meta, cfg):
        location = read(root/'DATA_LOCATION.json')
        folder = Path(location['directory'])
        assert sha(folder/'MANIFEST.json') == location['manifest_sha256']
        dataset = Objects(folder, 'train')
        template = {k: torch.from_numpy(v).to(device) for k, v in dataset.template.items()}
        all_counts = {}
        initial = None
        for arm in cfg['arms']:
            seed(cfg['seed'])
            model = ObjectModel(dataset.template, cfg, structured=arm == 'structured').to(device)
            if initial is None:
                initial = copy.deepcopy(model.state_dict())
            else:
                model.load_state_dict(initial)
            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_counts[arm] = parameters
            assert len(set(all_counts.values())) == 1, all_counts
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
            generator = torch.Generator().manual_seed(cfg['seed'])
            loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                                num_workers=2, generator=generator, persistent_workers=True)
            iterator = iter(loader)
            losses_log = []
            exposures = np.zeros(len(dataset), np.int64)
            model.train()
            for step in range(1, cfg['training_steps_per_arm']+1):
                try: batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader); batch = next(iterator)
                for index in batch['index'].tolist(): exposures[index] += 1
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                result = model(batch['rgb'], batch['camera'])
                paired = model(batch['pair_rgb'], batch['camera'])
                loss, components = objective(result, paired, batch, template, cfg)
                assert torch.isfinite(loss), (arm, step)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                meta['optimizer_steps'] += 1
                if step == 1 or step % 100 == 0:
                    row = dict(step=step, loss=float(loss.detach()), **{k: float(v.detach()) for k,v in components.items()})
                    losses_log.append(row)
                    dump(run/(arm+'_PROGRESS.json'), row)
                    dump(run/'run.json', meta)
                    print(arm, step, row['loss'], flush=True)
            path = run/(arm+'_last.pt')
            torch.save(dict(model=model.state_dict(), arm=arm, optimizer_steps=cfg['training_steps_per_arm'],
                            config=cfg, config_sha256=sha(root/'CONFIG_V0.json'), template_sha256=sha(folder/'template.npz'),
                            source_data_manifest_sha256=location['manifest_sha256'], trainable_parameters=parameters), path)
            dump(run/(arm+'_TRAINING.json'), dict(optimizer_steps=cfg['training_steps_per_arm'], parameters=parameters,
                                               losses=losses_log, checkpoint_sha256=sha(path),
                                               images_in_sampler=len(dataset), min_exposure=int(exposures.min()), max_exposure=int(exposures.max()),
                                               exposed_samples=int((exposures>0).sum())))
            del optimizer, model
        dump(run/'SUMMARY.json', dict(optimizer_steps=meta['optimizer_steps'], per_arm_steps=cfg['training_steps_per_arm'],
                                     trainable_parameters=all_counts, checkpoint_selection='fixed_last_step; no development selection',
                                     source_data_manifest_sha256=location['manifest_sha256'], real_data_used=False,
                                     environment={'torch':torch.__version__, 'numpy':np.__version__, 'cuda':torch.version.cuda}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--device', default='cuda')
    a=p.parse_args(); torch.set_num_threads(4); train(a.root.resolve(), a.device)
