from collections import Counter, defaultdict
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import dump, read, run_context, sha, state_hash
from data import contact_target
from model import load, p3, set_seed
from train import stage_checkpoint


@torch.no_grad()
def extract(root, store, encoder, run):
    rows = [r for r in store.rows if r['split'] == 'train']
    paired = {r['group']: r for r in rows if r['domain'] == 'synthetic' and r['appearance_id'] == 1}
    source = [r for r in rows if r['domain'] != 'synthetic' or r['appearance_id'] == 0]
    rng = np.random.default_rng(5000); aa, bb, targets, records = [], [], [], []
    for count, row in enumerate(source):
        first = store.batch([row], seed=0, augment=False)['img']
        if row['domain'] == 'synthetic':
            second = store.batch([paired[row['group']]], seed=0, augment=False)['img']
        else:
            second = ((first-.5)*float(rng.uniform(.8, 1.2))+.5+float(rng.uniform(-.06, .06))).clamp(0, 1)
        feats = p3(encoder, torch.cat([first, second])).permute(0, 2, 3, 1).reshape(2, 6400, 256).cpu()
        fg, contact, boundary, valid = contact_target(store.get(row))
        pools = {'C': valid & (contact > .02), 'B': valid & (boundary > .05) & (contact <= .02),
                 'I': valid & (fg > .95) & (boundary < .02) & (contact < .01),
                 'BG': valid & (fg < .01)}
        for stratum, mask in pools.items():
            eligible = np.flatnonzero(mask.reshape(-1))
            chosen = rng.choice(eligible, min(24, len(eligible)), replace=False)
            if not len(chosen):
                continue
            aa.append(feats[0, chosen]); bb.append(feats[1, chosen])
            targets.extend(np.stack([fg.reshape(-1)[chosen], contact.reshape(-1)[chosen]], 1).tolist())
            records.extend({'source_id': row['source_id'], 'ordinal': row['ordinal'], 'group': row['group'],
                            'domain': row['domain'], 'stratum': stratum, 'x': int(i % 80), 'y': int(i//80)} for i in chosen)
        if count % 40 == 0:
            print('FEATURES', count+1, len(source), len(records), flush=True)
    features = {'a': torch.cat(aa), 'b': torch.cat(bb), 'targets': torch.tensor(targets, dtype=torch.float32),
                'records': records, 'dataset_sha256': sha(root/'DATASET.json')}
    torch.save(features, run/'features.pt')
    return features


def fit_centers(z, n, seed):
    assert len(z) >= n
    gen = torch.Generator(device=z.device).manual_seed(seed)
    selected = [int(torch.randint(len(z), (), generator=gen, device=z.device))]
    distance = torch.full((len(z),), 2., device=z.device)
    for _ in range(1, n):
        distance = torch.minimum(distance, (1-z@z[selected[-1]]).clamp_min(0))
        selected.append(int(torch.multinomial(distance.add(1e-8), 1, generator=gen)))
    centers = z[selected].clone()
    for _ in range(20):
        assigned = (z@centers.T).argmax(1)
        for k in range(n):
            members = z[assigned == k]
            if len(members):
                centers[k] = F.normalize(members.mean(0), dim=0)
    return centers


def train_prototypes(root, store):
    cfg = read(root/'PILOT_CONFIG.json')
    with run_context(root, 'prototype') as (run, meta):
        set_seed(); checkpoint = stage_checkpoint(root, 'baseline')
        encoder = load(checkpoint).eval().requires_grad_(False); before = state_hash(encoder)
        features = extract(root, store, encoder, run)
        a, b, target = [features[k].to('cuda') for k in ['a', 'b', 'targets']]
        mean = torch.cat([a, b]).mean(0); std = torch.cat([a, b]).std(0).clamp_min(.05)
        a, b = (a-mean)/std, (b-mean)/std
        projector = nn.Linear(256, 128).cuda(); fg_head = nn.Linear(128, 1).cuda(); contact_head = nn.Linear(128, 1).cuda()
        params = list(projector.parameters())+list(fg_head.parameters())+list(contact_head.parameters())
        optimizer = torch.optim.Adam(params, lr=.001)
        pools = defaultdict(list)
        for i, r in enumerate(features['records']):
            pools[r['domain']+'/'+r['stratum']].append(i)
        keys = sorted(pools); rng = np.random.default_rng(6000)
        pos_weight = ((1-target[:, 1].mean())/target[:, 1].mean().clamp_min(.001)).clamp(1, 20)
        log = []; start = time.monotonic()
        for step in range(1, cfg['prototype_steps']+1):
            chosen_keys = rng.integers(len(keys), size=512)
            indices = [pools[keys[k]][int(rng.integers(len(pools[keys[k]])))] for k in chosen_keys]
            idx = torch.tensor(indices, device='cuda'); za, zb = projector(a[idx]), projector(b[idx]); y = target[idx]
            fg = .5*(F.binary_cross_entropy_with_logits(fg_head(za)[:, 0], y[:, 0])+
                      F.binary_cross_entropy_with_logits(fg_head(zb)[:, 0], y[:, 0]))
            contact = .5*(F.binary_cross_entropy_with_logits(contact_head(za)[:, 0], y[:, 1], pos_weight=pos_weight)+
                           F.binary_cross_entropy_with_logits(contact_head(zb)[:, 0], y[:, 1], pos_weight=pos_weight))
            cons = (1-F.cosine_similarity(za, zb)).mean()
            variance = .5*(F.relu(1-za.std(0)).mean()+F.relu(1-zb.std(0)).mean())
            centered = za-za.mean(0); cov = centered.T@centered/(len(za)-1)
            covariance = (cov.square().sum()-cov.diag().square().sum())/128
            total = fg+.5*contact+.2*cons+.1*variance+.01*covariance
            assert torch.isfinite(total)
            optimizer.zero_grad(); total.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(params, 10.)); assert np.isfinite(grad)
            optimizer.step()
            log.append({'step': step, 'loss': float(total.detach()), 'fg': float(fg.detach()),
                        'contact': float(contact.detach()), 'consistency': float(cons.detach()),
                        'variance': float(variance.detach()), 'covariance': float(covariance.detach()), 'gradient_norm': grad})
            if step == 1 or step % 100 == 0:
                print('PROTO', step, round(log[-1]['loss'], 4), flush=True)
                meta['optimizer_steps'] = step; dump(run/'run.json', meta); dump(run/'TRAINING.json', log)
        with torch.no_grad():
            z = F.normalize((projector(a)+projector(b))*.5, dim=1)
            foreground = target[:, 0] >= .5; background = target[:, 0] < .01
            fgcenters = fit_centers(z[foreground], 16, 1); bgcenters = fit_centers(z[background], 16, 2)
            centers = torch.cat([fgcenters, bgcenters])
            similarity = z@centers.T
            # Assign within the semantic partition; global cross-partition similarities are still visualized.
            assignment = torch.where(foreground, similarity[:, :16].argmax(1), similarity[:, 16:].argmax(1)+16)
            clusters = []
            for k in range(32):
                members = torch.nonzero(assignment == k)[:, 0]
                exemplar_ids = members[similarity[members, k].argsort(descending=True)[:8]].cpu().tolist()
                members_list = members.cpu().tolist()
                clusters.append({'prototype': k, 'partition': 'fg' if k < 16 else 'bg', 'support_tokens': len(members),
                                 'domain_counts': dict(Counter(features['records'][i]['domain'] for i in members_list)),
                                 'stratum_counts': dict(Counter(features['records'][i]['stratum'] for i in members_list)),
                                 'unique_groups': len({features['records'][i]['group'] for i in members_list}),
                                 'exemplars': [{**features['records'][i], 'cosine': float(similarity[i, k]), 'token_index': i}
                                               for i in exemplar_ids]})
            payload = {'mean': mean.cpu(), 'std': std.cpu(), 'projector_weight': projector.weight.detach().cpu(),
                       'projector_bias': projector.bias.detach().cpu(), 'centers': centers.cpu(),
                       'fg_head': {k: v.cpu() for k, v in fg_head.state_dict().items()},
                       'contact_head': {k: v.cpu() for k, v in contact_head.state_dict().items()},
                       'clusters': clusters, 'K': 32, 'topk': 4, 'temperature': .1,
                       'upstream_checkpoint_sha256': sha(checkpoint), 'upstream_state_sha256': before,
                       'dataset_sha256': sha(root/'DATASET.json'), 'prototype_run_id': run.name,
                       'projection_and_memory_frozen_for_fusion': True}
            torch.save(payload, run/'memory.pt')
            torch.save({'embedding': z.cpu(), 'assignment': assignment.cpu(), 'targets': target.cpu(),
                        'records': features['records']}, run/'embedding.pt')
        assert before == state_hash(encoder)
        summary = {'steps': cfg['prototype_steps'], 'tokens': len(z), 'K': 32, 'clusters': clusters,
                   'stratum_counts': dict(Counter(r['stratum'] for r in features['records'])),
                   'source_counts': dict(Counter(r['domain'] for r in features['records'])),
                   'memory_sha256': sha(run/'memory.pt'), 'first_loss': log[0]['loss'], 'last_loss': log[-1]['loss'],
                   'mean_cosine_correspondence': float(F.cosine_similarity(projector(a), projector(b)).mean()),
                   'encoder_unchanged': True, 'elapsed_training_seconds': time.monotonic()-start}
        meta['optimizer_steps'] = cfg['prototype_steps']; meta['initial_checkpoint_sha256'] = sha(checkpoint)
        dump(run/'SUMMARY.json', summary)
        dump(run/'DECISION.json', {'status': 'frozen_32_slot_memory_trained', 'AP_utility': 'not_evaluated_by_auxiliary_loss',
                                   'next_action': 'same_budget_continued_adapter_spm_training'})
        del encoder, optimizer, a, b, projector, z, similarity; torch.cuda.empty_cache()
    return summary
