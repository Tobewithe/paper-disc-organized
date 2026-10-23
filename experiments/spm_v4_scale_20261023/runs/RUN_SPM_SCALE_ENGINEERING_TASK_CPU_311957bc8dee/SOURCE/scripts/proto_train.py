from collections import Counter
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import dump, read, run_context, sha, state_hash
from data import Store
from proto_data import extract, merge_features
from task_model import Projector, load
from task_context import task_context


def checkpoint(root, stage):
    return root/'runs'/read(root/(stage.upper()+'_POINTER.json'))['run_id']/'best.pt'


def token_batch(features, domains, step):
    pools = {domain: [i for i, e in enumerate(features['episodes']) if e['domain'] == domain] for domain in set(domains)}
    rng = np.random.default_rng(231023000+step); idx, episode_ids, pairs = [], [], []
    for local, domain in enumerate(domains):
        ep = features['episodes'][int(rng.choice(pools[domain]))]
        idx.extend(range(ep['start'], ep['stop'])); episode_ids.extend([local]*(ep['stop']-ep['start']))
        pairs.extend((local*9+a, local*9+b) for a, b in ep['contact_pairs'])
    return (torch.tensor(idx, device='cuda'), torch.tensor(episode_ids, device='cuda'),
            torch.tensor(pairs, dtype=torch.long, device='cuda').reshape(-1, 2))


def task_loss(projector, fg_head, contact_head, a, b, labels, target, episode_ids, pairs, cfg, pos_weight):
    raw_a, raw_b = projector(a), projector(b)
    za, zb = F.normalize(raw_a, dim=1), F.normalize(raw_b, dim=1)
    flat_id = episode_ids*9+labels; group_count = 16*9
    counts = torch.bincount(flat_id, minlength=group_count).float()
    sums_a = torch.zeros((group_count, 128), device=a.device).index_add_(0, flat_id, za)
    sums_b = torch.zeros((group_count, 128), device=a.device).index_add_(0, flat_id, zb)
    means_a, means_b = F.normalize(sums_a/counts.clamp_min(1).unsqueeze(1), dim=1), F.normalize(sums_b/counts.clamp_min(1).unsqueeze(1), dim=1)
    logits_a = torch.einsum('nd,nkd->nk', za, means_b.view(16, 9, 128)[episode_ids])/.1
    logits_b = torch.einsum('nd,nkd->nk', zb, means_a.view(16, 9, 128)[episode_ids])/.1
    # Leave the queried pixel out of its positive support prototype.
    loo_b = F.normalize((sums_b[flat_id]-zb)/(counts[flat_id]-1).clamp_min(1).unsqueeze(1), dim=1)
    loo_a = F.normalize((sums_a[flat_id]-za)/(counts[flat_id]-1).clamp_min(1).unsqueeze(1), dim=1)
    own_a, own_b = (za*loo_b).sum(1)/.1, (zb*loo_a).sum(1)/.1
    logits_a = logits_a.scatter(1, labels[:, None], own_a[:, None]); logits_b = logits_b.scatter(1, labels[:, None], own_b[:, None])
    present = counts.view(16, 9)[episode_ids] > 0
    logits_a = logits_a.masked_fill(~present, -10000); logits_b = logits_b.masked_fill(~present, -10000)
    class_count = (counts.view(16, 9) > 0).sum(1).clamp_min(1)
    weight = 1/counts[flat_id].clamp_min(1)/class_count[episode_ids]/16
    ce = ((F.cross_entropy(logits_a, labels, reduction='none')+F.cross_entropy(logits_b, labels, reduction='none'))*.5*weight).sum()
    fg = ((F.binary_cross_entropy_with_logits(fg_head(raw_a)[:, 0], target[:, 0], reduction='none')+
           F.binary_cross_entropy_with_logits(fg_head(raw_b)[:, 0], target[:, 0], reduction='none'))*.5*weight).sum()
    contact = ((F.binary_cross_entropy_with_logits(contact_head(raw_a)[:, 0], target[:, 1], pos_weight=pos_weight, reduction='none')+
                F.binary_cross_entropy_with_logits(contact_head(raw_b)[:, 0], target[:, 1], pos_weight=pos_weight, reduction='none'))*.5*weight).sum()
    cons = ((1-(za*zb).sum(1))*weight).sum()
    pull = (((1-(za*loo_b).sum(1))+(1-(zb*loo_a).sum(1)))*.5*weight).sum()
    push = a.new_zeros(())
    if len(pairs):
        push = .5*(F.relu((means_a[pairs[:, 0]]*means_a[pairs[:, 1]]).sum(1)-.3).square().mean()+
                   F.relu((means_b[pairs[:, 0]]*means_b[pairs[:, 1]]).sum(1)-.3).square().mean())
    variance = .5*(F.relu(1-raw_a.std(0)).mean()+F.relu(1-raw_b.std(0)).mean())
    centered = raw_a-raw_a.mean(0); cov = centered.T@centered/max(1, len(a)-1)
    covariance = (cov.square().sum()-cov.diag().square().sum())/128
    pieces = {'instance_mask_alignment': ce, 'foreground': fg, 'contact': contact, 'correspondence': cons,
              'instance_pull': pull, 'contact_push': push, 'variance': variance, 'covariance': covariance}
    total = sum(cfg['loss_weights'][key]*value for key, value in pieces.items())
    return total, pieces


def optimize(root, run, meta, features, cfg, start, stop, state=None):
    a, b = features['a'].cuda().float(), features['b'].cuda().float()
    labels, targets = features['labels'].cuda(), features['targets'].cuda()
    if state is None:
        real_ids = [i for ep in features['episodes'] if ep['domain'] == 'real' for i in range(ep['start'], ep['stop'])]
        real = torch.cat([a[real_ids], b[real_ids]])
        mean, std = real.mean(0), real.std(0).clamp_min(.05); del real
    else:
        mean, std = state['mean'].cuda(), state['std'].cuda()
    a, b = (a-mean)/std, (b-mean)/std
    projector, fg_head, contact_head = Projector().cuda(), nn.Linear(128, 1).cuda(), nn.Linear(128, 1).cuda()
    params = list(projector.parameters())+list(fg_head.parameters())+list(contact_head.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg['prototype_lr'])
    if state is not None:
        projector.load_state_dict(state['projector']); fg_head.load_state_dict(state['fg_head']); contact_head.load_state_dict(state['contact_head'])
        optimizer.load_state_dict(state['optimizer'])
    pos_weight = ((1-targets[:, 1].mean())/targets[:, 1].mean().clamp_min(.001)).clamp(1, 20)
    log = []; begin = time.monotonic()
    for step in range(start+1, stop+1):
        if step <= cfg['prototype_warm_steps']:
            domains = ['real']*8+['cutout']*8
        elif step <= cfg['prototype_warm_steps']+cfg['prototype_mixed_steps']:
            domains = ['real']*8+['synthetic']*4+['cutout']*4
        else:
            domains = ['real']*16
        indices, episode_ids, pairs = token_batch(features, domains, step)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = task_loss(projector, fg_head, contact_head, a[indices], b[indices], labels[indices], targets[indices], episode_ids, pairs, cfg, pos_weight)
        assert torch.isfinite(loss), step
        loss.backward(); gradient = float(torch.nn.utils.clip_grad_norm_(params, 10.)); assert np.isfinite(gradient)
        optimizer.step()
        log.append({'step': step, 'loss': float(loss.detach()), 'gradient_norm': gradient, 'tokens': len(indices),
                    **{key: float(value.detach()) for key, value in parts.items()}, 'elapsed_seconds': time.monotonic()-begin})
        if step == start+1 or step % 100 == 0:
            print('TASK_PROTO', step, stop, round(log[-1]['loss'], 4), 'instance_ce', round(log[-1]['instance_mask_alignment'], 4), flush=True)
            meta.update(optimizer_steps=step-start, global_prototype_step=step)
            dump(run/'run.json', meta); dump(run/'TRAINING.json', log)
    state = {'projector': {k: v.detach().cpu() for k, v in projector.state_dict().items()},
             'fg_head': {k: v.detach().cpu() for k, v in fg_head.state_dict().items()},
             'contact_head': {k: v.detach().cpu() for k, v in contact_head.state_dict().items()},
             'mean': mean.cpu(), 'std': std.cpu(), 'optimizer': optimizer.state_dict(), 'global_step': stop}
    torch.save(state, run/'PROJECTOR_STATE.pt')
    dump(run/'OPTIMIZATION_SUMMARY.json', {'new_optimizer_steps': stop-start, 'global_step': stop, 'first_loss': log[0]['loss'],
                                          'last_loss': log[-1]['loss'], 'tokens_in_cache': len(a), 'elapsed_seconds': time.monotonic()-begin})
    return state, projector, a, b


def centers(z, count, seed):
    generator = torch.Generator(device=z.device).manual_seed(seed)
    chosen = [int(torch.randint(len(z), (), device=z.device, generator=generator))]
    distance = torch.full((len(z),), 2., device=z.device)
    for _ in range(1, count):
        distance = torch.minimum(distance, (1-z@z[chosen[-1]]).clamp_min(0))
        chosen.append(int(torch.multinomial(distance+1e-8, 1, generator=generator)))
    result = z[chosen].clone()
    for _ in range(20):
        assignment = (z@result.T).argmax(1)
        for k in range(count):
            if (assignment == k).any():
                result[k] = F.normalize(z[assignment == k].mean(0), dim=0)
    return result


def bank(root, run, features, state, projector, a, b, upstream):
    with torch.no_grad():
        z = torch.cat([F.normalize((projector(a[i:i+8192])+projector(b[i:i+8192]))*.5, dim=1) for i in range(0, len(a), 8192)])
        domains = np.empty(len(z), dtype=np.int8); strata = np.empty(len(z), dtype=np.int8)
        domain_codes = {'real': 0, 'synthetic': 1, 'cutout': 2}; stratum_codes = {'C': 0, 'B': 1, 'I': 2, 'BG': 3}
        for ep in features['episodes']:
            domains[ep['start']:ep['stop']] = domain_codes[ep['domain']]
            strata[ep['start']:ep['stop']] = [stratum_codes[s] for s in ep['strata']]
        rng = np.random.default_rng(23102361); reservoirs = []
        for foreground in [True, False]:
            choices = []
            for domain, budget in [(0, 100000), (1, 50000), (2, 50000)]:
                valid = (features['targets'][:, 0].numpy() >= .5) if foreground else (features['targets'][:, 0].numpy() < .01)
                for stratum in ([0, 1, 2] if foreground else [3]):
                    eligible = np.flatnonzero(valid & (domains == domain) & (strata == stratum))
                    count = min(len(eligible), budget//(3 if foreground else 1))
                    choices.extend(rng.choice(eligible, count, replace=False).tolist())
            reservoirs.append(torch.tensor(choices, device='cuda'))
        key = torch.cat([centers(z[reservoirs[0]], 16, 13), centers(z[reservoirs[1]], 16, 17)])
        similarity = z@key.T
        foreground = features['targets'][:, 0].cuda() >= .5
        assignment = torch.where(foreground, similarity[:, :16].argmax(1), similarity[:, 16:].argmax(1)+16)
        ends = np.asarray([e['stop'] for e in features['episodes']]); clusters = []
        for k in range(32):
            members = torch.nonzero(assignment == k)[:, 0]; selected = members[similarity[members, k].argsort(descending=True)[:8]].cpu().tolist()
            exemplars = []
            for index in selected:
                ep = features['episodes'][int(np.searchsorted(ends, index, side='right'))]; local = index-ep['start']; position = ep['indices'][local]
                exemplars.append({'source_id': ep['source_id'], 'ordinal': ep['ordinal'], 'group': ep['group'], 'domain': ep['domain'],
                                  'stratum': ep['strata'][local], 'x': position % 160, 'y': position//160,
                                  'token_index': index, 'cosine': float(similarity[index, k])})
            m = members.cpu().numpy()
            clusters.append({'prototype': k, 'support_tokens': len(m), 'exemplars': exemplars,
                             'domain_counts': {d: int((domains[m] == code).sum()) for d, code in domain_codes.items()},
                             'stratum_counts': {s: int((strata[m] == code).sum()) for s, code in stratum_codes.items()}})
        memory = {k: state[k] for k in ['projector', 'mean', 'std', 'fg_head', 'contact_head']}
        memory.update(centers=key.cpu(), clusters=clusters, K=32, dimension=128, feature_dimension=512, grid_size=160,
                      method='instance_task_mask_prototype_P2_P3',
                      upstream_checkpoint_sha256=sha(upstream), dataset_sha256=sha(root/'ALL_DATASET.json'),
                      prototype_total_steps=20000, train_only=True)
        torch.save(memory, run/'memory.pt')
        torch.save({'embedding': z.cpu(), 'assignment': assignment.cpu(), 'episodes': features['episodes']}, run/'embedding.pt')
        dump(run/'BANK_SUMMARY.json', {'K': 32, 'tokens': len(z), 'clusters': clusters, 'reservoir_counts': [len(r) for r in reservoirs],
                                      'memory_sha256': sha(run/'memory.pt'), 'all_sources_train_only': True})


def train_prototypes(root, stage):
    cfg = read(root/'TASK_CONFIG_V3.json'); torch.manual_seed(0)
    with task_context(root, 'prototype_'+stage) as (run, meta):
        upstream = checkpoint(root, 'baseline'); encoder = load(upstream).eval().requires_grad_(False)
        before = state_hash(encoder)
        if stage == 'warm':
            store = Store(read(root/'WARM_DATASET.json')['rows']); features = extract(root, store, encoder, run, ['real', 'cutout'])
            state, projector, a, b = optimize(root, run, meta, features, cfg, 0, 5000)
            complete_steps = 5000
        else:
            warmrun = root/'runs'/read(root/'PROTOTYPE_WARM_POINTER.json')['run_id']
            warm_features = torch.load(warmrun/'features.pt', map_location='cpu', weights_only=False)
            warm_state = torch.load(warmrun/'PROJECTOR_STATE.pt', map_location='cpu', weights_only=False)
            store = Store(read(root/'ALL_DATASET.json')['rows']); synthetic = extract(root, store, encoder, run, ['synthetic'])
            features = merge_features(warm_features, synthetic)
            state, projector, a, b = optimize(root, run, meta, features, cfg, 5000, 20000, warm_state)
            bank(root, run, features, state, projector, a, b, upstream); complete_steps = 15000
            meta['parent_warm_run_id'] = warmrun.name
        assert state_hash(encoder) == before
        meta.update(optimizer_steps=complete_steps, global_prototype_step=state['global_step'], upstream_checkpoint_sha256=sha(upstream))
        dump(run/'SUMMARY.json', {'new_optimizer_steps': complete_steps, 'global_step': state['global_step'], 'encoder_unchanged': True,
                                  'upstream_checkpoint_sha256': sha(upstream), 'tokens': len(features['a']),
                                  'source_episodes': len(features['episodes'])})
        dump(run/'DECISION.json', {'status': 'warmup_completed' if stage == 'warm' else 'task_memory_trained_and_frozen',
                                   'utility_target': 'not_measured_by_auxiliary_loss',
                                   'next_action': 'mixed_then_real_adaptation' if stage == 'warm' else 'matched_8000_step_four_arm_training'})
