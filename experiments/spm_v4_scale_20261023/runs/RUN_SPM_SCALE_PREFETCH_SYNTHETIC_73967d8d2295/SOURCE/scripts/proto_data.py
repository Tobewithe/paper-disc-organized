from collections import Counter

import cv2
import numpy as np
import torch

from common import dump, sha
from task_model import query_features


def contact_target(sample):
    ids = sample['ids']; fg = ids > 0; contact = np.zeros_like(fg); kernel = np.ones((9, 9), np.uint8)
    for identity in np.unique(ids):
        if identity:
            own = ids == identity; contact |= own & (cv2.dilate((fg & ~own).astype(np.uint8), kernel) > 0)
    boundary = fg & (cv2.dilate((~fg).astype(np.uint8), kernel) > 0)
    valid = np.zeros_like(fg); left, top, w, h = sample['valid_rect']; valid[top:top+h, left:left+w] = True
    valid &= cv2.dilate(sample['overlap'], np.ones((3, 3), np.uint8)) == 0
    resize = lambda value: cv2.resize(value.astype(np.float32), (160, 160), interpolation=cv2.INTER_AREA)
    return resize(fg), resize(contact), resize(boundary), resize(valid) > .999


def sample_episode(sample, rng):
    fg, contact, boundary, valid = contact_target(sample)
    ids = sample['ids']; ids80 = ids[2::4, 2::4]
    candidates = []
    for identity in np.unique(ids):
        if not identity:
            continue
        purity = cv2.resize((ids == identity).astype(np.float32), (160, 160), interpolation=cv2.INTER_AREA)
        allowed = valid & (ids80 == identity) & (purity >= .5)
        if allowed.sum() >= 4:
            candidates.append((int(identity), allowed, float(contact[allowed].sum()), purity))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[2])
    chosen = candidates[:4]
    rest = candidates[4:]
    if rest:
        chosen += [rest[i] for i in rng.permutation(len(rest))[:4]]
    indices, labels, strata, purities = [], [], [], []
    background = np.flatnonzero((valid & (fg < .01) & (ids80 == 0)).reshape(-1))
    bg = rng.choice(background, min(32, len(background)), replace=False)
    indices.extend(bg.tolist()); labels.extend([0]*len(bg)); strata.extend(['BG']*len(bg)); purities.extend([1.]*len(bg))
    mapping = {}
    for local, (identity, allowed, _, purity) in enumerate(chosen, 1):
        mapping[identity] = local
        contact_pool = np.flatnonzero((allowed & (contact > .02)).reshape(-1))
        chosen_contact = rng.choice(contact_pool, min(6, len(contact_pool)), replace=False).tolist()
        available = np.setdiff1d(np.flatnonzero(allowed.reshape(-1)), chosen_contact)
        other = rng.choice(available, min(12-len(chosen_contact), len(available)), replace=False).tolist()
        picks = chosen_contact+other
        indices.extend(picks); labels.extend([local]*len(picks))
        strata.extend('C' if contact.reshape(-1)[j] > .02 else 'B' if boundary.reshape(-1)[j] > .05 else 'I' for j in picks)
        purities.extend(purity.reshape(-1)[picks].tolist())
    assert 0 < len(indices) <= 128
    pairs = set()
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)]:
        y0, y1 = max(0, -dy), min(160, 160-dy); x0, x1 = max(0, -dx), min(160, 160-dx)
        a, b = ids80[y0:y1, x0:x1], ids80[y0+dy:y1+dy, x0+dx:x1+dx]
        for code in np.unique(a.astype(np.uint32)*65536+b.astype(np.uint32)):
            left, right = int(code)//65536, int(code) % 65536
            if left != right and int(left) in mapping and int(right) in mapping:
                pairs.add(tuple(sorted([mapping[int(left)], mapping[int(right)]])))
    idx = np.asarray(indices)
    return {'indices': indices, 'labels': labels, 'strata': strata, 'purity': purities,
            'targets': np.stack([fg.reshape(-1)[idx], contact.reshape(-1)[idx]], axis=1),
            'contact_pairs': sorted(pairs), 'instance_ids': [c[0] for c in chosen]}


@torch.no_grad()
def extract(root, store, encoder, run, domains):
    train = [r for r in store.rows if r['split'] == 'train' and r['domain'] in domains]
    paired = {r['group']: r for r in train if r['domain'] == 'synthetic' and r['appearance_id'] == 1}
    source = [r for r in train if r['domain'] != 'synthetic' or r['appearance_id'] == 0]
    rng = np.random.default_rng(23102321); aa, bb, labels, targets, episodes, skipped = [], [], [], [], [], []
    offset = 0
    for count, row in enumerate(source):
        episode = sample_episode(store.get(row), rng)
        if episode is None:
            skipped.append(row['source_id']); continue
        first = store.batch([row], 0, augment=False)['img']
        if row['domain'] == 'synthetic':
            second = store.batch([paired[row['group']]], 0, augment=False)['img']
        else:
            second = ((first-.5)*float(rng.uniform(.8, 1.2))+.5+float(rng.uniform(-.06, .06))).clamp(0, 1)
        feats = query_features(encoder, torch.cat([first, second])).permute(0, 2, 3, 1).reshape(2, 25600, 512).cpu()
        idx = episode['indices']; aa.append(feats[0, idx].half()); bb.append(feats[1, idx].half())
        labels.extend(episode['labels']); targets.extend(episode['targets'].tolist())
        episodes.append({'source_id': row['source_id'], 'ordinal': row['ordinal'], 'group': row['group'], 'domain': row['domain'],
                         'start': offset, 'stop': offset+len(idx), 'indices': idx, 'strata': episode['strata'],
                         'contact_pairs': episode['contact_pairs'], 'instance_ids': episode['instance_ids']})
        offset += len(idx)
        if count % 100 == 0:
            print('TASK_FEATURES', count+1, len(source), 'tokens', offset, flush=True)
    payload = {'a': torch.cat(aa), 'b': torch.cat(bb), 'labels': torch.tensor(labels, dtype=torch.long),
               'targets': torch.tensor(targets, dtype=torch.float32), 'episodes': episodes, 'skipped_proto_only': skipped,
               'cache_precision': 'fp16_storage_from_fp32_P2_P3', 'extractor_parameters': 'fixed_FP32_Bstar',
               'source_domains': list(domains)}
    torch.save(payload, run/'features.pt')
    dump(run/'FEATURE_RECEIPT.json', {'source_images': len(source), 'eligible_episodes': len(episodes), 'tokens': offset,
                                     'domains': dict(Counter(e['domain'] for e in episodes)), 'skipped_proto_only': skipped,
                                     'feature_cache_sha256': sha(run/'features.pt')})
    return payload


def merge_features(first, second):
    offset = len(first['a'])
    episodes = first['episodes']+[{**e, 'start': e['start']+offset, 'stop': e['stop']+offset} for e in second['episodes']]
    return {k: torch.cat([first[k], second[k]]) for k in ['a', 'b', 'labels', 'targets']} | {'episodes': episodes}
