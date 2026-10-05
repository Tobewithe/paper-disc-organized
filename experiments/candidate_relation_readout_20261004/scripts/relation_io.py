"""Join frozen prediction-derived relation maps; GT stays outside model inputs."""
from pathlib import Path
import torch
from runtime_utils import load
from train_evidence import read_training_image as original_read, feed as original_feed

RELATION_KEYS = ('relation_self', 'relation_true', 'relation_wrong')

def read_training_image(cfg, iid):
    x = original_read(cfg, iid)
    r = load(Path(cfg['relation_cache'])/'images'/f'{iid:012d}.pt')
    assert r['image_id'] == int(iid) and torch.equal(r['raw_ids'], x['raw_ids'])
    assert r['source_fingerprint'] == x['fingerprint']
    for key in RELATION_KEYS:
        assert r[key].shape == (len(x['raw_ids']),3,16,16)
        assert r[key].dtype == torch.float32 and torch.isfinite(r[key]).all()
    x['_relation'] = r
    return x

def feed(images):
    fs, selected = original_feed(images)
    for x, row in zip(images, selected):
        row.update({key:x['_relation'][key].cuda().detach() for key in RELATION_KEYS})
    return fs, selected
