"""Reuse the historical 10k image pool; download missing public COCO images only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from PIL import Image


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    old = json.loads(Path(cfg['historical_manifest']).read_text(encoding='utf-8-sig'))
    train = json.loads(Path(cfg['annotations_train']).read_text())
    val = json.loads(Path(cfg['annotations_val']).read_text())
    train_ids = {int(x['id']) for x in train['images']}
    val_ids = {int(x['id']) for x in val['images']}
    fit = list(map(int, old['train_images']))
    dev = list(map(int, old['old_development_images']))
    held = list(map(int, old['independent_test_images']))
    assert len(fit) == len(set(fit)) == 10000 and set(fit) <= train_ids
    assert not set(fit) & set(dev)
    pool = sorted(train_ids - set(fit) - set(dev), key=lambda v:
        hashlib.sha256(f'BGCR20261002:dev:{v}'.encode()).hexdigest())
    dev += pool[:1000-len(dev)]
    assert len(dev) == 1000 and len(held) == 2000 and set(held) <= val_ids
    split = dict(fit=fit, dev=dev, val=held)
    split_path = Path(cfg['split'])
    if split_path.exists():
        assert json.loads(split_path.read_text()) == split, 'Existing split changed'
    else:
        dump(split_path, split)
    todo = [('train2017', i) for i in fit + dev] + [('val2017', i) for i in held]
    start = time.monotonic()
    def ensure(item):
        domain, iid = item
        dest = Path(cfg['images']) / domain / f'{iid:012d}.jpg'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            with Image.open(dest) as im: im.verify()
            return dict(image_id=iid, domain=domain, reused=True, bytes=dest.stat().st_size)
        # Official COCO public image host uses HTTP; its HTTPS certificate
        # does not match images.cocodataset.org. Never disable TLS verification.
        url = f'http://images.cocodataset.org/{domain}/{iid:012d}.jpg'
        temp = dest.with_suffix('.download')
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={'User-Agent':'COCO-research-data-staging/1.0'})
                with urllib.request.urlopen(req, timeout=60) as src, temp.open('wb') as target:
                    while True:
                        block = src.read(1024*1024)
                        if not block: break
                        target.write(block)
                with Image.open(temp) as im: im.verify()
                temp.replace(dest)
                return dict(image_id=iid, domain=domain, reused=False, bytes=dest.stat().st_size, url=url)
            except Exception as exc:
                if attempt == 3: raise RuntimeError(f'{domain}/{iid}: {exc}') from exc
                time.sleep(1 + attempt)
    receipts = []; failed = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        jobs = {executor.submit(ensure, item):item for item in todo}
        for future in as_completed(jobs):
            try: receipts.append(future.result())
            except Exception as exc: failed.append(dict(item=jobs[future], error=str(exc)))
            n = len(receipts) + len(failed)
            if n % 100 == 0 or n == len(todo):
                progress = dict(stage='data', done=n, total=len(todo), failed=len(failed),
                    reused=sum(r['reused'] for r in receipts), elapsed_s=time.monotonic()-start)
                dump(out/'PROGRESS.json', progress); print(json.dumps(progress), flush=True)
                dump(out/'FAILED.json', failed)
    dump(out/'DATA_RECEIPTS.json', receipts); dump(out/'FAILED.json', failed)
    assert not failed, f'{len(failed)} files unavailable; do not shrink or replace the planned cohort'
    dump(out/'COMPLETE.json', dict(complete=True, images=len(receipts),
        planned={k:len(v) for k,v in split.items()}, old_pool_reused=True,
        evaluation_scope='held out from this training; historically examined, not a new blind test'))


if __name__ == '__main__': main()
