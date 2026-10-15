"""Behavioral duplicate-ID, crowd, ties, maxDets and empty-image tests."""
import common
import numpy as np
from pycocotools import mask as masks
from coco_bootstrap import evaluate, compact, verify_replication, resampled_evaluator, ap, clone_item


def main():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    rle = masks.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode()
    gt = dict(images=[dict(id=i, height=12, width=12) for i in (1, 2, 3)],
              categories=[dict(id=i, name=str(i)) for i in (1, 2)], info={},
              annotations=[dict(id=11, image_id=1, category_id=1, bbox=[2, 2, 6, 6],
                                segmentation=rle, area=36, iscrowd=0),
                           dict(id=900100374545, image_id=2, category_id=2, bbox=[2, 2, 6, 6],
                                segmentation=rle, area=36, iscrowd=1)])
    pred = [dict(image_id=1, category_id=1, score=.5, bbox=[2, 2, 6, 6], segmentation=rle) for _ in range(105)]
    pred += [dict(image_id=2, category_id=2, score=.5, bbox=[2, 2, 6, 6], segmentation=rle),
             dict(image_id=3, category_id=1, score=.5, bbox=[2, 2, 6, 6], segmentation=rle)]
    ev = evaluate(gt, pred, [1, 2, 3], ap_only=True, quiet=True)
    cache = compact(ev)
    assert ap(resampled_evaluator(cache, [0, 1, 2])) == ap(ev)
    for draws in ([0, 0, 2, 1, 0], [2, 0, 0, 1, 2], [1, 2, 0]):
        verify_replication(cache, gt, pred, draws)
    item = next(r for category in cache['entries'] for r in category if r is not None and 900100374545 in r['gtIds'])
    last = clone_item(item, 4999)
    assert max(last['gtIds']) < 2**53
    try:
        clone_item(item, 9007)
    except ValueError as exc:
        assert 'exact float64' in str(exc)
    else:
        raise AssertionError('large-copy-ID overflow not blocked')
    print('BOOTSTRAP_BEHAVIOR_TESTS_PASSED', flush=True)


if __name__ == '__main__':
    main()
