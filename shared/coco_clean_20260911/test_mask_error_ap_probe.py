"""Synthetic ownership and prediction-only competition witnesses for S056."""
import numpy as np
from pycocotools import mask as mu
from mask_error_ap_probe import compete, encode, error_partition


def run():
    own = np.array([[1, 1, 0, 0, 0, 0]], bool)
    same = np.array([[0, 1, 1, 1, 0, 0]], bool)
    different = np.array([[0, 0, 0, 1, 1, 0]], bool)
    crowd = np.zeros_like(own)
    part, counts = error_partition(own, own, same, different, crowd)
    assert sum(counts.values()) == 0, "Own/neighbor GT overlap is not target leakage"
    pred = np.ones_like(own)
    part, counts = error_partition(pred, own, same, different, crowd)
    assert counts == dict(target_fn=0, same_only=1, other_only=1, ambiguous_neighbor=1, background=1)
    crowd[0, 5] = True
    assert error_partition(pred, own, same, different, crowd)[1]["background"] == 0

    masks = [np.array([[1, 1, 1, 0]], bool), np.array([[0, 1, 1, 1]], bool)]
    predictions = [dict(category_id=1, score=.9, segmentation=encode(masks[0])),
                   dict(category_id=1, score=.8, segmentation=encode(masks[1]))]
    boxes = [dict(bbox=[0, 0, 2, 1]), dict(bbox=[2, 0, 2, 1])]
    scored = compete(predictions, boxes, {}, .1, "score")
    assert "0" not in scored
    assert np.array_equal(mu.decode(scored["1"]), [[0, 0, 0, 1]])
    centered = compete(predictions, boxes, {}, .1, "center")
    outputs = [mu.decode(centered.get(str(i), p["segmentation"])) for i, p in enumerate(predictions)]
    assert np.all(sum(outputs) == 1), "Same-class positive support is assigned exactly once"
    assert np.array_equal(np.logical_or.reduce(outputs), np.logical_or.reduce(masks))
    predictions[1]["category_id"] = 2
    assert not compete(predictions, boxes, {}, .1, "score"), "Different classes do not compete"
    assert all(np.array_equal(mu.decode(p["segmentation"]), m) for p, m in zip(predictions, masks))
    print("PASS: exclusive error partition, crowd exclusion, ownership conservation, category gate, immutable input")


if __name__ == "__main__":
    run()
