"""Meaningful guards for stream identity, actual geometry and full-GT scoring."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from pycocotools import mask as masks

from s1_geometry import iter_prediction_array, project_gt, predicted_roi, full_rle_iou, geometry


class S1GeometryTests(unittest.TestCase):
    def test_stream_preserves_objects_at_every_boundary(self):
        values = [{"image_id": i, "text": "\\\"} 中文", "bbox": [1.5, 2, 3, 4]} for i in range(9)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf8")
            for size in (1, 2, 7, 63, 1024):
                self.assertEqual(list(iter_prediction_array(path, size)), values)

    def test_stream_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            for bad in ('[{},]', '[{} {}]', '[{}', '[{}] {}', '{"x":1}', '[1]'):
                path.write_text(bad, encoding="utf8")
                with self.assertRaises(ValueError):
                    list(iter_prediction_array(path, 2))
            path.write_text(' [ ] ', encoding="utf8")
            self.assertEqual(list(iter_prediction_array(path, 1)), [])

    def test_recorded_anisotropic_resize_and_asymmetric_padding(self):
        cache = {"original_hw": [3, 5], "input_hw": [8, 12],
                 "ratio_pad": [[5 / 3, 2], [1, 1]], "proto": torch.zeros(2, 8, 12)}
        mask = np.zeros((3, 5), np.uint8)
        mask[1, 2] = 1
        got = project_gt(mask, cache)
        expected = torch.zeros(8, 12)
        expected[3:5, 5:7] = 1
        self.assertTrue(torch.equal(got, expected))
        self.assertEqual(geometry(cache)["ltrb_padding"], [1, 1, 1, 2])
        self.assertEqual(geometry(cache)["native_unpadded_hw"], [6, 10])

    def test_odd_padding_against_pinned_letterbox(self):
        from ultralytics.data.augment import LetterBox
        for h, w, ih, iw in ((427, 640, 448, 672), (426, 640, 448, 672), (640, 427, 672, 448)):
            params = LetterBox(new_shape=(ih, iw), scaleup=False).get_params({"img": np.zeros((h, w, 3), np.uint8)})
            cache = {"original_hw": [h, w], "input_hw": [ih, iw],
                     "ratio_pad": [[1., 1.], [float(params["left"]), float(params["top"])]]}
            expected = [params[k] for k in ("left", "top", "right", "bottom")]
            self.assertEqual(geometry(cache)["ltrb_padding"], expected)

    def test_roi_uses_half_open_native_pixel_support(self):
        cache = {"proto": torch.zeros(2, 4, 6), "input_hw": [8, 12],
                 "boxes_input": torch.tensor([[2.5, 2., 8., 6.]])}
        expected = torch.zeros(4, 6, dtype=torch.bool)
        expected[1:3, 2:4] = True
        self.assertTrue(torch.equal(predicted_roi(cache, 0), expected))

    def test_iou_keeps_gt_outside_prediction(self):
        gt = np.ones((4, 4), dtype=np.uint8)
        pred = gt.copy()
        pred[:, 2:] = 0
        self.assertEqual(full_rle_iou(masks.encode(np.asfortranarray(pred)),
                                    masks.encode(np.asfortranarray(gt))), .5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
