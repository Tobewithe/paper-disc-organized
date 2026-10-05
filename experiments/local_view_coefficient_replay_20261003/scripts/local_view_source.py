"""GT-free, frozen local-image source for the local-view replay study.

The source selects only from the actual Segment26 one-to-one head top-300.
It never reads annotations, masks, assignment owners, or target classes.
Local masks are continuous full-prototype logits (NOT cropped by source boxes).
The original target box remains the evaluator's sole final mask support.

Pixel mapping uses actual integer resize dimensions, separate x/y gains,
and align_corners=False pixel centers. Unobserved pixels retain original
P*c0 logits, including original letterbox padding. There is no zero fill.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


ALLOWED_IMAGE_FIELDS = frozenset({
    "image_id", "source_split", "input_shape", "original_shape", "ratio_pad",
    "boxes", "proto", "c0", "input_uint8", "image_path", "raw_ids",
})
REQUIRED_IMAGE_FIELDS = frozenset({
    "image_id", "input_shape", "original_shape", "ratio_pad", "boxes", "proto",
    "c0", "input_uint8",
})


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def original_geometry(original_shape, input_shape, ratio_pad):
    """Describe the actual cached JPEG->input resize and integer padding.

    YOLODataset stores ((resized_h/original_h, resized_w/original_w),
    (integer_left, integer_top)). Those two ratios need not be identical.
    """
    oh, ow = map(int, original_shape)
    ih, iw = map(int, input_shape)
    ratios, pads = ratio_pad
    if isinstance(ratios, (list, tuple)):
        rh, rw = map(float, ratios)
    else:
        rh = rw = float(ratios)
    nh, nw = int(round(oh * rh)), int(round(ow * rw))
    left, top = (int(round(float(v))) for v in pads)
    if min(oh, ow, ih, iw, nh, nw) <= 0 or left < 0 or top < 0:
        raise ValueError("Invalid original input geometry")
    if left + nw > iw or top + nh > ih:
        raise ValueError("Cached resize dimensions exceed original input canvas")
    if any(abs(float(a) - b) > 1e-7 for a, b in zip(pads, (left, top))):
        raise ValueError("Expected original YOLODataset integer padding")
    return dict(original_shape=[oh, ow], input_shape=[ih, iw],
                resized_shape=[nh, nw], left=left, top=top,
                right=iw-left-nw, bottom=ih-top-nh,
                gain_x=nw / ow, gain_y=nh / oh)


def input_box_to_original(box, geometry):
    """Edge-coordinate box inverse, with original-image clipping."""
    b = np.asarray(box, dtype=np.float64).copy()
    if b.shape != (4,) or not np.isfinite(b).all():
        return None
    b[[0, 2]] = (b[[0, 2]] - geometry["left"]) / geometry["gain_x"]
    b[[1, 3]] = (b[[1, 3]] - geometry["top"]) / geometry["gain_y"]
    oh, ow = geometry["original_shape"]
    b[[0, 2]] = np.clip(b[[0, 2]], 0, ow)
    b[[1, 3]] = np.clip(b[[1, 3]], 0, oh)
    return b if b[2] > b[0] and b[3] > b[1] else None


def context_roi(box_original, original_shape, scale=1.5):
    """Predicted-box context ROI; integer half-open bounds, clipped, >=2 px."""
    if scale != 1.5:
        raise ValueError("Frozen local-view context scale is 1.5")
    oh, ow = map(int, original_shape)
    if min(oh, ow) < 2:
        raise ValueError("Original image must be at least two pixels on each axis")
    x1, y1, x2, y2 = map(float, box_original)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    width, height = max(2., scale*(x2-x1)), max(2., scale*(y2-y1))
    left, right = max(0, math.floor(cx-width/2)), min(ow, math.ceil(cx+width/2))
    top, bottom = max(0, math.floor(cy-height/2)), min(oh, math.ceil(cy+height/2))
    if right-left < 2:
        left = min(max(0, left), ow-2); right = left+2
    if bottom-top < 2:
        top = min(max(0, top), oh-2); bottom = top+2
    return [left, top, right, bottom]


def map_local_logits(local_logits, crop_xyxy, local_geometry, original,
                     baseline_logits):
    """Map one full local-input logit plane to the original input canvas.

    Remove local integer padding, then compose the crop and original resize
    transforms in one bilinear sample. The sampling domain is the unpadded
    local image, so padding cannot leak into observed pixels. Border sampling
    matches the edge replication convention of align_corners=False resize.
    Returns (mapped_with_baseline_fallback, observed_support).
    """
    ih, iw = original["input_shape"]
    if tuple(baseline_logits.shape) != (ih, iw) or local_logits.ndim != 2:
        raise ValueError("Expected two-dimensional logits on declared canvases")
    lx, ly, rx, by = map(int, crop_xyxy)
    nw, nh = map(int, local_geometry["new_unpad"])
    left, top = int(local_geometry["left"]), int(local_geometry["top"])
    if rx <= lx or by <= ly or min(nw, nh) < 1:
        raise ValueError("Empty local sampling domain")
    content = local_logits[top:top+nh, left:left+nw]
    if tuple(content.shape) != (nh, nw):
        raise ValueError("Local content rectangle exceeds local logit plane")
    dtype, device = local_logits.dtype, local_logits.device
    # Pixel centers expressed as JPEG edge coordinates.
    xx = (torch.arange(iw, device=device, dtype=dtype)+.5-original["left"]) / original["gain_x"]
    yy = (torch.arange(ih, device=device, dtype=dtype)+.5-original["top"]) / original["gain_y"]
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack((2*(gx-lx)/(rx-lx)-1, 2*(gy-ly)/(by-ly)-1), dim=-1)
    sampled = F.grid_sample(content[None, None], grid[None], mode="bilinear",
                            padding_mode="border", align_corners=False)[0, 0]
    support = (gx >= lx) & (gx < rx) & (gy >= ly) & (gy < by)
    oh, ow = original["original_shape"]
    support &= (gx >= 0) & (gx < ow) & (gy >= 0) & (gy < oh)
    return torch.where(support, sampled, baseline_logits), support


def _test_geometries():
    # Non-square original and non-square local crop with odd local padding.
    original = original_geometry((377, 613), (640, 640),
                                 ((394/377, 640/613), (0, 123)))
    crop = [37, 19, 512, 318]
    local = dict(new_unpad=[640, 403], left=0, right=0, top=118, bottom=119)
    return original, crop, local


def test_constant_geometry(device="cpu"):
    """No model: test constant preservation, non-square padding, and fallback."""
    original, crop, local = _test_geometries()
    z = torch.full((640, 640), -91., dtype=torch.float64, device=device)
    z[118:521] = 7.25
    baseline = torch.full_like(z, -3.125)
    mapped, support = map_local_logits(z, crop, local, original, baseline)
    err = float((mapped[support]-7.25).abs().max())
    unchanged = bool(torch.equal(mapped[~support], baseline[~support]))
    padding_clear = not bool(support[:123].any() or support[517:].any())
    if err > 1e-10 or not unchanged or not padding_clear or not bool(support.any()):
        raise AssertionError("Constant local-view geometry/fallback check failed")
    return dict(passed=True, maximum_absolute_error=err, fallback_exact=unchanged,
                original_padding_unobserved=padding_clear, observed_pixels=int(support.sum()))


def test_coordinate_ramp_geometry(device="cpu"):
    """No model: x/y coordinate ramps verify centers, axes, scale, and offset."""
    original, crop, local = _test_geometries()
    lx, ly, rx, by = crop; nw, nh = local["new_unpad"]
    xx = lx+(torch.arange(nw, device=device, dtype=torch.float64)+.5)*(rx-lx)/nw
    yy = ly+(torch.arange(nh, device=device, dtype=torch.float64)+.5)*(by-ly)/nh
    z = torch.full((640, 640), -91., dtype=torch.float64, device=device)
    z[118:521] = xx[None]/613 + .37*yy[:, None]/377
    baseline = torch.full_like(z, -3.125)
    mapped, support = map_local_logits(z, crop, local, original, baseline)
    ox = (torch.arange(640, device=device, dtype=z.dtype)+.5)/original["gain_x"]
    oy = (torch.arange(640, device=device, dtype=z.dtype)+.5-123)/original["gain_y"]
    # grid_sample(border) replicates the first/last content pixel centers.
    expected = ox.clamp(float(xx[0]), float(xx[-1]))[None]/613
    expected = expected + .37*oy.clamp(float(yy[0]), float(yy[-1]))[:, None]/377
    err = float((mapped[support]-expected[support]).abs().max())
    if err > 1e-10 or not torch.equal(mapped[~support], baseline[~support]):
        raise AssertionError("Coordinate ramp local-view mapping check failed")
    return dict(passed=True, maximum_absolute_error=err, align_corners=False,
                non_square_original=True, asymmetric_local_padding=[118, 119])


def geometry_test_cases(device="cpu"):
    return dict(passed=True, constant=test_constant_geometry(device),
                coordinate_ramp=test_coordinate_ramp_geometry(device))


class LocalViewSource:
    """Read only original JPEG, prediction fields, and frozen original model.

    infer(payload) yields one candidate at a time. predict(payload) is a
    convenience stack; infer is preferred for images with many candidates.
    Payload rejects every field outside ALLOWED_IMAGE_FIELDS, preventing
    accidental receipt of GT-bearing full assets.
    """

    def __init__(self, cfg, device="cuda", replay=None):
        self.cfg = dict(cfg)
        scripts = Path(self.cfg.get("source_script_dir",
            Path(self.cfg["original_screen_root"]) / "scripts"))
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from online_runtime import FrozenReplay
        self.replay = replay if replay is not None else FrozenReplay(self.cfg, device=device)
        self.device = self.replay.device
        self.source, self.head = self.replay.source, self.replay.head
        self.weights_sha256 = self.replay.weights_sha256
        if not self.head.end2end or self.head.export or self.head.max_det != 300:
            raise AssertionError("Require original eval one-to-one nonexport head with max_det=300")
        if (float(cfg.get("context_scale", 1.5)) != 1.5
                or int(cfg.get("imgsz", 640)) != 640
                or int(cfg.get("source_topk", 300)) != 300
                or cfg.get("source_conf_threshold") is not None):
            raise ValueError("This source only implements frozen 1.5-context/640/top300/no-threshold protocol")
        from ultralytics.data.augment import LetterBox
        self.letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False,
                                  scaleup=True, center=True, stride=32, padding_value=114)
        self.import_info = dict(self.replay.import_info, detector_head_forward=True,
            TAL_recomputed=False, source_pool="official Segment26 one2one top300",
            extra_confidence_filter=False, GT_used=False,
            mask_source_crop=False, source_mask_thresholded=False,
            mapping="integer-content inverse, composed pixel-center bilinear sampling",
            unseen_pixels="unchanged original full P*c0 logits")
        self._head_flags = (bool(self.head.end2end), bool(self.head.export),
                            int(self.head.max_det), bool(self.head.agnostic_nms))

    def assert_unchanged(self):
        self.replay.assert_unchanged()
        now = (bool(self.head.end2end), bool(self.head.export),
               int(self.head.max_det), bool(self.head.agnostic_nms))
        if now != self._head_flags:
            raise AssertionError("Frozen source head inference policy changed")

    def geometry_test_cases(self):
        return geometry_test_cases(self.device)

    def _read_jpeg(self, image, geometry):
        import cv2
        from ultralytics.utils.patches import imread
        iid = int(image["image_id"])
        if "image_path" in image:
            path = Path(image["image_path"])
        else:
            domain = image.get("source_split", "train")
            if domain not in ("train", "val"):
                raise ValueError("source_split must identify the original COCO train/val JPEG domain")
            path = Path(self.cfg["images"]) / f"{domain}2017" / f"{iid:012d}.jpg"
        jpeg = imread(str(path), flags=cv2.IMREAD_COLOR)
        if jpeg is None or list(jpeg.shape[:2]) != geometry["original_shape"]:
            raise AssertionError("Original JPEG missing or original dimensions changed")
        nh, nw = geometry["resized_shape"]
        resized = jpeg if tuple(jpeg.shape[:2]) == (nh, nw) else cv2.resize(
            jpeg, (nw, nh), interpolation=cv2.INTER_LINEAR)
        recovered = cv2.copyMakeBorder(resized, geometry["top"], geometry["bottom"],
            geometry["left"], geometry["right"], cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb_chw = np.ascontiguousarray(recovered[:, :, ::-1].transpose(2, 0, 1))
        cached = image["input_uint8"].detach().cpu()
        if cached.dtype != torch.uint8 or not torch.equal(torch.from_numpy(rgb_chw), cached):
            raise AssertionError("Original JPEG->input uint8 replay is not pixel-exact; stop, do not relax geometry")
        return jpeg, dict(path=str(path), sha256=_sha256(path), original_input_replay_exact=True)

    @torch.no_grad()
    def _forward_pool(self, local_bgr):
        params = self.letterbox.get_params({"img": local_bgr})
        padded = self.letterbox.apply_image({"img": local_bgr}, params)["img"]
        if tuple(padded.shape) != (640, 640, 3):
            raise AssertionError("Unexpected local LetterBox output")
        chw = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
        retained, raw_ids, proto = self._forward_input(torch.from_numpy(chw))
        return retained, raw_ids, proto, params

    @torch.no_grad()
    def _forward_input(self, input_uint8):
        """Original input or local input, with identical official pool tracking."""
        if input_uint8.dtype != torch.uint8 or tuple(input_uint8.shape) != (3, 640, 640):
            raise ValueError("Exact uint8 RGB 640 input required")
        output = self.source(input_uint8.to(self.device).float()[None]/255.)
        # Verified against vendored Segment26.forward, not a heuristic tuple search.
        if not (isinstance(output, tuple) and len(output) == 2
                and isinstance(output[0], tuple) and len(output[0]) == 2
                and isinstance(output[1], dict) and "one2one" in output[1]):
            raise AssertionError("Unexpected Ultralytics 8.4.100 Segment26 eval output")
        (retained, proto), raw = output
        one = raw["one2one"]
        if not torch.is_tensor(proto) or proto.ndim != 4 or proto.shape[:2] != (1, 32):
            raise AssertionError("Expected full eval Proto26 tensor [1,32,H,W]")
        if not torch.equal(proto, one["proto"]):
            raise AssertionError("Eval returned prototype differs from one2one raw prototype")
        score, cls, raw_ids = self.head.get_topk_index(one["scores"].sigmoid().permute(0, 2, 1), 300)
        boxes = self.head._get_decode_boxes(one).permute(0, 2, 1).gather(1, raw_ids.repeat(1, 1, 4))
        coefficients = one["mask_coefficient"].permute(0, 2, 1).gather(1, raw_ids.repeat(1, 1, 32))
        reconstructed = torch.cat((boxes, score, cls, coefficients), dim=-1)
        if retained.shape != reconstructed.shape or not torch.equal(retained, reconstructed):
            raise AssertionError("Tracked raw indices fail to reproduce actual official top300 output exactly")
        if len(retained[0]) != 300:
            raise AssertionError("Unexpected number of official retained source candidates")
        return retained[0], raw_ids[0, :, 0], proto[0]

    @staticmethod
    def _choose_source(source_boxes, target_original, retained, ids):
        """Same selection rule for original and local view; no class/GT input."""
        if not np.isfinite(source_boxes).all() or not torch.isfinite(retained).all():
            raise AssertionError("Nonfinite frozen source prediction")
        inter_xy = np.maximum(0, np.minimum(source_boxes[:, 2:], target_original[2:])-
                              np.maximum(source_boxes[:, :2], target_original[:2]))
        inter = inter_xy.prod(1)
        sa = np.maximum(0, source_boxes[:, 2:]-source_boxes[:, :2]).prod(1)
        ta = np.prod(target_original[2:]-target_original[:2])
        iou = inter / np.maximum(sa+ta-inter, np.finfo(np.float64).tiny)
        scores = retained[:, 4].detach().cpu().double().numpy()
        raw_ids = ids.detach().cpu().numpy()
        best = int(np.lexsort((np.arange(len(iou)), raw_ids, -scores, -iou))[0])
        return best, iou, scores, raw_ids

    @torch.no_grad()
    def infer(self, image, deadline_monotonic=None):
        unknown = set(image)-ALLOWED_IMAGE_FIELDS
        missing = REQUIRED_IMAGE_FIELDS-set(image)
        if unknown or missing:
            raise ValueError(f"Source payload must be GT-free allowlist; unknown={sorted(unknown)}, missing={sorted(missing)}")
        if tuple(image["input_shape"]) != (640, 640):
            raise ValueError("Original input must remain 640x640")
        boxes = image["boxes"].detach().cpu().double().numpy()
        c0, proto0 = image["c0"].to(self.device).float(), image["proto"].to(self.device).float()
        if c0.shape != (len(boxes), 32) or tuple(proto0.shape) != (32, 160, 160):
            raise ValueError("Frozen original candidate/prototype shape mismatch")
        geometry = original_geometry(image["original_shape"], image["input_shape"], image["ratio_pad"])
        jpeg, jpeg_meta = self._read_jpeg(image, geometry)
        self.assert_unchanged()
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("Frozen local-view replay resource deadline reached")
        # One additional ORIGINAL input pass per image: isolate re-selection
        # from changed visual input. Do not resize/re-letterbox these pixels.
        same_retained, same_ids, same_proto = self._forward_input(image["input_uint8"])
        torch.testing.assert_close(same_proto, proto0, atol=3e-5, rtol=3e-5)
        proto_audit = dict(passed=True, atol=3e-5, rtol=3e-5,
                           maximum_absolute_error=float((same_proto-proto0).abs().max()))
        same_boxes = same_retained[:, :4].detach().cpu().double().numpy().copy()
        oh, ow = geometry["original_shape"]
        same_boxes[:, [0, 2]] = np.clip((same_boxes[:, [0, 2]]-geometry["left"])/geometry["gain_x"], 0, ow)
        same_boxes[:, [1, 3]] = np.clip((same_boxes[:, [1, 3]]-geometry["top"])/geometry["gain_y"], 0, oh)
        self.assert_unchanged()
        for index, target_box in enumerate(boxes):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError("Frozen local-view replay resource deadline reached")
            baseline = F.interpolate((c0[index] @ proto0.flatten(1)).reshape(1, 1, 160, 160),
                                     (640, 640), mode="bilinear", align_corners=False)[0, 0]
            target_original = input_box_to_original(target_box, geometry)
            meta = dict(image_id=int(image["image_id"]), candidate_index=index,
                        jpeg=jpeg_meta, original_geometry=geometry,
                        source_branch="one2one", extra_confidence_filter=False, GT_used=False)
            if "raw_ids" in image:
                meta["target_raw_id"] = int(image["raw_ids"][index])
            if target_original is None:
                meta.update(status="invalid_original_predicted_box_baseline_fallback",
                            observed_pixels=0, fallback_pixels=640*640)
                yield dict(candidate_index=index, logit_input=baseline,
                           support_input=torch.zeros_like(baseline, dtype=torch.bool), metadata=meta,
                           same_view_coeff=c0[index], same_view_source_meta=dict(
                               status="invalid_original_predicted_box_baseline_fallback",
                               original_prototype_audit=proto_audit, GT_used=False))
                continue
            sbest, siou, sscores, sids = self._choose_source(
                same_boxes, target_original, same_retained, same_ids)
            same_view_coeff = same_retained[sbest, 6:]
            same_view_meta = dict(status="original_view_reselected", source_branch="one2one",
                original_prototype_audit=proto_audit, source_top300_reproduction_exact=True,
                source_pool_size=len(siou), source_pool_rank=sbest,
                source_raw_id=int(sids[sbest]), source_class=int(same_retained[sbest, 5]),
                source_confidence=float(sscores[sbest]), source_box_iou=float(siou[sbest]),
                source_box_original=same_boxes[sbest].tolist(),
                original_cached_input_used_exactly=True, extra_confidence_filter=False,
                source_box_crop_applied=False, GT_used=False)
            if "raw_ids" in image:
                target_id = int(image["raw_ids"][index])
                same_view_meta.update(target_raw_id=target_id,
                    target_in_source_pool=bool((same_ids == target_id).any()),
                    selected_original_target_raw=(int(sids[sbest]) == target_id))
            crop = context_roi(target_original, geometry["original_shape"])
            lx, ly, rx, by = crop
            retained, ids, local_proto, local_geometry = self._forward_pool(jpeg[ly:by, lx:rx])
            nw, nh = map(int, local_geometry["new_unpad"])
            source_boxes = retained[:, :4].detach().cpu().double().numpy().copy()
            source_boxes[:, [0, 2]] = (source_boxes[:, [0, 2]]-local_geometry["left"])*(rx-lx)/nw
            source_boxes[:, [1, 3]] = (source_boxes[:, [1, 3]]-local_geometry["top"])*(by-ly)/nh
            source_boxes[:, [0, 2]] = np.clip(source_boxes[:, [0, 2]], 0, rx-lx)+lx
            source_boxes[:, [1, 3]] = np.clip(source_boxes[:, [1, 3]], 0, by-ly)+ly
            best, iou, scores, raw_ids = self._choose_source(source_boxes, target_original, retained, ids)
            coeff = retained[best, 6:]
            local_logits = F.interpolate((coeff @ local_proto.flatten(1)).reshape(1, 1, *local_proto.shape[-2:]),
                                         (640, 640), mode="bilinear", align_corners=False)[0, 0]
            mapped, support = map_local_logits(local_logits, crop, local_geometry, geometry, baseline)
            # The final evaluator applies official crop_mask, not this bookkeeping mask.
            yy = torch.arange(640, device=self.device)[:, None]
            xx = torch.arange(640, device=self.device)[None]
            target_support = ((xx >= target_box[0]) & (xx < target_box[2]) &
                              (yy >= target_box[1]) & (yy < target_box[3]))
            serial_local = {k: [int(t) for t in v] if isinstance(v, (tuple, list)) else int(v)
                            for k, v in local_geometry.items() if k in
                            ("orig_shape", "new_shape", "new_unpad", "left", "right", "top", "bottom")}
            meta.update(status="local_view_selected", crop_xyxy=crop,
                target_box_original=target_original.tolist(), local_geometry=serial_local,
                source_pool_size=len(iou), source_pool_rank=best,
                source_raw_id=int(raw_ids[best]), source_class=int(retained[best, 5]),
                source_confidence=float(scores[best]), source_box_iou=float(iou[best]),
                source_box_original=source_boxes[best].tolist(),
                source_top300_reproduction_exact=True,
                observed_pixels=int(support.sum()), fallback_pixels=int((~support).sum()),
                target_support_pixels=int(target_support.sum()),
                target_support_observed_pixels=int((target_support & support).sum()),
                target_support_fallback_pixels=int((target_support & ~support).sum()),
                local_prototype_shape=list(local_proto.shape), source_box_crop_applied=False)
            yield dict(candidate_index=index, logit_input=mapped,
                       support_input=support, metadata=meta,
                       same_view_coeff=same_view_coeff, same_view_source_meta=same_view_meta)
        self.assert_unchanged()

    def predict(self, image, deadline_monotonic=None):
        rows = list(self.infer(image, deadline_monotonic=deadline_monotonic))
        if not rows:
            return dict(logits=torch.empty((0, 640, 640), device=self.device),
                        support=torch.empty((0, 640, 640), dtype=torch.bool, device=self.device), metadata=[],
                        same_view_coeff=torch.empty((0, 32), device=self.device), same_view_source_meta=[])
        return dict(logits=torch.stack([r["logit_input"] for r in rows]),
                    support=torch.stack([r["support_input"] for r in rows]),
                    metadata=[r["metadata"] for r in rows],
                    same_view_coeff=torch.stack([r["same_view_coeff"] for r in rows]),
                    same_view_source_meta=[r["same_view_source_meta"] for r in rows])
