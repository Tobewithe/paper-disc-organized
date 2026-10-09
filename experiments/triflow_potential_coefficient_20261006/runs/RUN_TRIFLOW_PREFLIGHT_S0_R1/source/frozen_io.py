"""Exact frozen YOLO26 one2one inputs for TriFlow, without GT at inference."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

OFFICIAL_SHA256 = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
EXTRACTOR_VERSION = "triflow_frozen_io_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def state_digest(model):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((str(value.dtype), tuple(value.shape))).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def box_iou_xyxy(boxes1, boxes2):
    """Numpy box IoU, stable caller-defined ordering, no class/scores."""
    import numpy as np
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    a = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    b = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = np.maximum(b - a, 0).prod(-1)
    area1 = np.maximum(boxes1[:, 2:] - boxes1[:, :2], 0).prod(-1)
    area2 = np.maximum(boxes2[:, 2:] - boxes2[:, :2], 0).prod(-1)
    return intersection / np.maximum(area1[:, None] + area2[None, :] - intersection, 1e-12)


def predicted_neighbors(boxes, raw_indices, scores=None, minimum_confidence=.001, k=2):
    """Top overlapping predicted boxes only, excluding duplicate raw identities.

    Scores only define the native eligible prediction set. Ranking is box IoU
    followed by stable output row, and never depends on GT.
    """
    import numpy as np
    boxes = np.asarray(boxes)
    raw_indices = np.asarray(raw_indices).reshape(-1)
    overlaps = box_iou_xyxy(boxes, boxes)
    eligible = np.ones(len(boxes), dtype=bool) if scores is None else np.asarray(scores) >= minimum_confidence
    rows = np.full((len(boxes), k), -1, dtype=np.int64)
    valid = np.zeros((len(boxes), k), dtype=bool)
    for row in range(len(boxes)):
        choices = [j for j in range(len(boxes)) if eligible[j] and raw_indices[j] != raw_indices[row]
                   and overlaps[row, j] > 0]
        choices.sort(key=lambda j: (-float(overlaps[row, j]), j))
        for slot, j in enumerate(choices[:k]):
            rows[row, slot], valid[row, slot] = j, True
    return rows, valid


class FrozenYOLO:
    """Frozen official unfused FP32 model with genuine coefficient hidden query.

    extract returns CPU tensors. `h` is sampled from the input to the final
    convolution of one2one_cv4 at the exact native raw index; it is not c0.
    Native top-k rows, including class-expanded duplicate raw indices, remain
    intact. Any failure to reconstruct native rows raises instead of fallback.
    """

    def __init__(self, weights, vendor, device="cuda", image_size=640):
        vendor = Path(vendor).resolve()
        sys.path.insert(0, str(vendor))
        import cv2
        import numpy as np
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.data.augment import LetterBox
        if ultralytics.__version__ != "8.4.100":
            raise ValueError(f"Wrong vendor version: {ultralytics.__version__}")
        if not Path(ultralytics.__file__).resolve().is_relative_to(vendor):
            raise ValueError("Imported ultralytics does not belong to declared vendor")
        if sha256(weights) != OFFICIAL_SHA256:
            raise ValueError("Official checkpoint SHA256 differs from locked protocol")
        if image_size != 640:
            raise ValueError("Protocol requires square 640 input")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(4)
        self.torch, self.np, self.cv2 = torch, np, cv2
        self.weights, self.vendor = str(Path(weights).resolve()), str(vendor)
        self.device = device
        self.model = YOLO(self.weights).model.float().eval().to(device)
        self.model.requires_grad_(False)
        self.head = self.model.model[-1]
        if len(self.model.model) != 24 or not self.head.end2end:
            raise ValueError("Expected official YOLO26m native one2one head")
        self.head.max_det, self.head.agnostic_nms = 300, False
        self.letterbox = LetterBox((640, 640), auto=False, scaleup=False, stride=32)
        self.hidden = {}
        self.hooks = []
        for level, branch in enumerate(self.head.one2one_cv4):
            def hook(module, args, level=level):
                self.hidden[level] = args[0].detach()
            self.hooks.append(branch[-1].register_forward_pre_hook(hook))
        self.initial_digest = state_digest(self.model)
        self.extracted_images = 0

    def extract(self, image_path):
        torch, np, cv2 = self.torch, self.np, self.cv2
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Cannot read image: {image_path}")
        params = self.letterbox.get_params({"img": original})
        resized = self.letterbox(image=original)
        tensor = torch.from_numpy(np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1)))
        tensor = tensor[None].to(self.device).float() / 255
        input_sha = hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()
        self.hidden.clear()
        with torch.inference_mode():
            (native, proto), raw = self.model(tensor)
            if isinstance(proto, tuple):
                proto = proto[0]
            one = raw["one2one"]
            decoded = self.head._get_decode_boxes(one).transpose(1, 2)
            scores, classes, indices = self.head.get_topk_index(one["scores"].sigmoid().transpose(1, 2), 300)
            coefficients = one["mask_coefficient"].transpose(1, 2)
            boxes = decoded.gather(1, indices.expand(-1, -1, 4))
            c0 = coefficients.gather(1, indices.expand(-1, -1, self.head.nm))
            rebuilt = torch.cat((boxes, scores, classes, c0), -1)
            if not torch.equal(native, rebuilt):
                raise RuntimeError("Native top-k detection replay is not bitwise exact")
            if set(self.hidden) != set(range(self.head.nl)):
                raise RuntimeError("Hidden-query prehooks did not cover every native level")
            hidden = torch.cat([self.hidden[level].flatten(2) for level in range(self.head.nl)], 2).transpose(1, 2)
            h = hidden.gather(1, indices.expand(-1, -1, hidden.shape[-1]))
            result = {
                "P": proto[0].cpu().clone(), "F": one["feats"][0][0].cpu().clone(),
                "h": h[0].cpu().clone(), "c0": c0[0].cpu().clone(),
                "boxes": boxes[0].cpu().clone(), "scores": scores[0, :, 0].cpu().clone(),
                "classes": classes[0, :, 0].long().cpu().clone(),
                "raw_indices": indices[0, :, 0].long().cpu().clone(),
                "output_rows": torch.arange(native.shape[1], dtype=torch.int64),
                "original_shape": list(original.shape[:2]), "letterbox": params,
                "image_path": str(Path(image_path).resolve()), "image_sha256": sha256(image_path),
                "input_sha256": input_sha, "native_replay_exact": True,
                "hidden_query_source": "one2one_cv4 final convolution input, exact native raw index",
            }
            height, width = result["P"].shape[-2:]
            result["boxes_proto"] = result["boxes"] * torch.tensor([width/640, height/640, width/640, height/640])
        for name in ("P", "F", "h", "c0", "boxes", "scores", "boxes_proto"):
            if result[name].dtype != torch.float32 or not torch.isfinite(result[name]).all():
                raise RuntimeError(f"Frozen extraction requires finite FP32 {name}")
        self.extracted_images += 1
        return result

    def verify_frozen(self):
        torch = self.torch
        final_digest = state_digest(self.model)
        no_trainable = not any(p.requires_grad for p in self.model.parameters())
        all_bn_eval = all(not module.training for module in self.model.modules()
                          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
        no_gradients = all(p.grad is None for p in self.model.parameters())
        value = {"initial_state_sha256": self.initial_digest, "final_state_sha256": final_digest,
                 "all_state_exact": self.initial_digest == final_digest,
                 "all_params_requires_grad_false": no_trainable, "all_bn_eval": all_bn_eval,
                 "all_gradients_none": no_gradients, "extracted_images": self.extracted_images,
                 "base_weights_sha256": sha256(self.weights), "vendor": self.vendor,
                 "extractor_version": EXTRACTOR_VERSION}
        value["passed"] = all(value[key] for key in ("all_state_exact", "all_params_requires_grad_false",
                                                     "all_bn_eval", "all_gradients_none"))
        if not value["passed"]:
            raise RuntimeError(f"Frozen-model integrity failure: {value}")
        return value

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def transformed_mask(mask, params, prototype_shape):
    """COCO annToMask -> exact integer LetterBox -> nearest native proto grid."""
    import cv2
    import numpy as np
    if tuple(mask.shape) != tuple(params["orig_shape"]):
        raise ValueError("COCO GT mask dimensions differ from the input image")
    mask = mask.astype(np.uint8)
    if mask.shape[::-1] != tuple(params["new_unpad"]):
        mask = cv2.resize(mask, tuple(params["new_unpad"]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.copyMakeBorder(mask, params["top"], params["bottom"], params["left"], params["right"],
                             cv2.BORDER_CONSTANT, value=0)
    h, w = prototype_shape
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def transformed_box(box_xywh, params):
    import numpy as np
    x, y, width, height = box_xywh
    box = np.asarray([x, y, x+width, y+height], dtype=np.float32)
    return box * np.asarray([params["ratio"][0], params["ratio"][1]] * 2, dtype=np.float32) + \
        np.asarray([params["left"], params["top"]] * 2, dtype=np.float32)
