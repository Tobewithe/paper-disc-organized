"""Small stdlib readers for immutable native-comparison scoring inputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ARMS = ("baseline", "RCMC_full", "global_minus025_full", "RCMC_first64", "global_minus025_first64", "multi_local_full", "multi_local_first64")
COCO_KEYS = ("image_id", "category_id", "bbox", "score", "segmentation")
IDENTITY_KEYS = ("image_id", "category_id", "bbox", "score", "detection_index", "raw_input_box_xyxy", "raw_confidence", "box_xyxy", "model_class")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    target = Path(path)
    temporary = target.with_name(target.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(target)


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def json_array(path):
    decoder, buffer, position, first, value_next = json.JSONDecoder(), "", 0, True, True
    with Path(path).open(encoding="utf-8") as f:
        buffer = f.read(1024*1024)
        position = len(buffer)-len(buffer.lstrip())
        check(position < len(buffer) and buffer[position] == "[", "COCO array opener is missing")
        position += 1
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                buffer, position = f.read(1024*1024), 0
                check(bool(buffer), "Truncated COCO array")
            if buffer[position] == "]":
                check(first or not value_next, "COCO array trailing comma")
                check(not buffer[position+1:].strip() and not f.read().strip(), "Unexpected trailing COCO array data")
                return
            if not value_next:
                check(buffer[position] == ",", "COCO record separator missing")
                position += 1
                value_next = True
                continue
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                extra = f.read(1024*1024)
                check(bool(extra), "Malformed COCO record")
                buffer, position = buffer[position:]+extra, 0
                continue
            check(isinstance(value, dict) and set(COCO_KEYS).issubset(value), "Required COCO detection fields missing")
            yield value
            position, first, value_next = end, False, False
            if position > 1024*1024:
                buffer, position = buffer[position:], 0


class InputLock:
    def __init__(self):
        self.files = {}

    def add(self, path, known_bytes=None):
        path = Path(path).resolve()
        if str(path) not in self.files:
            data_sha = hashlib.sha256(known_bytes).hexdigest() if known_bytes is not None else sha(path)
            self.files[str(path)] = {"sha256_before": data_sha, "bytes": path.stat().st_size}
        return self.files[str(path)]["sha256_before"]

    def finish(self):
        for path, evidence in self.files.items():
            evidence["sha256_after"] = sha(path)
            check(evidence["sha256_before"] == evidence["sha256_after"], "Source bytes changed: "+path)
        return self.files


class ArmReader:
    def __init__(self, arm, lock, ids):
        self.arm, self.lock = Path(arm), lock
        self.predictions = self.arm/"predictions.json"
        self.predictions_sha256 = lock.add(self.predictions)
        self.receipt = read_json(self.arm/"COMPLETE.json")
        lock.add(self.arm/"COMPLETE.json")
        check(self.receipt.get("status") == "prediction_complete" and self.receipt.get("image_count") == len(ids), "Incomplete arm receipt: "+self.arm.name)
        check(self.receipt.get("predictions_sha256") == self.predictions_sha256, "Arm prediction SHA differs: "+self.arm.name)
        self.mode = "per_image_native_fields" if (self.arm/"images"/f"{ids[0]:012d}.json").is_file() else "producer_assembly_ordinal"
        self.stream = iter(json_array(self.predictions))
        self.images = 0

    def read(self, iid, count):
        exported = [next(self.stream, None) for _ in range(count)]
        check(all(row is not None and row["image_id"] == iid for row in exported), "Arm assembly order/count differs: "+self.arm.name)
        if self.mode == "per_image_native_fields":
            path = self.arm/"images"/f"{iid:012d}.json"
            data = path.read_bytes()
            self.lock.add(path, data)
            value = json.loads(data)
            rows = value["detections"]
            check(value["image_id"] == iid and len(rows) == count, "Arm image identity/count differs")
            for i, row in enumerate(rows):
                check(row.get("detection_index") == i and all(row[k] == exported[i][k] for k in COCO_KEYS), "Arm native index/assembled COCO fields differ")
        else:
            rows = exported
        self.images += 1
        return rows

    def finish(self, image_count):
        check(self.images == image_count and next(self.stream, None) is None, "Arm complete sequence cardinality differs: "+self.arm.name)


def encoded_rle(rle):
    return rle | {"counts": rle["counts"].encode("ascii") if isinstance(rle["counts"], str) else rle["counts"]}


def rle_area(rle, mask_utils=None):
    if mask_utils is not None:
        return int(mask_utils.area(encoded_rle(rle)))
    counts = rle["counts"]
    if isinstance(counts, str):
        decoded, p = [], 0
        while p < len(counts):
            x, shift = 0, 0
            while True:
                check(p < len(counts), "Truncated compressed RLE")
                c = ord(counts[p])-48
                p += 1
                check(0 <= c < 64, "Invalid compressed RLE character")
                x |= (c & 31) << shift
                shift += 5
                if not c & 32:
                    if c & 16:
                        x |= -1 << shift
                    break
            if len(decoded) > 2:
                x += decoded[-2]
            decoded.append(x)
        counts = decoded
    check(isinstance(counts, list) and all(type(x) is int and x >= 0 for x in counts), "Invalid RLE counts")
    check(sum(counts) == rle["size"][0]*rle["size"][1], "RLE pixel support size differs")
    return sum(counts[1::2])
