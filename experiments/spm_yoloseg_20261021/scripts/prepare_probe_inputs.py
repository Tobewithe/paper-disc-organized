"""Prepare a bounded, source-train-only input bundle without changing source files."""
import argparse
import hashlib
import json
import pathlib
import tarfile
from collections import defaultdict


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    source, images, out = pathlib.Path(a.source), pathlib.Path(a.images), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    videos = defaultdict(list)
    for im in data["images"]:
        stem = pathlib.Path(im["file_name"]).stem
        videos[stem.rsplit("-", 1)[0]].append(im)
    selected = []
    for video, rows in sorted(videos.items()):
        rows.sort(key=lambda im: (int(pathlib.Path(im["file_name"]).stem.rsplit("-", 1)[1]), im["id"]))
        n = min(6, len(rows))
        indices = sorted({round(i * (len(rows) - 1) / max(n - 1, 1)) for i in range(n)})
        for i in indices:
            im = dict(rows[i])
            src = images / im["file_name"]
            assert src.is_file(), src
            im.update(source_file_name=im["file_name"], file_name=f"images/{im['id']}.jpg",
                      source_sha256=sha(src), video_id=video,
                      group_id=video.split("s5")[0])
            selected.append(im)
    ids = {im["id"] for im in selected}
    payload = dict(data)
    payload["images"] = selected
    payload["annotations"] = [ann for ann in data["annotations"] if ann["image_id"] in ids]
    ann_path = out / "annotations.json"
    ann_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest = {"source_annotation": str(source), "source_annotation_sha256": sha(source),
                "source_images": len(data["images"]), "source_videos": len(videos),
                "selection": "six evenly spaced source-train frames per video; no feature/score selection",
                "selected_images": len(selected), "annotations": len(payload["annotations"]),
                "group_policy": "filename prefix before s5, merging clips/style variants; inferred family, not verified pen",
                "historical_test_included": False, "source_geometry_policy": "preserve polygons including out-of-bounds; rasterizer clips at canvas",
                "annotations_sha256": sha(ann_path), "image_entries": selected}
    (out / "INPUT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with tarfile.open(out / "input_bundle.tar", "w") as tar:
        tar.add(ann_path, arcname="annotations.json")
        tar.add(out / "INPUT_MANIFEST.json", arcname="INPUT_MANIFEST.json")
        for im in selected:
            tar.add(images / im["source_file_name"], arcname=im["file_name"])
    print(json.dumps({k: v for k, v in manifest.items() if k != "image_entries"}, ensure_ascii=False))
    print("bundle_bytes", (out / "input_bundle.tar").stat().st_size)


if __name__ == "__main__":
    main()
