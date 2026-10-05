"""Package preselected held-out COCO images without opening their labels."""
import argparse
import json
from pathlib import Path
import tarfile


def main(a):
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    ids = [int(x) for x in manifest["independent_test_images"]]
    assert len(ids) == len(set(ids)) == 2000
    a.out.mkdir(parents=True, exist_ok=True)
    archive = a.out / "test_images.tar"
    with tarfile.open(archive, "w") as tar:
        for iid in ids:
            path = a.images / f"{iid:012d}.jpg"
            assert path.is_file(), path
            tar.add(path, arcname=f"images/val2017/{iid:012d}.jpg", recursive=False)
    (a.out / "COMPLETE.json").write_text(json.dumps(dict(images=len(ids),
        archive=str(archive), bytes=archive.stat().st_size), indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("manifest", "images", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    main(p.parse_args())
