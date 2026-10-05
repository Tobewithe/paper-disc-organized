"""Package only the frozen COCO image lists and experiment sources for remote use."""
import json
from pathlib import Path
import random
import tarfile

root = Path(__file__).resolve().parents[3]
study = Path(__file__).resolve().parents[1]
data = root / "assets/datasets/coco"
rng = random.Random(20260924)
train = sorted(int(p.stem) for p in (data / "images/train2017").glob("*.jpg"))
val = sorted(int(p.stem) for p in (data / "images/val2017").glob("*.jpg"))
rng.shuffle(train)
rng.shuffle(val)
split = {"fit": sorted(train[:800]), "dev": sorted(train[800:1000]), "val": sorted(val[:200])}
assert set(split["fit"]).isdisjoint(split["dev"])
assert set(split["fit"] + split["dev"]).isdisjoint(split["val"])
(study / "SPLIT.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
with tarfile.open(study / "remote_input.tar.gz", "w:gz", compresslevel=1) as archive:
    for group in ("fit", "dev", "val"):
        source = "val2017" if group == "val" else "train2017"
        for iid in split[group]:
            relative = f"images/{source}/{iid:012d}.jpg"
            archive.add(data / relative, arcname="data/" + relative)
    for source in ("train2017", "val2017"):
        relative = f"annotations/instances_{source}.json"
        archive.add(data / relative, arcname="data/" + relative)
    for name in ("SPLIT.json", "study.json", "PROTOCOL.md"):
        archive.add(study / name, arcname=name)
    for script in (study / "scripts").glob("*.py"):
        archive.add(script, arcname="scripts/" + script.name)
    archive.add(root / "assets/models/coco_clean_20260911/yolo26m-seg.pt", arcname="yolo26m-seg.pt")
    archive.add(Path(r"C:\Dpan\codexproject\Aggregation Workbench\workbench\research\runner.py"), arcname="runner.py")
print(json.dumps({"fit":len(split["fit"]),"dev":len(split["dev"]),"val":len(split["val"]),
                  "bytes":(study / "remote_input.tar.gz").stat().st_size}))
