"""Keep only GT-free IDs, predicted level and box scale for matched 7J-N controls."""
import argparse
import json
from pathlib import Path

import torch


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    result = {}
    for split in ("fit", "dev", "val"):
        source = getattr(a, split)
        data = torch.load(source / "FEATURES.pt", weights_only=True, map_location="cpu")
        rows = data["rows"]
        compact = [dict(image_id=int(r["image_id"]), annotation_id=int(r["annotation_id"]),
                        raw_id=int(r["raw_id"]), level=int(r["level"]),
                        basic=r["basic"].clone()) for r in rows]
        out = a.out / split
        out.mkdir()
        torch.save(dict(rows=compact), out / "METADATA.pt")
        result[split] = dict(instances=len(compact), images=len({r["image_id"] for r in compact}),
                             source=str(source))
        del data, rows, compact
    (a.out / "COMPLETE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("fit", "dev", "val", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    main(p.parse_args())
