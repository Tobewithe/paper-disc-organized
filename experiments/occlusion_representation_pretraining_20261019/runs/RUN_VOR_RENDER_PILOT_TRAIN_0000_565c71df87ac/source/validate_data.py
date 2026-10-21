from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

from .common import ROOT, Run, read, save, sha


def validate(dataset):
    dataset = Path(dataset)
    families = sorted(p for p in dataset.iterdir() if p.is_dir())
    reports, files, seen = [], [], set()
    for family in families:
        assert read(family/"COMPLETE.json")["frames"] == 144
        cfg = read(family/"config.json")
        assert cfg["id"] not in seen
        seen.add(cfg["id"])
        tracks, counter = [], Counter()
        for variant in range(3):
            folder = family/str(variant)
            a = dict(np.load(folder/"tracks.npz"))
            assert a["xy"].shape == (48,24,2)
            assert np.isfinite(a["xy"]).all() and a["valid"].all()
            assert a["reprojection_error"].max() <= .5
            assert set(np.unique(a["state"])).issubset({0,1,2,3})
            assert np.array_equal(a["active"],a["state"][0] == 0)
            assert ((a["instance_ratio"] == -1) == ~a["instance_inframe"]).all()
            assert ((a["instance_ratio"][a["instance_inframe"]] >= 0) &
                    (a["instance_ratio"][a["instance_inframe"]] <= 1)).all()
            for f in range(48):
                with Image.open(folder/"rgb"/f"{f:03d}.png") as im:
                    assert im.size == (512,320)
                labels = np.load(folder/"labels"/f"{f:03d}.npz")
                assert labels["owner"].shape == (80,128)
                assert np.isfinite(labels["ray_depth"]).all()
                for owner in range(1,4):
                    assert not np.any((labels["owner"] == owner) & (labels[f"amodal_{owner}"] == 0))
            events = read(folder/"events.json")
            counter.update(dict(events=len(events),
                external_return=sum(x["physical_external_only"] and x["gap"] >= 3 and
                                    x["reappear"] is not None for x in events),
                long_return=sum(x["gap"] >= 13 and x["reappear"] is not None for x in events),
                external_long_return=sum(x["physical_external_only"] and x["gap"] >= 13
                                         and x["reappear"] is not None for x in events),
                censored=sum(x["right_censored"] for x in events)))
            tracks.append(a)
        assert np.array_equal(tracks[0]["state"],tracks[1]["state"])
        assert np.allclose(tracks[0]["xy"],tracks[1]["xy"],atol=1e-7)
        assert np.array_equal(tracks[0]["world"],tracks[2]["world"])
        for p in sorted(family.rglob("*")):
            if p.is_file():
                files.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
        reports.append(dict(family=cfg["id"],split=cfg["split"],
                            event_type=cfg["event_type"],**counter,
                            active_queries=[int(x["active"].sum()) for x in tracks],
                            projection_max=max(float(x["reprojection_error"].max()) for x in tracks)))
    return reports,files


def main(args):
    with Run("validate_"+args.mode,dict(dataset=args.dataset,training_steps=0)) as run:
        reports,files = validate(args.dataset)
        if args.mode == "pilot":
            expected={x["id"]:x for x in read(ROOT/"pilot_manifest.json")}
            assert set(expected)=={x["family"] for x in reports}
            for name,cfg in expected.items():
                assert read(Path(args.dataset)/name/"config.json")==cfg
            assert len(reports)==280
        save(run.path/"DATA_MANIFEST.json",files)
        save(run.path/"SUMMARY.json",dict(valid=True,families=reports,
             total_frames=len(reports)*144,scene_ray_crosschecks=len(reports)*144*24))
        save(ROOT/("validation_"+args.mode+".json"),dict(valid=True,run_id=run.id,
             dataset=args.dataset,manifest=str(run.path/"DATA_MANIFEST.json"),
             manifest_sha256=sha(run.path/"DATA_MANIFEST.json")))

