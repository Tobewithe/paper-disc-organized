"""Check final evidence and deliverable integrity without changing model outputs."""
from pathlib import Path
import math
import cv2
from .common import Run, read_json, write_json, sha256


OLD = {
    "dev": "RUN_EVALUATE_DEV_20261018T230537_e9de99ae",
    "test": "RUN_EVALUATE_TEST_20261018T230813_21c1e03e",
    "demo": "RUN_EVALUATE_DEMO_20261018T231456_c77d80dd",
}


def verify(root):
    manifest = read_json(root / "input_manifest.json")
    prior = read_json(root / "input_manifest_v1.json")
    parents = [read_json(root / f"evaluation_{s}_pointer.json")["run_id"] for s in OLD]
    with Run(root, "delivery_verification", dict(purpose="post-repair evidence and video integrity; no fitting"), parents) as run:
        old_by_name = {s["name"]: s for s in prior["sequences"]}
        fixed = [s for s in manifest["sequences"] if s["split"] != "demo"]
        assert all(s == old_by_name[s["name"]] for s in fixed)
        unchanged, frames, observations, deadlines = [], 0, 0, []
        for split in OLD:
            ptr = read_json(root / f"evaluation_{split}_pointer.json")
            target = root / ptr["path"]
            cache_key = "demo_features_pointer.json" if split == "demo" else "features_pointer.json"
            cache = root / read_json(root / cache_key)["cache"]
            for seq in [s for s in manifest["sequences"] if s["split"] == split]:
                name = seq["name"]
                dets = {d["frame"]: d for d in read_json(cache / name / "detections.json")}
                for arm in read_json(root / "config.json")["arms"]:
                    p = target / arm / name / "tracks.json"
                    rows = read_json(p)
                    assert len(rows) == seq["frames"]
                    if arm in {"bytetrack", "deepocsort", "overlap_freeze", "utility_memory"}:
                        old = root / "runs" / OLD[split] / arm / name / "tracks.json"
                        assert sha256(p) == sha256(old), (split, arm, name)
                        unchanged.append(dict(split=split, arm=arm, sequence=name, sha256=sha256(p)))
                    for row in rows:
                        ids = [t["identity"] for t in row["tracks"] if t["identity"] is not None]
                        assert len(ids) == len(set(ids)), (split, arm, name, row["frame"], "duplicate identity")
                        det = dets[row["frame"]]
                        for t in row["tracks"]:
                            i = t["det_id"]
                            assert 0 <= i < len(det["boxes"])
                            assert all(math.isclose(a,b,abs_tol=1e-4) for a,b in zip(t["box"],det["boxes"][i]))
                            assert math.isclose(t["score"],det["scores"][i],abs_tol=1e-6)
                            assert not ({"gt_id", "annotation_id", "future_id"} & t.keys())
                        frames += 1
                        observations += len(row["tracks"])
                    events = read_json(target / arm / name / "policy_events.json")["recovery_events"]
                    for e in events:
                        delay = e["delay_frames"]
                        assert 0 <= delay <= 6, (split, arm, name, e)
                        deadlines.append(delay)
        summary = read_json(root / read_json(root / "evaluation_test_pointer.json")["path"] / "SUMMARY.json")
        for arm, row in summary["combined"].items():
            expected = 200 * row["IDTP"] / (2*row["IDTP"]+row["IDFP"]+row["IDFN"])
            assert math.isclose(expected, row["IDF1"], abs_tol=1e-10)
            assert row["IDTP"] + row["IDFN"] == 41875
        videos = []
        for p in sorted((root / "showcase/videos").glob("*.mp4")):
            cap = cv2.VideoCapture(str(p)); expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            count = 0
            while True:
                ok, img = cap.read()
                if not ok: break
                assert img.shape[:2] == (1080,1920)
                count += 1
            cap.release()
            assert count == expected and count > 0, (p.name, count, expected)
            videos.append(dict(name=p.name,decoded_frames=count,sha256=sha256(p)))
        assert len(videos) == 5
        jobs = [dict(job=p.parent.name, **read_json(p)) for p in (root / "jobs").glob("*/job.json")]
        assert any(j["status"] == "completed" and j.get("frames") == 15 for j in jobs)
        result = dict(status="passed",unchanged_input_sequences=len(fixed),
                      unchanged_arm_files=unchanged,output_frames_checked=frames,
                      observations_checked=observations,recovery_events_checked=len(deadlines),
                      maximum_observed_confirmation_delay=max(deadlines),
                      identity_unique_per_frame=True,outputs_from_original_detections=True,
                      IDF1_recomputed_from_components=True,test_GT_detections=41875,
                      video_decode=videos,import_jobs=jobs,
                      scope="engineering and evidence integrity, not method efficacy")
        write_json(run.path / "SUMMARY.json", result)
        write_json(run.path / "DECISION.json", dict(status="engineering_passed",scientific_claim=False,
                   next_action="deliver completed pilot; retain negative efficacy conclusion"))
        write_json(root / "delivery_verification_pointer.json",dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
        print({k:v for k,v in result.items() if k not in {"unchanged_arm_files","video_decode","import_jobs"}},flush=True)


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();verify(a.root)
