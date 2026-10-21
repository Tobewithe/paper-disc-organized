"""Post-run verification/preview packaging only; no model or scientific edits."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from pycocotools import mask as mask_api

ROOT=Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def alternate_topology(gt,p,foreign):
    gn,gl=cv2.connectedComponents(gt.astype(np.uint8),connectivity=8)
    pn,pl=cv2.connectedComponents(p.astype(np.uint8),connectivity=8)
    split_pixels=0; missed=split=0; touching=collections.defaultdict(set)
    for part in range(1,gn):
        labels,counts=np.unique(pl[gl==part],return_counts=True)
        positive={int(i):int(n) for i,n in zip(labels,counts) if i>0}
        missed+=not positive;split+=len(positive)>1
        split_pixels+=sum(positive.values())-max(positive.values(),default=0)
        for i in positive:touching[i].add(part)
    touched=np.unique(pl[gt]);touched=touched[touched>0]
    bridge_pixels=int(np.count_nonzero(np.isin(pl,touched)&foreign))
    return dict(gt_components=gn-1,prediction_components=pn-1,
        split_pixels=split_pixels,bridge_pixels=bridge_pixels,missed_gt_components=missed,
        split_gt_components=split,within_instance_false_joins=sum(len(v)>1 for v in touching.values()),
        unmatched_prediction_islands=pn-1-len(touching),
        material_split=split_pixels>=.01*gt.sum(),material_bridge=bridge_pixels>=.01*gt.sum())


def main(run):
    run=Path(run)
    assert read(run/"run.json")["execution_status"]=="completed"
    manifest=read(run/"ARTIFACT_MANIFEST.json")
    for r in manifest:
        p=run/r["path"]
        assert p.stat().st_size==r["size"] and sha(p)==r["sha256"],r["path"]
    freeze=read(ROOT/"FREEZE.json")
    for r in freeze["files"]:assert sha(ROOT/r["path"])==r["sha256"],r["path"]
    native_args=read(run/"native/NATIVE_ARGS.json");profile=read(ROOT/"PROFILE.json")
    v1=ROOT.parent/"yoloseg_blender_stress_20261020/runs/RUN_YB_SMOKE_c6f0333ccc994518bdf0790b52265526"
    assert native_args==read(v1/"native/NATIVE_ARGS.json")
    assert all(native_args[k]==v for k,v in profile["native_args"].items())
    env=read(run/"ENVIRONMENT.json")["native"]
    assert env["source_sha256"]==read(v1/"ENVIRONMENT.json")["native"]["source_sha256"]
    assert all(env[k]==v for k,v in profile["runtime_policy"].items())
    identity=[json.loads(x) for x in (run/"NATIVE_IDENTITY.jsonl").read_text().splitlines()]
    images=[i for batch in identity for i in batch["images"]]
    assert len(images)==576 and len({i["image_id"] for i in images})==576
    assert all(i["geometry"]["input_hw"]==[416,672] and i["geometry"]["original_hw"]==[384,640] for i in images)
    replay=read(run/"NATIVE_REPLAY.json")
    assert replay["frames"]==576 and replay["all_exact_zero_delta_RLE"]
    assert replay["weights_sha256"]==profile["model_sha256"]
    timeline=[json.loads(x) for x in (run/"TIMELINE.jsonl").read_text().splitlines()]
    predictions={r["output_detection_id"]:r for r in read(run/"PREDICTIONS.json")}
    by_image=collections.defaultdict(list)
    for row in timeline:by_image[row["image_id"]].append(row)
    topology_fields=0
    for iid,rows in by_image.items():
        label=np.load(run/"dataset/labels"/f"{iid:012d}.npz")["instance_id"]
        for row in rows:
            if not row["class_correct"]:continue
            gt=label==row["track_id"];foreign=(label>=1)&(label<=3)&~gt
            p=mask_api.decode(predictions[row["output_detection_id"]]["segmentation"]).astype(bool)
            check=alternate_topology(gt,p,foreign)
            for k,v in check.items():
                assert v==row["mask_metrics"][k],(iid,row["track_id"],k,v,row["mask_metrics"][k])
                topology_fields+=1
    summary=read(run/"SUMMARY.json")
    table=[]
    for variant in ["clear","near","rear","front"]:
        rows=[r for r in timeline if r["variant"]==variant]
        row=dict(variant=variant,all_n=len(rows),all_correct=sum(r["class_correct"] for r in rows),
                 all_failures=sum(r["segmentation_failure"] for r in rows))
        for scope,rs in [("all",rows),("focus",[r for r in rows if r["track_id"]==1])]:
            row.update({scope+"_n":len(rs),scope+"_correct":sum(r["class_correct"] for r in rs),
                        scope+"_failures":sum(r["segmentation_failure"] for r in rs),
                        scope+"_operational_iou":sum(r["operational_iou"] for r in rs)/len(rs),
                        scope+"_bridge":sum(bool(r.get("mask_metrics",{}).get("material_bridge")) for r in rs),
                        scope+"_split":sum(bool(r.get("mask_metrics",{}).get("material_split")) for r in rs)})
        table.append(row)
    save(ROOT/"FINAL_NUMBERS.json",dict(table=table,comparisons=summary["paired_comparisons"],
                                         domain_gate=summary["domain_gate"]))
    save(ROOT/"LOCAL_CONFIGURATION_AND_TRANSFER.json",dict(status="passed",run_id=run.name,
        transferred_files=len(manifest),transferred_bytes=sum(r["size"] for r in manifest),
        all_manifest_files_verified=True,all_111_native_args_equal_v1=True,
        source_pins_equal_v1=True,runtime_policy_exact=True,input_shapes=[[416,672]],
        native_rle_predictions=replay["predictions"],independent_topology_fields=topology_fields,
        freeze_sha256=sha(ROOT/"FREEZE.json"),training_steps=0,verification_model_forwards=0,
        verification_python=sys.version,verification_numpy=np.__version__,verification_opencv=cv2.__version__))
    # Playback derivatives retain the exact order and timing of all three seeds.
    videos=ROOT/"videos";videos.mkdir(exist_ok=True);ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    receipts=[]
    for control,pressure in [("clear","near"),("rear","front")]:
        sources=[run/"previews"/f"compare_{seed}_{control}_{pressure}.mp4" for seed in (20261020,20261021,20261022)]
        listing=videos/f"{control}_{pressure}.txt"
        listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in sources),encoding="utf-8")
        output=videos/f"{control}_vs_{pressure}_all_seeds.mp4"
        command=[ffmpeg,"-y","-v","error","-f","concat","-safe","0","-i",str(listing),
                 "-an","-c:v","libx264","-crf","18","-preset","fast","-pix_fmt","yuv420p",
                 "-movflags","+faststart",str(output)]
        subprocess.run(command,check=True)
        capture=cv2.VideoCapture(str(output))
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==144
        assert capture.get(cv2.CAP_PROP_FPS)==12
        capture.release()
        receipts.append(dict(command=command,sources=[dict(path=str(p),sha256=sha(p)) for p in sources],
                             output=str(output),sha256=sha(output),frames=144,fps=12))
    save(videos/"DERIVATION.json",dict(scope="presentation only; no data/mask/metric edits",videos=receipts))
    print(json.dumps(read(ROOT/"LOCAL_CONFIGURATION_AND_TRANSFER.json")),flush=True)


if __name__=="__main__":
    main(sys.argv[1])
