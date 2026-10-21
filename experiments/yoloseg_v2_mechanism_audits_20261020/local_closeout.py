"""Post-run independent table/config/transfer checks. No model execution."""
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
R=ROOT/"revisions/r1"
V2=ROOT.parent/"yoloseg_blender_scale_controlled_20261020"
BASE=V2/"runs/RUN_YB2_FULL_9154f356ba5f435f8c73db254817abb3"


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def lines(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def closeout(kinds=("chain","shift")):
    result={"status":"passed","training_steps":0,"checks":{}}
    spec=importlib.util.spec_from_file_location("frozen_stats",R/"scripts/summary_stats.py")
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    source_count=0
    for revision in [ROOT,R]:
        for item in read(revision/"FREEZE.json")["files"]:
            assert sha(revision/item["path"])==item["sha256"]
            source_count+=1
    result["frozen_source_checks"]=source_count
    native_args=read(BASE/"native/NATIVE_ARGS.json")
    baseline={(x["image_id"],x["track_id"]):x for x in lines(BASE/"TIMELINE.jsonl")}
    for kind in kinds:
        transfer=read(R/f"{kind.upper()}_TRANSFER.json")
        for item in transfer:
            p=R/item["path"]
            assert p.stat().st_size==item["size"] and sha(p)==item["sha256"],str(p)
        receipt={"transfer_files":len(transfer),"runs":[]}
        for stage in ["SMOKE","FULL"]:
            status=read(R/f"{kind.upper()}_{stage}_STATUS.json")
            run=R/"runs"/status["run_id"]
            assert status["execution_status"]=="completed"
            assert status==read(run/"run.json")
            assert status["training_steps"]==0 and status["protocol_sha256"]==sha(R/"PROTOCOL.md")
            args=list(run.glob("native*/NATIVE_ARGS.json"))
            assert len(args)==(1 if kind=="chain" else 9)
            for p in args:assert read(p)==native_args
            summary=read(run/"SUMMARY.json")
            assert frozen.summarize(run,kind)==summary
            assert read(run/"INDEPENDENT_VERIFICATION.json")["status"]=="passed"
            independent={}
            if kind=="chain":
                rows=lines(run/"CHAIN_ROWS.jsonl")
                for row in rows:
                    base=baseline[row["image_id"],row["track_id"]]
                    assert row["correct_class"]==base["class_correct"]
                    assert row["source_output_id"]==base.get("output_detection_id")
                failures=[r for r in rows if not r["correct_class"]]
                histogram=collections.Counter(str(r["first_failure"]) for r in failures)
                assert dict(histogram)==summary["first_failure"]
                pairs=lines(run/"FACTORIAL_ROWS.jsonl")
                checks=0
                for contrast in ["near_clear","front_rear"]:
                    valid=[p for p in pairs if p["eligible"] and p["contrast"]==contrast]
                    for anchor in ["control","pressure"]:
                        for factor,key in [("restore_P","P0c1"),("restore_c","P1c0")]:
                            br=math.fsum(p["results"][anchor]["P1c1"]["bridge_mass"]-p["results"][anchor][key]["bridge_mass"] for p in valid)/len(valid)
                            di=math.fsum(p["results"][anchor][key]["iou"]-p["results"][anchor]["P1c1"]["iou"] for p in valid)/len(valid)
                            expected=summary["factorial"][contrast]["anchors"][anchor][factor]
                            assert abs(br-expected["bridge_reduction"])<1e-12
                            assert abs(di-expected["delta_iou"])<1e-12
                            checks+=2
                independent={"baseline_identity_rows":len(rows),"failure_histogram":dict(histogram),"factorial_mean_checks":checks}
            else:
                rows=lines(run/"SHIFT_ROWS.jsonl")
                zeros={(x["image_id"],x["track_id"]):x for x in rows if x["offset"]==[0,0]}
                by_instance=collections.defaultdict(list)
                for x in rows:
                    if not x["eligible"] or sum(abs(v) for v in x["offset"])!=1:continue
                    b=zeros[x["image_id"],x["track_id"]]
                    f=(x["correct"]!=b["correct"] or x["success"]!=b["success"])
                    assert f==x["either_flip"]
                    assert abs(x["operational_iou"]-b["operational_iou"]-x["delta_iou"])<1e-12
                    by_instance[x["image_id"],x["track_id"]].append(float(f))
                assert all(len(x)==4 for x in by_instance.values())
                fraction=math.fsum(math.fsum(v)/4 for v in by_instance.values())/len(by_instance)
                assert abs(fraction-summary["one_pixel_flip_fraction"])<1e-12
                independent={"eligible_instances":len(by_instance),"independent_equal_instance_flip_fraction":fraction}
            receipt["runs"].append({"run_id":status["run_id"],"native_args":len(native_args),"native_configs":len(args),
                "frozen_summary_equal":True,"independent":independent})
        result["checks"][kind]=receipt
    (ROOT/"LOCAL_VERIFICATION.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__=="__main__":
    import sys
    closeout(tuple(sys.argv[1:]) or ("chain","shift"))
