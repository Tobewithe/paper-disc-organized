"""One-shot Slurm smoke/full runs with separate experiment locks."""
import importlib.metadata
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import uuid
from common import ROOT, read, save, sha, Inputs
from summary_stats import summarize


def freeze_check():
    manifest=read(ROOT/"FREEZE.json")
    for r in manifest["files"]:
        assert sha(ROOT/r["path"])==r["sha256"], ("changed_source",r["path"])
    for r in manifest["external_inputs"]:
        assert sha(r["path"])==r["sha256"], ("changed_input",r["path"])
    for r in read(ROOT/"INPUTS.json")["files"]:
        assert sha(r["server_path"])==r["sha256"], ("changed_V2_artifact",r["server_path"])


def main():
    kind,stage=sys.argv[1:]
    assert kind in ["chain","shift"] and stage in ["smoke","full"]
    assert os.environ.get("SLURM_JOB_ID")
    freeze_check()
    status=ROOT/f"{kind.upper()}_{stage.upper()}_STATUS.json"
    assert not status.exists(),"One-shot only; preserve previous result"
    if stage=="full":
        old=read(ROOT/f"{kind.upper()}_SMOKE_STATUS.json")
        assert old["execution_status"]=="completed"
        assert read(ROOT/"runs"/old["run_id"]/"INDEPENDENT_VERIFICATION.json")["status"]=="passed"
    lock=ROOT/f"{kind}.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    rid=f"RUN_V2_{kind.upper()}_{stage.upper()}_{uuid.uuid4().hex}"
    run=ROOT/"runs"/rid;run.mkdir(parents=True)
    meta=dict(study_id="STUDY_YOLOSEG_V2_MECHANISM_AUDITS_20261020",run_id=rid,experiment=kind,
        stage_id=stage,baseline_run_id="RUN_YB2_FULL_9154f356ba5f435f8c73db254817abb3",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),executor=socket.gethostname(),
        slurm_job_id=os.environ["SLURM_JOB_ID"],command=sys.argv,training_steps=0,
        execution_status="running",protocol_sha256=sha(ROOT/"PROTOCOL.md"),
        freeze_sha256=sha(ROOT/"FREEZE.json"),code_commit=read(ROOT/"FREEZE.json")["code_commit"])
    save(run/"run.json",meta);save(status,meta)
    save(run/"EVAL_CONTRACT.json",dict(protocol_sha256=meta["protocol_sha256"],
         inputs_sha256=sha(ROOT/"INPUTS.json"),candidate_policy="native_output_and_separate_GT_diagnostic",
         training_steps=0,branch="O2O",batch=16,actual_input_hw=[416,672],
         matching="V2_class_agnostic_max_cardinality_Box50_then_class_check",
         threshold_export=.001,threshold_operating=.25,existing_method_gain="not_tested"))
    try:
        cfg=read(ROOT/"CONFIG.json")
        for p,v in read(cfg["profile"])["packages"].items():
            assert importlib.metadata.version(p)==v,("package",p,v)
        if kind=="chain":
            from chain import execute
        else:
            from shift import execute
        execute(run,stage=="smoke")
        result=summarize(run,kind)
        save(run/"SUMMARY.json",result)
        from verify import verify
        save(run/"INDEPENDENT_VERIFICATION.json",verify(run,kind,cfg["v2_run"]))
        freeze_check()
        total=sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
        assert total<12*2**30,("disk_budget",total)
        artifacts=[dict(path=str(p.relative_to(run)),size=p.stat().st_size,sha256=sha(p),
                        scope="server_only_large" if "cache" in p.parts or "images" in p.parts else "return")
                   for p in sorted(run.rglob("*")) if p.is_file() and p.name!="run.json"]
        save(run/"ARTIFACT_MANIFEST.json",artifacts)
        meta.update(execution_status="completed",exit_code=0,artifact_bytes=total)
    except BaseException:
        (run/"ERROR.txt").write_text(traceback.format_exc(),encoding="utf-8")
        meta.update(execution_status="failed",exit_code=1)
        raise
    finally:
        meta["ended_at"]=time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save(run/"run.json",meta);save(status,meta);lock.unlink()


if __name__=="__main__":
    main()
