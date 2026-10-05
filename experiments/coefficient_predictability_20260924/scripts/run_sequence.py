"""Run one fixed diagnostic/probe sequence independently of the SSH connection."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid

root=Path(__file__).resolve().parents[1]
state={"started_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"stages":[],"status":"running"}


def run_stage(name,script,args,inputs):
    run_id="RUN_"+uuid.uuid4().hex
    out=root/"runs"/run_id
    entry={"stage":name,"run_directory":str(out),"status":"running"}
    state["stages"].append(entry)
    (root/"QUEUE.json").write_text(json.dumps(state,indent=2))
    cmd=[sys.executable,str(root/"runner.py"),"--study","STUDY_67821036f497467aa3baf93c4fb9ee30",
         "--output",str(out),"--cwd",str(root),"--snapshot",str(root/"scripts"/script),
         "--snapshot",str(root/"PROTOCOL.md"),"--expect",str(out/"COMPLETE.json")]
    for path in inputs:
        cmd.extend(["--input",str(path)])
    cmd.extend(["--",sys.executable,"-u",str(root/"scripts"/script),*map(str,args),"--out",str(out)])
    code=subprocess.call(cmd)
    entry["status"]="completed" if code==0 else "failed"
    entry["return_code"]=code
    if code:
        state["status"]="failed"
    (root/"QUEUE.json").write_text(json.dumps(state,indent=2))
    if code:
        sys.exit(code)
    return out


bank=run_stage("cache","native_bank.py",["--data",root/"data","--weights",root/"yolo26m-seg.pt",
    "--split",root/"SPLIT.json"],[root/"SPLIT.json",root/"yolo26m-seg.pt"])
oracle=run_stage("individual_and_joint_oracle","oracle.py",["--bank",bank,"--annotations",
    root/"data/annotations/instances_val2017.json","--each-group","100"],[bank/"COMPLETE.json"])
probe=run_stage("h_and_prototype_probes","feature_probes.py",["--bank",bank,"--annotations",
    root/"data/annotations/instances_val2017.json","--epochs","30"],[bank/"COMPLETE.json"])
state["status"]="completed"
state["finished_at"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
(root/"QUEUE.json").write_text(json.dumps(state,indent=2))
