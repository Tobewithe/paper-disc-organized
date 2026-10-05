"""Sequential 7Q queue; every retry/seed has a distinct, preserved run."""
import json,shutil,subprocess,sys,time,uuid
from datetime import datetime,timezone
from common import *

def stamp():return datetime.now(timezone.utc).isoformat()
def stage(name,script,args):
    state=json.loads((ROOT/"QUEUE.json").read_text()) if (ROOT/"QUEUE.json").exists() else {"runs":{}}
    if name in state["runs"]:
        previous=ROOT/"runs"/state["runs"][name]
        rec=json.loads((previous/"run.json").read_text())
        if rec.get("status")=="completed" and (previous/"COMPLETE.json").exists():return previous
        raise RuntimeError(f"Existing unfinished stage {name}; inspect {previous}")
    rid="RUN_"+uuid.uuid4().hex
    out=ROOT/"runs"/rid;out.mkdir(parents=True)
    shutil.copytree(ROOT/"scripts",out/"execution_source",ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT/"PROTOCOL.md",out/"PROTOCOL.md")
    command=[sys.executable,"-u",str(out/"execution_source"/script),*map(str,args),"--out",str(out)]
    record(out,name,"running",started_at=stamp(),command=command)
    state["runs"][name]=rid;write(ROOT/"QUEUE.json",state)
    print(json.dumps(dict(stage=name,run_id=rid)),flush=True)
    with (out/"stdout.log").open("w") as log,(out/"stderr.log").open("w") as err:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=err)
    good=result.returncode==0 and (out/"COMPLETE.json").exists()
    record(out,name,"completed" if good else "failed",ended_at=stamp(),return_code=result.returncode)
    if not good:raise RuntimeError(f"Failed {name}; {out}")
    return out

def main():
    stage("smoke_data","build_train.py",["--count",16])
    data=stage("fit_dev_data","build_train.py",[])
    for seed in (0,1):
        for mode in MODES:
            for kind in KINDS:
                stage(f"train_{mode}_{kind}_s{seed}","train.py",
                    ["--data",data,"--mode",mode,"--kind",kind,"--seed",seed])
    write(ROOT/"TRAINING_COMPLETE.json",dict(time=stamp(),models=20,data=str(data)))
    print("TRAINING COMPLETE; test evaluation is a separate preserved run",flush=True)

if __name__=="__main__":main()

