"""Continue automatically after training; no model changes from test results."""
import time
from common import *
from execute import stage
while not (ROOT/"TRAINING_COMPLETE.json").exists():
    state=json.loads((ROOT/"QUEUE.json").read_text()) if (ROOT/"QUEUE.json").exists() else {}
    for name,rid in state.get("runs",{}).items():
        p=ROOT/"runs"/rid/"run.json"
        if p.exists() and json.loads(p.read_text()).get("status")=="failed":
            raise RuntimeError(f"Training stage failed: {name} {rid}")
    time.sleep(20)
test=stage("test_data","build_test.py",[])
evaluation=stage("test_evaluation","evaluate.py",["--test",test])
summary=stage("summary","summarize.py",["--evaluation",evaluation])
write(ROOT/"COMPLETE.json",dict(test=str(test),evaluation=str(evaluation),summary=str(summary)))
print("7Q COMPLETE",flush=True)

