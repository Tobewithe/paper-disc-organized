"""Finish verification/figure only after scientific evaluation succeeds."""
import time
from common import *
from execute import stage
while not (ROOT/"COMPLETE.json").exists():
    state=json.loads((ROOT/"QUEUE.json").read_text()) if (ROOT/"QUEUE.json").exists() else {}
    for name,rid in state.get("runs",{}).items():
        p=ROOT/"runs"/rid/"run.json"
        if p.exists() and json.loads(p.read_text()).get("status")=="failed":
            raise RuntimeError(f"Stage failed: {name} {rid}")
    time.sleep(20)
complete=json.loads((ROOT/"COMPLETE.json").read_text())
audit=stage("completion_audit","completion_audit.py",[])
figure=stage("figure","plot_results.py",["--summary",complete["summary"]])
write(ROOT/"VERIFIED.json",dict(audit=str(audit),figure=str(figure),**complete))
print("7Q VERIFIED; human interpretation and local artifact return remain",flush=True)

