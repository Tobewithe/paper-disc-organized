import json,sys,time,subprocess,uuid
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/"runs/RUN_f06a565538da4bcc892ae289dc2851db"
stats=ROOT/"runs"/("RUN_"+uuid.uuid4().hex)
fig=ROOT/"runs"/("RUN_"+uuid.uuid4().hex)
def write(p,obj):p.write_text(json.dumps(obj,indent=2),encoding="utf-8")
write(ROOT/"FOLLOWUP_RUNS.json",dict(source=str(source),stats=str(stats),figure=str(fig),status="waiting_for_decomposition"))
while not (source/"COMPLETE.json").exists():
    if (source/"FAILED.json").exists():raise RuntimeError("decomposition failed; see FAILED.json")
    time.sleep(5)
stats.mkdir(parents=True,exist_ok=True)
with (stats/"console.log").open("w",encoding="utf-8") as log:
    subprocess.run([sys.executable,"-u",str(ROOT/"scripts/summarize.py"),"--source",str(source),"--out",str(stats)],
                   stdout=log,stderr=subprocess.STDOUT,check=True)
fig.mkdir(parents=True,exist_ok=True)
write(fig/"run.json",dict(run_id=fig.name,status="running",started_at=datetime.now(timezone.utc).isoformat(),source=str(stats)))
with (fig/"console.log").open("w",encoding="utf-8") as log:
    subprocess.run([sys.executable,"-u",str(ROOT/"scripts/plot_results.py"),"--stats",str(stats),"--out",str(fig)],
                   stdout=log,stderr=subprocess.STDOUT,check=True)
record=json.loads((fig/"run.json").read_text())
write(fig/"run.json",dict(record,status="completed",finished_at=datetime.now(timezone.utc).isoformat()))
write(ROOT/"FOLLOWUP_RUNS.json",dict(source=str(source),stats=str(stats),figure=str(fig),status="completed"))
print(json.dumps(dict(stats=str(stats),figure=str(fig))),flush=True)
