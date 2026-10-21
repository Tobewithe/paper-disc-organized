"""Export only this study's completed artifacts; source dataset/cache stay on laptop."""
import json, hashlib, zipfile, sys
from pathlib import Path

study = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
run_id = read(study / "LATEST_RUN.json")["run_id"]
run = study / "runs" / run_id
assert read(run / "run.json")["execution_status"] == "completed"
assert read(run / "VISUALIZATION_RECEIPT.json")["status"] == "completed"
assert not (study / "RUNNING.lock").exists()
files = [p for p in sorted(study.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
         and p.name not in ["ARTIFACT_MANIFEST.json"] and p.suffix != ".zip"]
total = sum(p.stat().st_size for p in files)
assert total < 1024**3
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):h.update(block)
    return h.hexdigest()
manifest = dict(run_id=run_id, files=[dict(relative_path=p.relative_to(study).as_posix(),
                    size_bytes=p.stat().st_size, sha256=sha(p), storage_scope="laptop_and_local")
                    for p in files], total_bytes=total, source_large_artifacts="original baseline caches and COCO stay on laptop")
mp = study / "ARTIFACT_MANIFEST.json"
mp.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
archive = Path("D:/coco_wire") / ("FREQ_RESULTS_" + run_id.split("_")[-1] + ".zip")
assert not archive.exists()
with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in [*files,mp]:
        z.write(p,p.relative_to(study).as_posix())
print(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
                      files=len(files)+1,run_id=run_id)))
