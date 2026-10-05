"""Desktop-only, bounded, verified laptop -> desktop -> GPU-server relay.

Only Python standard-library transport and hashes; no tensor/model imports.
--once transfers currently committed assets and exits partial when the producer
is still running. Without --once, poll until full producer completion and a
final destination hash verification. Neither mode starts any model experiment.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from server_transport import ROOT as SERVER_ROOT, ssh as _server_ssh, put as _server_put

# Redirected Windows python/pythonw streams may default to GBK even when every
# transport payload is UTF-8. A replacement character in an SSH error must not
# cause the retry logger itself to fail. This affects this process only.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

LAPTOP_ROOT = "D:/coco_wire/prototype_guided_evidence_selection_20261003"
LOCAL_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = LOCAL_ROOT.parent / "box_evidence_projection_20261003/scripts/collect_pipeline.py"
_spec = importlib.util.spec_from_file_location("existing_hidden_transport", COLLECTOR)
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
hidden = _old.hidden_process_options
MAX_STAGE = 2 * 1024**3
SERVER_RESERVE = 1536 * 1024**2
TRANSPORT_LOG = None


def transport_retry(function, *args, **kwargs):
    for attempt in range(1, 5):
        try:
            return function(*args, **kwargs)
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            entry = dict(at=now(), stage="transport_retry", attempt=attempt,
                         function=getattr(function, "__name__", "subprocess"), error=str(exc)[-2000:])
            if TRANSPORT_LOG is not None:
                with TRANSPORT_LOG.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(json.dumps(entry, ensure_ascii=True), flush=True)
            if attempt == 4:
                raise
            time.sleep(min(5 * 2**(attempt-1), 30))


def server_ssh(*args, **kwargs):
    return transport_retry(_server_ssh, *args, **kwargs)


def server_put(*args, **kwargs):
    return transport_retry(_server_put, *args, **kwargs)


def laptop_ssh(*args, **kwargs):
    return transport_retry(_old.remote, *args, **kwargs)


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def server_python(code, timeout=300):
    # Base64 is data passed to a fixed Python program, never a shell-built path.
    encoded = base64.b64encode(code.encode()).decode()
    command = "/root/miniconda3/bin/python -c \"import base64;exec(base64.b64decode('" + encoded + "'))\""
    output = server_ssh(command, timeout=timeout)
    return json.loads(output.strip().splitlines()[-1])


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def laptop_get(remote, local, timeout=1800):
    local = Path(local)
    temporary = local.with_name(local.name + ".download")
    def receive():
        result = subprocess.run(["scp", "-B", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "28358lan:" + remote, str(temporary)], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, **hidden())
        if result.returncode:
            raise RuntimeError(f"Laptop SCP exit {result.returncode}: {result.stderr[-2000:]}")
    transport_retry(receive)
    os.replace(temporary, local)


def snapshot_laptop(run_ids):
    # INDEX is an atomic producer snapshot. Metadata are only fetched for
    # complete entries; a currently written .tmp archive never enters the queue.
    code = "$ErrorActionPreference='Stop';$r=" + ps_quote(LAPTOP_ROOT) + ";"
    code += "function ReadJ($p){if(Test-Path -LiteralPath $p){Get-Content -LiteralPath $p -Raw -Encoding UTF8|ConvertFrom-Json}else{$null}};"
    code += "$a=Join-Path $r 'assets';$s=Join-Path $r 'SUPPORT.zip';"
    code += "@{index=(ReadJ (Join-Path $a 'INDEX.json'));identity=(ReadJ (Join-Path $a 'MIGRATION_IDENTITY.json'));"
    code += "complete=(ReadJ (Join-Path $a 'COMPLETE.json'));partial=(ReadJ (Join-Path $a 'PARTIAL.json'));"
    code += "progress=(ReadJ (Join-Path $a 'PROGRESS.json'));"
    code += "producer=(ReadJ (Join-Path $r " + ps_quote("runs/" + run_ids["asset_pack"] + "/run.json") + "));"
    code += "producer_failure=(ReadJ (Join-Path $r " + ps_quote("runs/" + run_ids["asset_pack"] + "/FAILURE.json") + "));"
    code += "support_exists=(Test-Path -LiteralPath $s);support_bytes=$(if(Test-Path -LiteralPath $s){(Get-Item -LiteralPath $s).Length}else{0})}|ConvertTo-Json -Depth 24 -Compress"
    return json.loads(laptop_ssh(code, timeout=180))


def laptop_metadata(ids):
    if not ids: return []
    # One fixed path expression, not one long command fragment per candidate.
    # EncodedCommand expands UTF-16 + base64 and otherwise hits Windows' limit.
    code = "$ErrorActionPreference='Stop';$r=" + ps_quote(LAPTOP_ROOT + "/assets/images") + ";$rows=@();"
    code += "$ids=@(" + ",".join(str(int(iid)) for iid in ids) + ");"
    code += "foreach($iid in $ids){$name=('{0:D12}.meta.json' -f [long]$iid);$rows+=Get-Content -LiteralPath (Join-Path $r $name) -Raw -Encoding UTF8|ConvertFrom-Json};"
    code += "ConvertTo-Json -InputObject @($rows) -Depth 24 -Compress"
    result = json.loads(laptop_ssh(code, timeout=180))
    return result if isinstance(result, list) else [result]


def remote_state(expected, verify_hash=True):
    payload = [(int(m["image_id"]), m["compressed_sha256"], int(m["compressed_bytes"])) for m in expected]
    code = """from pathlib import Path
import hashlib,json,shutil
r=Path(ROOT); p=r/'assets/images';p.mkdir(parents=True,exist_ok=True)
def digest(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
rows=[]
for iid,want,size in ITEMS:
 f=p/(f'{iid:012d}.pt.gz');good=False
 if f.exists():
  if f.stat().st_size!=size:raise RuntimeError('Existing server file size differs: '+str(f))
  if VERIFY and digest(f)!=want:raise RuntimeError('Existing server file hash differs: '+str(f))
  good=True
 rows.append({'image_id':iid,'present':good})
print(json.dumps({'files':rows,'free_bytes':shutil.disk_usage(r).free}))
"""
    code = "ROOT=" + repr(SERVER_ROOT) + "\nITEMS=" + repr(payload) + "\nVERIFY=" + repr(verify_hash) + "\n" + code
    return server_python(code, timeout=900)


def commit_server(metas):
    # Each file is re-hashed at the receiver immediately before committing.
    code = """from pathlib import Path
import hashlib,json,os
r=Path(ROOT); p=r/'assets/images';incoming=r/'.relay_incoming';p.mkdir(parents=True,exist_ok=True)
def digest(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
done=[]
for m in METAS:
 iid=int(m['image_id']);name=f'{iid:012d}.pt.gz';target=p/name
 source=target if target.exists() else incoming/name
 if not source.is_file() or source.stat().st_size!=m['compressed_bytes'] or digest(source)!=m['compressed_sha256']:
  raise RuntimeError('Receiver hash/size check failed '+name)
 if source!=target:os.replace(source,target)
 mp=p/(f'{iid:012d}.meta.json')
 if mp.exists():
  old=json.loads(mp.read_text())
  if old['compressed_sha256']!=m['compressed_sha256']:raise RuntimeError('Committed metadata differs')
 temp=mp.with_suffix(mp.suffix+'.tmp');temp.write_text(json.dumps(m,ensure_ascii=False,indent=2));os.replace(temp,mp)
 done.append({'image_id':iid,'sha256':m['compressed_sha256'],'bytes':m['compressed_bytes']})
print(json.dumps({'verified':done}))
"""
    return server_python("ROOT=" + repr(SERVER_ROOT) + "\nMETAS=" + repr(metas) + "\n" + code, timeout=900)


class Relay:
    def __init__(self, args):
        global TRANSPORT_LOG
        self.args = args
        self.ids = json.loads((LOCAL_ROOT / "RUN_IDS.json").read_text(encoding="utf-8-sig"))
        self.run_id = args.run_id or self.ids["runs"]["transfer"]
        self.out = Path(args.out or LOCAL_ROOT / "runs" / self.run_id).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.stage = self.out / "staging"; self.stage.mkdir(exist_ok=True)
        self.started = time.monotonic()
        self.log = self.out / "RELAY.log"
        TRANSPORT_LOG = self.log
        self.receipts = {}
        record = self.out / "TRANSFER_MANIFEST.json"
        if record.exists(): self.receipts = json.loads(record.read_text(encoding="utf-8"))
        self.run = {"run_id":self.run_id,"study_id":self.ids["study_id"],"status":"running",
                    "started_at":now(),"kind":"transport","command":sys.argv,
                    "source_host":"28358lan","destination":SERVER_ROOT,
                    "scope":"stdlib transport only; no model computation","script_sha256":sha(__file__)}
        prior = self.out / "run.json"
        if prior.exists():
            previous = json.loads(prior.read_text(encoding="utf-8-sig"))
            dump(self.out / ("RESUME_" + uuid.uuid4().hex + ".json"), previous)
            self.run["first_started_at"] = previous.get("first_started_at", previous.get("started_at"))
        dump(prior, self.run)

    def event(self, stage, **fields):
        record = dict(at=now(),stage=stage,elapsed_s=time.monotonic()-self.started,
                      verified_images=len(self.receipts),**fields)
        with self.log.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+"\n")
        dump(self.out / "PROGRESS.json", record)
        print(json.dumps(record,ensure_ascii=True),flush=True)

    def ensure_stage_room(self, need):
        used = sum(p.stat().st_size for p in self.stage.iterdir() if p.is_file())
        if used + need > MAX_STAGE or shutil.disk_usage(self.stage).free < need + SERVER_RESERVE:
            raise RuntimeError("Relay staging 2GiB/free-space safeguard: preserve files, do not shrink cohort")

    def delete_verified(self, path, digest):
        path = Path(path).resolve()
        if path.parent != self.stage.resolve() or not path.is_file() or sha(path) != digest:
            raise RuntimeError("Refusing removal outside this Run staging or before exact hash verification")
        path.unlink()

    def support(self, snapshot):
        if not snapshot["support_exists"]:
            self.event("waiting_support_archive"); return False
        code = "$ErrorActionPreference='Stop';$p=" + ps_quote(LAPTOP_ROOT+"/SUPPORT.zip") + ";"
        code += "@{sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower();bytes=(Get-Item -LiteralPath $p).Length}|ConvertTo-Json -Compress"
        info = json.loads(laptop_ssh(code, timeout=180))
        check = server_python("from pathlib import Path\nimport json,shutil\nr=Path("+repr(SERVER_ROOT)+");r.mkdir(parents=True,exist_ok=True)\np=r/'SUPPORT_RELAY_RECEIPT.json'\nprint(json.dumps({'receipt':json.loads(p.read_text()) if p.exists() else None,'free_bytes':shutil.disk_usage(r).free}))")
        if check["receipt"] and check["receipt"].get("archive_sha256") == info["sha256"]:
            dump(self.out/"SUPPORT_RELAY_RECEIPT.json",check["receipt"])
            self.event("support_previously_verified",files=check["receipt"]["files"]);return True
        if check["free_bytes"] < int(info["bytes"])*4 + SERVER_RESERVE:
            raise RuntimeError("Server space insufficient for support transfer/extraction plus reserve")
        local = self.stage / "SUPPORT.zip"
        if local.exists() and sha(local) != info["sha256"]:
            raise RuntimeError("Existing staged support differs; preserve failed attempt")
        if not local.exists():
            self.ensure_stage_room(int(info["bytes"]))
            self.event("support_laptop_download",bytes=info["bytes"])
            laptop_get(LAPTOP_ROOT+"/SUPPORT.zip",local)
        if local.stat().st_size != int(info["bytes"]) or sha(local) != info["sha256"]:
            raise RuntimeError("Desktop support archive hash mismatch")
        server_ssh("mkdir -p "+SERVER_ROOT+"/.relay_incoming",timeout=60)
        self.event("support_server_upload",bytes=info["bytes"])
        server_put(local,SERVER_ROOT+"/.relay_incoming/SUPPORT.zip",timeout=1800)
        code = """from pathlib import Path
import json,hashlib,zipfile,os,stat
r=Path(ROOT);archive=r/'.relay_incoming/SUPPORT.zip'
def digest(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
if digest(archive)!=WANT:raise RuntimeError('Support archive hash mismatch')
with zipfile.ZipFile(archive) as z:
 manifest=json.loads(z.read('SUPPORT_MANIFEST.json'))
 for m in z.infolist():
  p=(r/m.filename).resolve()
  if not p.is_relative_to(r.resolve()) or stat.S_ISLNK(m.external_attr>>16):raise RuntimeError('Unsafe support path')
 z.extractall(r)
for name,v in manifest.items():
 p=r/name
 if p.stat().st_size!=v['bytes'] or digest(p)!=v['sha256']:raise RuntimeError('Support member mismatch '+name)
receipt={'status':'complete','archive_sha256':WANT,'files':len(manifest),'transport':'desktop relay','verified_at':TIME}
p=r/'SUPPORT_RELAY_RECEIPT.json';temp=p.with_suffix('.json.tmp');temp.write_text(json.dumps(receipt,indent=2));os.replace(temp,p)
if archive.parent==(r/'.relay_incoming') and digest(archive)==WANT:archive.unlink()
print(json.dumps(receipt))
"""
        receipt = server_python("ROOT="+repr(SERVER_ROOT)+"\nWANT="+repr(info["sha256"])+"\nTIME="+repr(now())+"\n"+code,timeout=900)
        dump(self.out/"SUPPORT_RELAY_RECEIPT.json",receipt)
        self.delete_verified(local,info["sha256"])
        self.event("support_complete",files=receipt["files"])
        return True

    def batch(self, metas):
        check = remote_state(metas)
        present = {int(row["image_id"]):row["present"] for row in check["files"]}
        needed = sum(int(m["compressed_bytes"]) for m in metas if not present[int(m["image_id"])])
        if check["free_bytes"] < needed + SERVER_RESERVE:
            raise RuntimeError("Server data disk free space below transfer batch + 1.5GiB output reserve; no cohort reduction")
        server_ssh("mkdir -p "+SERVER_ROOT+"/.relay_incoming "+SERVER_ROOT+"/assets/images",timeout=60)
        locals_to_clean = []
        for meta in metas:
            iid = int(meta["image_id"]); name=f"{iid:012d}.pt.gz"
            if not present[iid]:
                local=self.stage/name
                if local.exists():
                    if local.stat().st_size != int(meta["compressed_bytes"]) or sha(local) != meta["compressed_sha256"]:
                        raise RuntimeError("Staged archive differs; preserve for investigation: "+str(local))
                else:
                    self.ensure_stage_room(int(meta["compressed_bytes"]))
                    laptop_get(LAPTOP_ROOT+"/assets/images/"+name,local,timeout=900)
                if local.stat().st_size != int(meta["compressed_bytes"]) or sha(local) != meta["compressed_sha256"]:
                    raise RuntimeError("Desktop archive verification failed: "+name)
                server_put(local,SERVER_ROOT+"/.relay_incoming/"+name,timeout=900)
                locals_to_clean.append((local,meta["compressed_sha256"]))
        result=commit_server(metas)
        for meta,verified in zip(metas,result["verified"]):
            if int(meta["image_id"])!=verified["image_id"] or meta["compressed_sha256"]!=verified["sha256"]:
                raise AssertionError("Receiver receipt identity mismatch")
            self.receipts[str(meta["image_id"])]=dict(meta,receiver_verified_at=now())
        dump(self.out/"TRANSFER_MANIFEST.json",self.receipts)
        for path,digest in locals_to_clean:self.delete_verified(path,digest)
        self.event("batch_verified",images=len(metas),new_bytes=needed,server_free_before=check["free_bytes"])

    def mirror(self,snapshot):
        server_ssh("mkdir -p "+SERVER_ROOT+"/assets",timeout=60)
        for key,name in (("index","INDEX.json"),("identity","MIGRATION_IDENTITY.json"),("progress","SOURCE_PROGRESS.json")):
            if snapshot.get(key) is not None:
                dump(self.out/name,snapshot[key])
                server_put(self.out/name,SERVER_ROOT+"/assets/"+name+".tmp",timeout=120)
                server_ssh("mv "+SERVER_ROOT+"/assets/"+name+".tmp "+SERVER_ROOT+"/assets/"+name,timeout=60)
        dump(self.out/"PRODUCER_SNAPSHOT.json",{k:v for k,v in snapshot.items() if k not in ("index","identity")})

    def historical_joint(self):
        source="D:/coco_wire/box_evidence_joint_training_20261003/runs/RUN_575d0d7a163343908c0fd8beccedc0f8/PER_CANDIDATE.jsonl"
        code="$ErrorActionPreference='Stop';$p="+ps_quote(source)+";@{sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower();bytes=(Get-Item -LiteralPath $p).Length}|ConvertTo-Json -Compress"
        info=json.loads(laptop_ssh(code,timeout=180))
        name="HISTORICAL_JOINT_PER_CANDIDATE.jsonl"
        check=server_python("from pathlib import Path\nimport hashlib,json\np=Path("+repr(SERVER_ROOT+"/data/"+name)+")\nprint(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}))")
        if check["sha256"]==info["sha256"]:
            dump(self.out/"HISTORICAL_JOINT_TRANSFER.json",dict(status="complete",reused=True,**info));return
        if check["sha256"] is not None:raise RuntimeError("Existing historical joint target differs")
        local=self.stage/name
        if not local.exists():
            self.ensure_stage_room(info["bytes"]);laptop_get(source,local)
        if sha(local)!=info["sha256"]:raise RuntimeError("Historical joint local hash differs")
        server_ssh("mkdir -p "+SERVER_ROOT+"/data",timeout=60)
        server_put(local,SERVER_ROOT+"/data/"+name+".tmp",timeout=600)
        result=server_python("from pathlib import Path\nimport json,hashlib,os\np=Path("+repr(SERVER_ROOT+"/data/"+name)+")\nt=p.with_name(p.name+'.tmp')\nh=hashlib.sha256(t.read_bytes()).hexdigest()\nassert h=="+repr(info["sha256"])+"\nos.replace(t,p)\nprint(json.dumps({'status':'complete','sha256':h,'bytes':p.stat().st_size}))")
        dump(self.out/"HISTORICAL_JOINT_TRANSFER.json",result);self.delete_verified(local,info["sha256"])
        self.event("historical_joint_complete",bytes=info["bytes"])

    def finish(self,snapshot):
        all_rows=[row for rows in snapshot["index"].values() for row in rows]
        if not snapshot.get("complete",{}).get("complete") or len(all_rows)!=13000:
            return False
        if any(row.get("status")!="complete" or str(row["image_id"]) not in self.receipts for row in all_rows):
            return False
        self.event("final_receiver_hash_audit",images=len(all_rows))
        for start in range(0,len(all_rows),64):
            metas=[self.receipts[str(row["image_id"])] for row in all_rows[start:start+64]]
            state=remote_state(metas)
            if not all(row["present"] for row in state["files"]):raise RuntimeError("Final receiver inventory incomplete")
        complete=snapshot["complete"]
        if complete["fingerprint"]!=snapshot["identity"]["fingerprint"]:
            raise AssertionError("Producer completion identity differs")
        self.mirror(snapshot)
        dump(self.out/"SOURCE_COMPLETE.json",complete)
        server_put(self.out/"SOURCE_COMPLETE.json",SERVER_ROOT+"/assets/COMPLETE.json.tmp",timeout=120)
        server_ssh("mv "+SERVER_ROOT+"/assets/COMPLETE.json.tmp "+SERVER_ROOT+"/assets/COMPLETE.json",timeout=60)
        receipt=dict(status="complete",verified_images=len(all_rows),verified_bytes=sum(m["compressed_bytes"] for m in self.receipts.values()),
                     completed_at=now(),fingerprint=complete["fingerprint"],
                     no_model_computation=True,destination_F_audit="still required before training")
        dump(self.out/"transfer.json",receipt)
        dump(self.out/"COMPLETE.json",receipt)
        server_put(self.out/"transfer.json",SERVER_ROOT+"/assets/transfer.json",timeout=120)
        # The training/pipeline entrypoint waits on this root-level receipt.
        # It is created only after producer completion and the full receiver
        # hash audit above, never after a partial or one-shot transfer.
        server_put(self.out/"transfer.json",SERVER_ROOT+"/ASSETS_TRANSFER_COMPLETE.json.tmp",timeout=120)
        server_ssh("mv "+SERVER_ROOT+"/ASSETS_TRANSFER_COMPLETE.json.tmp "+SERVER_ROOT+"/ASSETS_TRANSFER_COMPLETE.json",timeout=60)
        self.run.update(status="completed",completed_at=now(),exit_code=0)
        dump(self.out/"run.json",self.run)
        self.event("completed",verified_bytes=receipt["verified_bytes"])
        return True

    def execute(self):
        support_ready=False
        historical_ready=False
        resumed_receipts_verified=False
        deadline=time.monotonic()+self.args.max_hours*3600
        while time.monotonic()<deadline:
            snapshot=snapshot_laptop(self.ids["runs"])
            if not support_ready:support_ready=self.support(snapshot)
            if support_ready and not historical_ready:
                self.historical_joint();historical_ready=True
            if support_ready and not resumed_receipts_verified:
                previous=list(self.receipts.values())
                for start in range(0,len(previous),64):
                    state=remote_state(previous[start:start+64])
                    for row in state["files"]:
                        if not row["present"]:self.receipts.pop(str(row["image_id"]),None)
                dump(self.out/"TRANSFER_MANIFEST.json",self.receipts)
                resumed_receipts_verified=True
            if support_ready and snapshot.get("index"):
                self.mirror(snapshot)
                expected=[]
                for split in ("fit","dev","val"):
                    source_rows=snapshot["index"][split]
                    if self.args.limit_per_split is not None:source_rows=source_rows[:self.args.limit_per_split]
                    expected += [row for row in source_rows if row.get("status")=="complete"]
                # First the migration-smoke images, so the destination can audit
                # full F witnesses without waiting for the growing fit queue.
                expected.sort(key=lambda r:(not bool(r.get("witness")),int(r["image_id"])))
                pending=[r for r in expected if str(r["image_id"]) not in self.receipts
                    or self.receipts[str(r["image_id"])]["compressed_sha256"]!=r["compressed_sha256"]]
                for start in range(0,len(pending),self.args.batch_size):
                    rows=pending[start:start+self.args.batch_size]
                    metas=laptop_metadata([r["image_id"] for r in rows])
                    lookup={int(m["image_id"]):m for m in metas}
                    metas=[lookup[int(row["image_id"])] for row in rows]
                    for row,meta in zip(rows,metas):
                        if row["compressed_sha256"]!=meta["compressed_sha256"] or meta["migration_fingerprint"]!=snapshot["identity"]["fingerprint"]:
                            raise AssertionError("Producer INDEX/meta identity differs")
                    self.batch(metas)
                if self.args.limit_per_split is None and self.finish(snapshot):return
            if snapshot.get("producer_failure"):
                dump(self.out/"PRODUCER_FAILURE.json",snapshot["producer_failure"])
                raise RuntimeError("Producer reported failure; retained transferred assets, do not shrink cohort")
            if self.args.once:
                receipt=dict(status="partial",verified_images=len(self.receipts),at=now(),reason="one-shot transport; producer not yet fully complete")
                dump(self.out/"transfer.json",receipt);dump(self.out/"PARTIAL.json",receipt)
                self.run.update(status="completed",scientific_status="partial_transport",exit_code=0,completed_at=now())
                dump(self.out/"run.json",self.run);self.event("partial_one_shot");return
            self.event("waiting_producer",source_progress=snapshot.get("progress"))
            time.sleep(min(self.args.poll_seconds,60))
        raise TimeoutError("Relay resource limit; source packing and remote experiments were not stopped")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out");p.add_argument("--run-id");p.add_argument("--once",action="store_true")
    p.add_argument("--limit-per-split",type=int)
    p.add_argument("--batch-size",type=int,default=16);p.add_argument("--poll-seconds",type=int,default=45)
    p.add_argument("--max-hours",type=float,default=24)
    args=p.parse_args()
    if not 1<=args.batch_size<=64 or not 1<=args.poll_seconds<=60 or args.max_hours<=0:
        p.error("batch 1..64; poll 1..60 seconds; max-hours positive")
    if args.limit_per_split is not None and (args.limit_per_split<1 or not args.once):
        p.error("--limit-per-split is positive and requires --once")
    relay=Relay(args)
    try:relay.execute()
    except Exception as exc:
        error=dict(at=now(),error=repr(exc),traceback=traceback.format_exc())
        dump(relay.out/("FAILURE_"+uuid.uuid4().hex+".json"),error)
        relay.run.update(status="failed",exit_code=1,completed_at=now(),error=str(exc))
        dump(relay.out/"run.json",relay.run)
        dump(relay.out/"transfer.json",dict(status="partial_failed",verified_images=len(relay.receipts),**error))
        failure_receipt=relay.out/"ASSETS_TRANSFER_FAILURE.json"
        dump(failure_receipt,dict(status="failed",run_id=relay.run_id,verified_images=len(relay.receipts),**error))
        try:
            # Preserve a useful completed receipt if a later transport attempt
            # encounters an unrelated error. Never replace or remove it here.
            server_put(failure_receipt,SERVER_ROOT+"/ASSETS_TRANSFER_FAILURE.json.tmp",timeout=120)
            code="from pathlib import Path\nimport json,os\nr=Path("+repr(SERVER_ROOT)+")\nt=r/'ASSETS_TRANSFER_FAILURE.json.tmp'\np=r/'ASSETS_TRANSFER_COMPLETE.json'\ncomplete=p.exists() and json.loads(p.read_text()).get('status')=='complete'\nif not complete:os.replace(t,r/'ASSETS_TRANSFER_FAILURE.json')\nprint(json.dumps({'failure_published':not complete,'completed_receipt_preserved':complete}))"
            server_python(code,timeout=120)
        except Exception as transport_exc:
            dump(relay.out/"FAILURE_RECEIPT_UPLOAD_ERROR.json",dict(at=now(),error=str(transport_exc)))
        relay.event("failed",error=str(exc));raise


if __name__=="__main__":main()
