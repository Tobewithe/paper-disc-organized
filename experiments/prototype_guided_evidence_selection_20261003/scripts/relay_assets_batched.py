"""Batched transport variant; does not modify the currently running relay.

Each batch bundles already committed gzip archives with ZIP_STORED on the
laptop. One SCP retrieves the bundle and one server_transport SFTP uploads it.
Archive bytes, original metadata and receiver per-image SHA checks are unchanged.
Uses the original relay's support reuse, completion gates and hidden transports.
No model/tensor imports and no experimental computation. Run only after the
older transfer worker is stopped; this script never interrupts another process.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import uuid

import relay_assets as base


LAPTOP_PYTHON = "C:/Users/28358/anaconda3/envs/pytorch/python.exe"
COMMAND_OUT = None
COMMAND_RUN_ID = None


def laptop_python(source, timeout=900):
    # Transfer exact source as a file; embedding source+metadata in a nested
    # base64/EncodedCommand exceeds the Windows command-line length limit.
    if COMMAND_OUT is None or COMMAND_RUN_ID is None:
        raise RuntimeError("Laptop command provenance context is not initialized")
    import hashlib
    digest=hashlib.sha256(source.encode("utf-8")).hexdigest()
    local_dir=COMMAND_OUT/"transport_commands";local_dir.mkdir(parents=True,exist_ok=True)
    local=local_dir/(digest+".py")
    if local.exists():
        if base.sha(local)!=digest:raise RuntimeError("Saved transport helper differs")
    else:
        temporary=local.with_suffix(".py.tmp")
        with temporary.open("w",encoding="utf-8",newline="\n") as stream:
            stream.write(source);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,local)
    remote_dir=base.LAPTOP_ROOT+"/.relay_commands/"+COMMAND_RUN_ID
    remote=remote_dir+"/"+local.name
    check="$ErrorActionPreference='Stop';$d="+base.ps_quote(remote_dir)+";New-Item -ItemType Directory -Path $d -Force|Out-Null;$p="+base.ps_quote(remote)+";"
    check+="@{sha256=$(if(Test-Path -LiteralPath $p){(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower()}else{$null})}|ConvertTo-Json -Compress"
    existing=json.loads(base.laptop_ssh(check,timeout=180))
    if existing["sha256"] not in (None,digest):raise RuntimeError("Laptop helper path has different content")
    if existing["sha256"] is None:
        def upload_helper():
            result=subprocess.run(["scp","-B","-o","BatchMode=yes","-o","ConnectTimeout=15",str(local),"28358lan:"+remote],
                capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=180,**base.hidden())
            if result.returncode:raise RuntimeError("Laptop helper SCP failed: "+result.stderr[-2000:])
        base.transport_retry(upload_helper)
    code="$ErrorActionPreference='Stop';$p="+base.ps_quote(remote)+";"
    code+="if((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower() -ne "+base.ps_quote(digest)+"){throw 'Transport helper hash mismatch'};"
    code+="& "+base.ps_quote(LAPTOP_PYTHON)+" -X utf8 "+base.ps_quote(remote)
    code+=";if($LASTEXITCODE -ne 0){throw 'Laptop bundle worker failed'}"
    output = base.laptop_ssh(code, timeout=timeout)
    return json.loads(output.strip().splitlines()[-1])


def make_bundle(run_id, metas):
    request = [(int(m["image_id"]), m["compressed_sha256"], int(m["compressed_bytes"])) for m in metas]
    import hashlib
    key = hashlib.sha256(json.dumps(request, separators=(",", ":")).encode()).hexdigest()
    code = """from pathlib import Path
import hashlib,json,os,shutil,zipfile
r=Path(ROOT);directory=r/'.relay_bundles'/RUN;directory.mkdir(parents=True,exist_ok=True)
bundle=directory/(KEY+'.zip');receipt=directory/(KEY+'.json')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
if bundle.exists() and receipt.exists():
 old=json.loads(receipt.read_text())
 if old['items']!=ITEMS or bundle.stat().st_size!=old['bytes'] or digest(bundle)!=old['sha256']:
  raise RuntimeError('Existing bundle differs; preserve failed attempt')
 print(json.dumps(old))
else:
 estimate=sum(row[2] for row in ITEMS)+len(ITEMS)*65536+1024**2
 if estimate>2*1024**3 or shutil.disk_usage(directory).free<estimate+1024**3:
  raise RuntimeError('Laptop bundle staging/free-space guard reached')
 source=r/'assets/images';temporary=bundle.with_suffix('.zip.tmp')
 with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
  for iid,want,size in ITEMS:
   data=source/(f'{iid:012d}.pt.gz');meta=source/(f'{iid:012d}.meta.json')
   m=json.loads(meta.read_text(encoding='utf-8-sig'))
   if int(m['image_id'])!=iid or m['compressed_sha256']!=want or m['compressed_bytes']!=size:
    raise RuntimeError('Committed laptop metadata differs')
   if data.stat().st_size!=size or digest(data)!=want:raise RuntimeError('Laptop archive changed')
   archive.write(data,data.name);archive.write(meta,meta.name)
 os.replace(temporary,bundle)
 result={'path':str(bundle).replace('\\\\','/'),'sha256':digest(bundle),'bytes':bundle.stat().st_size,'items':ITEMS}
 temporary=receipt.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2));os.replace(temporary,receipt)
 print(json.dumps(result))
"""
    # Lists survive JSON round trips identically on laptop bundle resume.
    header = "ROOT=" + repr(base.LAPTOP_ROOT) + "\nRUN=" + repr(run_id) + "\nKEY=" + repr(key)
    header += "\nITEMS=" + repr([list(row) for row in request]) + "\n"
    return laptop_python(header + code)


def receive_bundle(bundle, metas):
    code = """from pathlib import Path
import hashlib,json,os,zipfile,stat
r=Path(ROOT);incoming=r/'.relay_incoming';archive=incoming/NAME
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
if archive.stat().st_size!=SIZE or digest(archive)!=WANT:raise RuntimeError('Server batch bundle hash mismatch')
expected={}
for m in METAS:
 iid=int(m['image_id']);expected[f'{iid:012d}.pt.gz']=m
 expected[f'{iid:012d}.meta.json']=m
with zipfile.ZipFile(archive) as z:
 members=z.infolist()
 if len(members)!=len(expected) or set(m.filename for m in members)!=set(expected):
  raise RuntimeError('Bundle has missing/extra/duplicate members')
 for item in members:
  if item.compress_type!=zipfile.ZIP_STORED or stat.S_ISLNK(item.external_attr>>16):raise RuntimeError('Unexpected bundle encoding')
  target=(incoming/item.filename).resolve()
  if target.parent!=incoming.resolve():raise RuntimeError('Unsafe bundle path')
  m=expected[item.filename]
  if item.filename.endswith('.pt.gz') and item.file_size!=m['compressed_bytes']:raise RuntimeError('Bundle member size differs')
  if item.filename.endswith('.meta.json'):
   mm=json.loads(z.read(item))
   if mm!=m:raise RuntimeError('Bundled source metadata differs from requested metadata')
  z.extract(item,incoming)
print(json.dumps({'bundle_verified':True,'members':len(expected),'sha256':WANT}))
"""
    header = "ROOT=" + repr(base.SERVER_ROOT) + "\nNAME=" + repr(Path(bundle["path"]).name)
    header += "\nSIZE=" + repr(bundle["bytes"]) + "\nWANT=" + repr(bundle["sha256"])
    header += "\nMETAS=" + repr(metas) + "\n"
    return base.server_python(header + code, timeout=900)


def remove_remote_bundle(bundle, run_id):
    # Delete only exact, authenticated files created by this transfer Run.
    laptop_code = """from pathlib import Path
import hashlib,json
r=Path(ROOT)/'.relay_bundles'/RUN;p=Path(PATH)
if p.resolve().parent!=r.resolve() or p.suffix!='.zip':raise RuntimeError('Unsafe laptop bundle removal')
if p.exists():
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 if h.hexdigest()!=WANT:raise RuntimeError('Laptop bundle changed before removal')
 p.unlink()
print(json.dumps({'removed':str(p)}))
"""
    laptop_python("ROOT="+repr(base.LAPTOP_ROOT)+"\nRUN="+repr(run_id)+"\nPATH="+repr(bundle["path"])+"\nWANT="+repr(bundle["sha256"])+"\n"+laptop_code)
    server_code = """from pathlib import Path
import hashlib,json
r=Path(ROOT)/'.relay_incoming';p=r/NAME
if p.resolve().parent!=r.resolve():raise RuntimeError('Unsafe server bundle removal')
if p.exists():
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 if h.hexdigest()!=WANT:raise RuntimeError('Server bundle changed before removal')
 p.unlink()
print(json.dumps({'removed':str(p)}))
"""
    base.server_python("ROOT="+repr(base.SERVER_ROOT)+"\nNAME="+repr(Path(bundle["path"]).name)+"\nWANT="+repr(bundle["sha256"])+"\n"+server_code)


class BatchedRelay(base.Relay):
    def __init__(self, args):
        global COMMAND_OUT, COMMAND_RUN_ID
        super().__init__(args)
        COMMAND_OUT=self.out
        COMMAND_RUN_ID=self.run_id
        self.run.update(script_sha256=base.sha(__file__), parent_relay_sha256=base.sha(base.__file__),
                        transport_mode="ZIP_STORED batch relay; original gzip payloads unchanged")
        imported = []
        for value in args.resume_from:
            source = Path(value)
            if source.is_dir(): source = source / "TRANSFER_MANIFEST.json"
            records = json.loads(source.read_text(encoding="utf-8-sig"))
            for key, meta in records.items():
                if key in self.receipts and self.receipts[key]["compressed_sha256"] != meta["compressed_sha256"]:
                    raise RuntimeError("Imported transfer receipts disagree on image " + key)
                self.receipts[key] = meta
            imported.append(dict(path=str(source.resolve()), sha256=base.sha(source), images=len(records)))
        self.run["imported_transfer_receipts"] = imported
        base.dump(self.out / "run.json", self.run)
        base.dump(self.out / "TRANSFER_MANIFEST.json", self.receipts)

    def batch(self, metas):
        check = base.remote_state(metas)
        present = {int(row["image_id"]): row["present"] for row in check["files"]}
        missing = [meta for meta in metas if not present[int(meta["image_id"])]]
        needed = sum(int(m["compressed_bytes"]) for m in missing)
        # Incoming ZIP_STORED and extracted gzip files coexist until commit.
        overhead = len(missing) * 65536 + 1024**2
        if check["free_bytes"] < 2*needed + overhead + base.SERVER_RESERVE:
            raise RuntimeError("Server lacks batch ZIP + extracted assets + reserved outputs; do not reduce cohort")
        bundle = local = None
        if missing:
            self.ensure_stage_room(needed + overhead)
            self.event("batch_bundle_preparing", images=len(missing), payload_bytes=needed)
            bundle = make_bundle(self.run_id, missing)
            local = self.stage / Path(bundle["path"]).name
            if local.exists():
                if local.stat().st_size != bundle["bytes"] or base.sha(local) != bundle["sha256"]:
                    raise RuntimeError("Existing desktop bundle differs; preserve failed data")
            else:
                self.ensure_stage_room(bundle["bytes"])
                self.event("batch_laptop_download", images=len(missing), bytes=bundle["bytes"])
                base.laptop_get(bundle["path"], local, timeout=1800)
            if local.stat().st_size != bundle["bytes"] or base.sha(local) != bundle["sha256"]:
                raise RuntimeError("Desktop batch bundle hash mismatch")
            base.server_ssh("mkdir -p "+base.SERVER_ROOT+"/.relay_incoming "+base.SERVER_ROOT+"/assets/images",timeout=60)
            self.event("batch_server_upload", images=len(missing), bytes=bundle["bytes"])
            base.server_put(local, base.SERVER_ROOT+"/.relay_incoming/"+local.name, timeout=1800)
            receive_bundle(bundle, missing)
        result = base.commit_server(metas)
        if len(result["verified"]) != len(metas): raise AssertionError("Incomplete receiver receipt")
        for meta, verified in zip(metas, result["verified"]):
            if int(meta["image_id"]) != verified["image_id"] or meta["compressed_sha256"] != verified["sha256"]:
                raise AssertionError("Receiver identity mismatch")
            self.receipts[str(meta["image_id"])] = dict(meta, receiver_verified_at=base.now())
        base.dump(self.out/"TRANSFER_MANIFEST.json", self.receipts)
        if bundle is not None:
            self.delete_verified(local, bundle["sha256"])
            remove_remote_bundle(bundle, self.run_id)
        self.event("batch_verified", images=len(metas), new_bytes=needed,
                   reused_images=len(metas)-len(missing), server_free_before=check["free_bytes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New independent transfer Run ID; do not reuse active relay Run")
    parser.add_argument("--out")
    parser.add_argument("--resume-from", action="append", default=[], help="Prior transfer Run or manifest; repeatable")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--poll-seconds", type=int, default=45)
    parser.add_argument("--max-hours", type=float, default=24)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64 or not 1 <= args.poll_seconds <= 60 or args.max_hours <= 0:
        parser.error("batch 1..64, polling 1..60, max-hours positive")
    if args.limit_per_split is not None and (args.limit_per_split < 1 or not args.once):
        parser.error("--limit-per-split requires positive value and --once")
    if not args.run_id.startswith("RUN_") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_id):
        parser.error("run-id must be a safe RUN_ identifier")
    relay = BatchedRelay(args)
    try:
        relay.execute()
    except Exception as exc:
        error = dict(at=base.now(), error=repr(exc), traceback=traceback.format_exc())
        base.dump(relay.out/("FAILURE_"+uuid.uuid4().hex+".json"), error)
        relay.run.update(status="failed",exit_code=1,completed_at=base.now(),error=str(exc))
        base.dump(relay.out/"run.json",relay.run)
        receipt = dict(status="failed",run_id=relay.run_id,verified_images=len(relay.receipts),**error)
        base.dump(relay.out/"transfer.json",receipt)
        target = relay.out/"ASSETS_TRANSFER_FAILURE.json";base.dump(target,receipt)
        try:
            base.server_put(target,base.SERVER_ROOT+"/ASSETS_TRANSFER_FAILURE.json.tmp",timeout=120)
            code="from pathlib import Path\nimport json,os\nr=Path("+repr(base.SERVER_ROOT)+")\np=r/'ASSETS_TRANSFER_COMPLETE.json'\ncomplete=p.exists() and json.loads(p.read_text()).get('status')=='complete'\nif not complete:os.replace(r/'ASSETS_TRANSFER_FAILURE.json.tmp',r/'ASSETS_TRANSFER_FAILURE.json')\nprint(json.dumps({'failure_published':not complete}))"
            base.server_python(code,timeout=120)
        except Exception as upload_error:
            base.dump(relay.out/"FAILURE_RECEIPT_UPLOAD_ERROR.json",dict(at=base.now(),error=str(upload_error)))
        relay.event("failed",error=str(exc))
        raise


if __name__ == "__main__":
    main()
