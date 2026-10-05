"""One-shot, model-free verified Run collection over the existing hidden SSH.

Default: reports/logs/registered source snapshots only. Checkpoints stay remote
and are listed with size/SHA256. Existing differing local versions are retained
under each Run's _collection_history, never discarded by extraction.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import uuid
import zipfile

STAGES = ("smoke", "evaluation")
REMOTE_ROOT = "/root/native_semantic_projection_20261004_retry2"
REMOTE_PYTHON = "/root/miniconda3/bin/python"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


# This script is transmitted as source to the existing transport's SSH call.
# It uses only Python's standard library and creates a unique transport archive.
PACKER = r'''
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,zipfile

r=Path(request['root']).resolve()
if str(r)!='/root/native_semantic_projection_20261004_retry2':
 raise RuntimeError('Unexpected screen root')
identity=json.loads((r/'RUN_IDS.json').read_text())
if identity['study_id']!=request['study_id']:
 raise RuntimeError('Local/remote Study IDs disagree')
selected={stage:identity['runs'][stage] for stage in request['stages']}
if selected!=request['runs']:
 raise RuntimeError('Local/remote Run associations disagree')
directory=r/'_collections';directory.mkdir(exist_ok=True)
archive=directory/(request['collection_id']+'.zip')
temporary=archive.with_name(archive.name+'.tmp')
if archive.exists() or temporary.exists():raise RuntimeError('Collection ID already used')
allowed={'.json','.jsonl','.md','.log','.txt','.sha256'}
snapshot_extensions=allowed|{'.py','.yaml','.yml','.toml','.ini','.cfg'}
checkpoint_extensions={'.pt','.pth','.ckpt','.safetensors'}
manifest={}
records={}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def remote_file_record(path):
 h=hashlib.sha256();size=0
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):
   h.update(block);size+=len(block)
 return {'bytes':size,'sha256':h.hexdigest()}
with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
 for stage,rid in selected.items():
  if not rid.startswith('RUN_') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in rid):
   raise RuntimeError('Unsafe Run ID')
  base=r/'runs'/rid
  run_path=base/'run.json'
  if not run_path.is_file():
   records[stage]={'run_id':rid,'available':False,'execution_status':'not_started_or_no_run_record'}
   continue
  if base.resolve().parent!=(r/'runs').resolve():raise RuntimeError('Run path escapes Study')
  run_data=run_path.read_bytes();run=json.loads(run_data)
  if run.get('run_id')!=rid:raise RuntimeError('Run record identity differs')
  if run.get('study_id',identity['study_id'])!=identity['study_id']:
   raise RuntimeError('Run belongs to another Study')
  snapshots={}
  for s in run.get('snapshots',[]):
   name=s.get('snapshot')
   source=s.get('path_relative_to_cwd') or s.get('path','')
   if name and Path(source).suffix.lower() in snapshot_extensions:
    snapshots[name]=source
  files=[];remote_only=[]
  for p in sorted(base.rglob('*')):
   if not p.is_file():continue
   if not p.resolve().is_relative_to(base.resolve()):raise RuntimeError('Run artifact leaves its directory')
   rel=p.relative_to(base).as_posix()
   if rel.split('/')[0] in {'_collection_history','_collections','_transfer'}:continue
   if rel in {'transfer.json','manifest.sha256'}:continue
   if p.name.endswith(('.tmp','.part','.partial')) or '.incoming.' in p.name:continue
   selected_file=(p.suffix.lower() in allowed or rel in snapshots
                  or ('snapshots' in p.relative_to(base).parts and p.suffix.lower() in snapshot_extensions))
   is_checkpoint=p.suffix.lower() in checkpoint_extensions
   if request['include_checkpoints'] and is_checkpoint:selected_file=True
   if not selected_file:
    remote_only.append({'path':rel,**remote_file_record(p),
                        'kind':'checkpoint' if is_checkpoint else 'other_excluded_artifact'})
    continue
   # Read once: a running log can grow, but its packaged bytes and hash still
   # describe exactly the same point-in-time snapshot. run.json uses the
   # original bytes whose execution status was read above.
   data=run_data if rel=='run.json' else p.read_bytes()
   member='runs/'+rid+'/'+rel
   z.writestr(member,data)
   item={'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
   manifest[member]=item;files.append(item)
  records[stage]={'run_id':rid,'available':True,'execution_status':run.get('status','unknown'),
                  'exit_code':run.get('exit_code'),'files':files,'remote_only':remote_only,
                  'snapshot_while_running':run.get('status') not in ('completed','failed','interrupted','cancelled')}
 root_state={}
 for name in ['PIPELINE_STATUS.json','LAUNCH.json','LAUNCH_RECEIPT.json','LAUNCH']:
  p=r/name
  if p.is_file():
   data=p.read_bytes()
   try:value=json.loads(data)
   except (ValueError,UnicodeDecodeError):value=data.decode('utf-8',errors='replace')
   root_state[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'content':value}
 metadata={'schema':'fast-screen-run-transfer-v1','collection_id':request['collection_id'],
           'study_id':identity['study_id'],'root':str(r),'collected_at':datetime.now(timezone.utc).isoformat(),
           'include_checkpoints':request['include_checkpoints'],'runs':records,'files':manifest,
           'root_state':root_state,'scope':'transport snapshot; not scientific-goal completion'}
 z.writestr('COLLECTION_MANIFEST.json',json.dumps(metadata,ensure_ascii=False,indent=2))
os.replace(temporary,archive)
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive),
                  'collection_id':request['collection_id'],'runs':{k:{x:v for x,v in row.items() if x not in ('files','remote_only')} for k,row in records.items()}}))
'''


def safe_relative(value):
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ValueError("Unsafe archive-relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(part in ("..", ".") or ":" in part for part in p.parts):
        raise ValueError("Unsafe archive-relative path")
    return p


def keep_previous(path, base, collection_id):
    """Preserve current bytes before publishing a newer verified snapshot."""
    if not path.exists():
        return None
    relative = path.relative_to(base)
    backup = base / "_collection_history" / collection_id / relative
    if backup.exists():
        raise RuntimeError(f"History destination already exists: {backup}")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    return backup.relative_to(base).as_posix()


def install_file(source, target, base, collection_id, expected_hash):
    if not target.resolve().is_relative_to(base.resolve()):
        raise RuntimeError("Local destination escapes its Run")
    if target.exists():
        if digest(target) == expected_hash:
            return "identical", None
        if "snapshots" in target.relative_to(base).parts:
            raise RuntimeError(f"Immutable local snapshot differs; preserving both packages: {target}")
        backup = keep_previous(target, base, collection_id)
    else:
        backup = None
    target.parent.mkdir(parents=True, exist_ok=True)
    # Appending a collection UUID to a 64-character snapshot filename can
    # exceed Windows MAX_PATH. Use a short exclusive name in the same folder
    # so replacement stays atomic without making the full target path longer.
    fd, temporary_name = tempfile.mkstemp(prefix=".inc-", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    shutil.copyfile(source, temporary)
    if digest(temporary) != expected_hash:
        raise RuntimeError("Local verified copy changed")
    temporary.replace(target)
    return "updated" if backup else "new", backup


def collect(args):
    root = Path(args.root or Path(__file__).resolve().parents[1]).resolve()
    identity = json.loads((root / "RUN_IDS.json").read_text(encoding="utf-8-sig"))
    cfg = json.loads((root / "RUN_CONFIG.json").read_text(encoding="utf-8-sig"))
    if cfg["server_root"] != REMOTE_ROOT:
        raise RuntimeError("Unexpected remote screen root")
    # Reuse the existing transport unchanged, including CREATE_NO_WINDOW and
    # its Paramiko fallback; never open an interactive OpenSSH window.
    transport_dir = root.parent / "prototype_guided_evidence_selection_20261003" / "scripts"
    sys.path.insert(0, str(transport_dir))
    from server_transport import ssh, get, HOST, PORT
    collection_id = "COLLECT_" + uuid.uuid4().hex
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    requested = {stage: identity["runs"][stage] for stage in stages}
    request = dict(root=REMOTE_ROOT, study_id=identity["study_id"], runs=requested, stages=stages,
                   collection_id=collection_id, include_checkpoints=args.include_checkpoints)
    request_b64 = base64.b64encode(json.dumps(request).encode()).decode()
    program = "import base64,json\nrequest=json.loads(base64.b64decode('" + request_b64 + "'))\n" + PACKER
    command = REMOTE_PYTHON + " - <<'PY'\n" + program + "\nPY"
    received = json.loads(ssh(command, timeout=args.timeout).strip().splitlines()[-1])
    if received["collection_id"] != collection_id or not received["archive"].startswith(REMOTE_ROOT + "/_collections/"):
        raise RuntimeError("Unexpected remote archive receipt")
    transfer_dir = root / "_transfer" / collection_id
    transfer_dir.mkdir(parents=True, exist_ok=False)
    dump(transfer_dir / "REMOTE_ARCHIVE.json", received)
    archive = transfer_dir / "runs.zip"
    get(received["archive"], archive, timeout=args.timeout)
    if archive.stat().st_size != received["bytes"] or digest(archive) != received["sha256"]:
        raise RuntimeError("Returned ZIP size/SHA256 mismatch")
    # Run/snapshots/SHA256 nested below the already long project path exceeded
    # Windows MAX_PATH. A short system-temp root preserves the exact manifest
    # paths while keeping every extraction path comfortably shorter. Retain
    # this directory for provenance/recovery, including after collection errors.
    staged = Path(tempfile.mkdtemp(prefix="pfs-collect-"))
    dump(transfer_dir / "STAGING.json", dict(path=str(staged), retained=True,
        reason="short system-temp path avoids Windows MAX_PATH during verification"))
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len({m.filename for m in members}) != len(members):
            raise RuntimeError("Duplicate archive members")
        metadata = json.loads(z.read("COLLECTION_MANIFEST.json"))
        if (metadata["collection_id"] != collection_id or metadata["study_id"] != identity["study_id"]
                or metadata["root"] != REMOTE_ROOT):
            raise RuntimeError("Collection manifest identity mismatch")
        expected = set(metadata["files"]) | {"COLLECTION_MANIFEST.json"}
        if {m.filename for m in members} != expected:
            raise RuntimeError("Archive contains unexpected or missing files")
        for item in members:
            rel = safe_relative(item.filename)
            if item.compress_type != zipfile.ZIP_STORED or stat.S_ISLNK(item.external_attr >> 16):
                raise RuntimeError("Unsupported archive member type")
            if item.filename == "COLLECTION_MANIFEST.json":
                continue
            if len(rel.parts) < 3 or rel.parts[0] != "runs" or rel.parts[1] not in requested.values():
                raise RuntimeError("Archive member belongs to another Run")
            record = metadata["files"][item.filename]
            if item.file_size != record["bytes"]:
                raise RuntimeError("Archive member size differs")
            target = staged.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024**2)
            if digest(target) != record["sha256"]:
                raise RuntimeError(f"Returned file SHA256 mismatch: {item.filename}")
    dump(transfer_dir / "COLLECTION_MANIFEST.json", metadata)
    statuses = {}
    for stage, row in metadata["runs"].items():
        rid = row["run_id"]
        if requested.get(stage) != rid:
            raise RuntimeError("Unexpected stage/Run mapping in package")
        if not row["available"]:
            statuses[stage] = dict(run_id=rid, available=False, execution_status=row["execution_status"])
            continue
        base = root / "runs" / rid
        base.mkdir(parents=True, exist_ok=True)
        changes = []
        # Preflight immutable local snapshots before updating any Run artifact.
        for record in row["files"]:
            rel = safe_relative(record["path"])
            target = base.joinpath(*rel.parts)
            if "snapshots" in rel.parts and target.exists() and digest(target) != record["sha256"]:
                raise RuntimeError(f"Local snapshot conflict retained at {target}")
        for record in row["files"]:
            rel = safe_relative(record["path"])
            status, backup = install_file(staged / "runs" / rid / Path(*rel.parts),
                base.joinpath(*rel.parts), base, collection_id, record["sha256"])
            changes.append(dict(path=record["path"], action=status, preserved_previous=backup))
        collected_run = json.loads((base / "run.json").read_text(encoding="utf-8-sig"))
        if collected_run.get("status", "unknown") != row["execution_status"]:
            raise RuntimeError("Collected execution status differs from manifest")
        for filename in ("manifest.sha256", "transfer.json"):
            keep_previous(base / filename, base, collection_id)
        manifest_text = "".join(f"{r['sha256']}  {r['path']}\n" for r in sorted(row["files"], key=lambda x: x["path"]))
        temporary = base / ("manifest.sha256.tmp." + collection_id)
        temporary.write_text(manifest_text, encoding="utf-8"); temporary.replace(base / "manifest.sha256")
        receipt = dict(status="complete", run_id=rid, study_id=identity["study_id"],
            execution_status=row["execution_status"], exit_code=row.get("exit_code"),
            snapshot_while_running=row["snapshot_while_running"], source_host=HOST, source_port=PORT,
            source_run=REMOTE_ROOT + "/runs/" + rid, collection_id=collection_id, collected_at=metadata["collected_at"],
            transferred_at=now(), verified_files=len(row["files"]), remote_only=row["remote_only"],
            include_checkpoints=args.include_checkpoints, archive_sha256=received["sha256"], changes=changes,
            scope="artifact transfer completed; execution/scientific outcome retain their own states; not Goal completion")
        dump(base / "transfer.json", receipt)
        statuses[stage] = {key: receipt[key] for key in ("run_id", "execution_status", "verified_files", "snapshot_while_running")}
    current = root / "CURRENT_REMOTE_STATE.json"
    keep_previous(current, root, collection_id)
    dump(current, dict(collected_at=metadata["collected_at"], transferred_at=now(), collection_id=collection_id,
        source_host=HOST, source_root=REMOTE_ROOT, root_files=metadata["root_state"], stages=statuses,
        scope="read-only remote snapshot; no execution or scientific-goal status is inferred"))
    summary = dict(collection_id=collection_id, status="verified", runs=statuses,
                   local_archive=str(archive), remote_archive=received["archive"], remote_archive_retained=True,
                   verified_staging=str(staged), verified_staging_retained=True)
    dump(transfer_dir / "COLLECTION_COMPLETE.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Local Study directory; defaults to this script's Study")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    root = Path(args.root or Path(__file__).resolve().parents[1]).resolve()
    claim = root / ".collection.claim"
    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        collect(args)
    except BaseException as exc:
        dump(root / ("COLLECTION_FAILURE_" + uuid.uuid4().hex + ".json"),
             dict(at=now(), error=repr(exc), stage=args.stage, remote_computation_untouched=True))
        raise
    finally:
        # Only remove the exact transport claim created by this invocation.
        if claim.is_file() and claim.read_text() == str(os.getpid()):
            claim.unlink()


if __name__ == "__main__":
    main()
