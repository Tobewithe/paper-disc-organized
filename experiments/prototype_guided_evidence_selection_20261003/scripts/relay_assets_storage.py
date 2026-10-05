"""Independent batched relay with explicit storage-policy continuation.

Reuse verified transport receipts. New val archives may occupy the server's
Study-specific system-disk directory; the original asset paths are symlinks to
the same hash-verified gzip bytes. Existing files remain in place. This worker
does not stop any other worker and never runs a model. Start only after the old
relay has been stopped and its receipts retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
import uuid

import relay_assets as base
import relay_assets_batched as batched


OVERFLOW_ROOT = "/root/prototype_guided_evidence_selection_20261003_overflow"
SYSTEM_RESERVE = 2 * 1024**3


def source_policy():
    path = base.LAPTOP_ROOT + "/assets/STORAGE_POLICY_AMENDMENT.json"
    code = "$ErrorActionPreference='Stop';$p=" + base.ps_quote(path) + ";"
    code += "@{text=[IO.File]::ReadAllText($p,[Text.Encoding]::UTF8);sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower()}|ConvertTo-Json -Depth 8 -Compress"
    value = json.loads(base.laptop_ssh(code, timeout=120))
    raw = value["text"].encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise AssertionError("Source storage-policy text/hash mismatch")
    return json.loads(raw), raw, value["sha256"]


class StorageRelay(batched.BatchedRelay):
    def __init__(self, args):
        super().__init__(args)
        self.original_producer = self.ids.get("superseded_asset_runs", [self.ids["runs"]["asset_pack"]])[-1]
        self.ids["runs"]["asset_pack"] = args.producer_run_id
        self.policy = None
        self.policy_sha256 = None
        self.run.update(script_sha256=base.sha(__file__),
                        parent_batched_relay_sha256=base.sha(batched.__file__),
                        producer_run_id=args.producer_run_id,
                        historical_producer_run_id=self.original_producer,
                        overflow_root=OVERFLOW_ROOT,
                        storage_scope="new val files only; existing bytes and paths retained")
        base.dump(self.out / "run.json", self.run)
        # Only this new interpreter is affected. The imported source files and
        # any earlier running relay remain untouched.
        base.commit_server = self.storage_commit_server

    def mirror(self, snapshot):
        policy, raw, digest = source_policy()
        if (policy["base_migration_fingerprint"] != snapshot["identity"]["fingerprint"]
                or policy["historical_max_asset_bytes"] != 47 * 1024**3
                or policy["effective_max_asset_bytes"] != 49 * 1024**3):
            raise AssertionError("Storage amendment is incompatible with the source assets")
        if self.policy_sha256 is not None and digest != self.policy_sha256:
            raise AssertionError("Storage policy changed while relay was running")
        if self.policy_sha256 is None:
            target = self.out / "STORAGE_POLICY_AMENDMENT.json"
            target.write_bytes(raw)
            base.server_put(target, base.SERVER_ROOT + "/STORAGE_POLICY_AMENDMENT.json.tmp", timeout=120)
            program = """from pathlib import Path
import hashlib,json,os
r=Path(ROOT);p=r/'STORAGE_POLICY_AMENDMENT.json';t=p.with_name(p.name+'.tmp')
if hashlib.sha256(t.read_bytes()).hexdigest()!=WANT:raise RuntimeError('Receiver policy hash mismatch')
if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=WANT:
 raise RuntimeError('Existing receiver policy differs')
os.replace(t,p)
print(json.dumps({'sha256':WANT}))
"""
            base.server_python("ROOT=" + repr(base.SERVER_ROOT) + "\nWANT=" + repr(digest) + "\n" + program)
            self.policy, self.policy_sha256 = policy, digest
            self.run["storage_policy_sha256"] = digest
            base.dump(self.out / "run.json", self.run)
            self.event("storage_policy_verified", sha256=digest, effective_budget_bytes=49 * 1024**3)
        super().mirror(snapshot)

    def storage_commit_server(self, metas):
        if self.policy_sha256 is None:
            raise RuntimeError("Storage policy must be mirrored before committing assets")
        program = """from pathlib import Path
import hashlib,json,os,shutil
r=Path(ROOT);p=r/'assets/images';incoming=r/'.relay_incoming';overflow=Path(OVERFLOW)/'images'
p.mkdir(parents=True,exist_ok=True);overflow.mkdir(parents=True,exist_ok=True)
if overflow.resolve()!=Path(OVERFLOW+'/images') or p.resolve()!=Path(ROOT+'/assets/images'):
 raise RuntimeError('Asset or overflow directory unexpectedly redirects')
def digest(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def verified(f,m):
 return f.is_file() and f.stat().st_size==m['compressed_bytes'] and digest(f)==m['compressed_sha256']
def allowed_existing(target):
 if target.is_symlink() and target.resolve().parent!=overflow.resolve():
  raise RuntimeError('Existing asset symlink leaves this Study overflow')
need=sum(m['compressed_bytes'] for m in METAS if m['split']=='val' and not (p/(f"{int(m['image_id']):012d}.pt.gz")).exists())
if need and shutil.disk_usage(overflow).free<need+RESERVE:raise RuntimeError('System disk overflow reserve would be reached')
done=[]
for m in METAS:
 iid=int(m['image_id']);name=f'{iid:012d}.pt.gz';target=p/name;source=incoming/name
 allowed_existing(target)
 if target.exists():
  if not verified(target,m):raise RuntimeError('Existing receiver asset differs: '+name)
 elif m['split']=='val':
  actual=overflow/name;temporary=actual.with_name(actual.name+'.tmp')
  if actual.exists():
   if not verified(actual,m):raise RuntimeError('Existing overflow bytes differ: '+name)
  else:
   if not verified(source,m):raise RuntimeError('Incoming val hash mismatch: '+name)
   if temporary.exists():
    if verified(temporary,m):os.replace(temporary,actual)
    else:
     retained=temporary.with_name(temporary.name+'.partial.'+RUN)
     if retained.exists():raise RuntimeError('Prior overflow partial already retained; inspect before retry')
     os.replace(temporary,retained)
   if not actual.exists():
    with source.open('rb') as src,temporary.open('wb') as dst:
     shutil.copyfileobj(src,dst,8*1024**2);dst.flush();os.fsync(dst.fileno())
    if not verified(temporary,m):raise RuntimeError('Copied overflow hash mismatch: '+name)
    os.replace(temporary,actual)
  linktemp=target.with_name(target.name+'.linktmp')
  if linktemp.is_symlink():
   if linktemp.resolve()!=actual.resolve():raise RuntimeError('Unexpected temporary asset symlink')
   linktemp.unlink()
  elif linktemp.exists():raise RuntimeError('Temporary symlink path is a regular file')
  if target.is_symlink():
   if target.resolve()!=actual.resolve():raise RuntimeError('Unexpected dangling asset symlink')
  os.symlink(str(actual),str(linktemp));os.replace(linktemp,target)
  if not verified(target,m):raise RuntimeError('Committed overflow symlink hash mismatch')
  if source.exists():
   if source.resolve().parent!=incoming.resolve() or not verified(source,m):raise RuntimeError('Unsafe incoming cleanup')
   source.unlink()
 else:
  if target.is_symlink():raise RuntimeError('Non-val target unexpectedly has a dangling symlink')
  if not verified(source,m):raise RuntimeError('Incoming asset hash mismatch: '+name)
  os.replace(source,target)
 mp=p/(f'{iid:012d}.meta.json')
 if mp.exists() and json.loads(mp.read_text())['compressed_sha256']!=m['compressed_sha256']:
  raise RuntimeError('Committed metadata differs')
 mt=mp.with_suffix(mp.suffix+'.tmp');mt.write_text(json.dumps(m,ensure_ascii=False,indent=2));os.replace(mt,mp)
 done.append({'image_id':iid,'sha256':m['compressed_sha256'],'physical_path':str(target.resolve()),'symlink':target.is_symlink()})
with (r/'STORAGE_PLACEMENTS.jsonl').open('a',encoding='utf8') as f:
 for row in done:f.write(json.dumps(dict(row,run_id=RUN,storage_policy_sha256=POLICY))+'\\n')
 f.flush();os.fsync(f.fileno())
print(json.dumps({'verified':done,'system_free_bytes':shutil.disk_usage(overflow).free,'data_free_bytes':shutil.disk_usage(r).free}))
"""
        header = "ROOT=" + repr(base.SERVER_ROOT) + "\nOVERFLOW=" + repr(OVERFLOW_ROOT)
        header += "\nMETAS=" + repr(metas) + "\nRESERVE=" + repr(SYSTEM_RESERVE)
        header += "\nRUN=" + repr(self.run_id) + "\nPOLICY=" + repr(self.policy_sha256) + "\n"
        return base.server_python(header + program, timeout=900)

    def finish(self, snapshot):
        # Reproduce the original completion gates, but publish the root receipt
        # only after recording the explicit storage amendment and placement.
        rows = [row for group in snapshot["index"].values() for row in group]
        if not snapshot.get("complete", {}).get("complete") or len(rows) != 13000:
            return False
        if any(row.get("status") != "complete" or str(row["image_id"]) not in self.receipts for row in rows):
            return False
        if set(self.receipts) != {str(row["image_id"]) for row in rows}:
            raise AssertionError("Transfer receipts include images outside the frozen manifest")
        complete = snapshot["complete"]
        if (complete["fingerprint"] != snapshot["identity"]["fingerprint"]
                or complete.get("storage_policy_sha256") != self.policy_sha256):
            raise AssertionError("Producer completion identity/storage policy differs")
        if complete.get("storage_resume_run_id") != self.args.producer_run_id:
            raise AssertionError("Completion belongs to a different producer Run; follow its registered continuation")
        self.event("final_receiver_hash_audit", images=len(rows))
        for start in range(0, len(rows), 64):
            metas = [self.receipts[str(row["image_id"])] for row in rows[start:start + 64]]
            state = base.remote_state(metas)
            if not all(row["present"] for row in state["files"]):
                raise RuntimeError("Final receiver inventory incomplete")
        self.mirror(snapshot)
        base.dump(self.out / "SOURCE_COMPLETE.json", complete)
        base.server_put(self.out / "SOURCE_COMPLETE.json", base.SERVER_ROOT + "/assets/COMPLETE.json.tmp", timeout=120)
        base.server_ssh("mv " + base.SERVER_ROOT + "/assets/COMPLETE.json.tmp " + base.SERVER_ROOT + "/assets/COMPLETE.json", timeout=60)
        placement_code = """from pathlib import Path
import json,shutil
r=Path(ROOT);p=r/'assets/images';o=Path(OVERFLOW)/'images';links=list(p.glob('*.pt.gz'))
linked=[q for q in links if q.is_symlink()]
if any(q.resolve().parent!=o.resolve() for q in linked):raise RuntimeError('Asset link leaves registered overflow')
print(json.dumps({'archives':len(links),'overflow_images':len(linked),'overflow_bytes':sum(q.stat().st_size for q in linked),'data_free_bytes':shutil.disk_usage(r).free,'system_free_bytes':shutil.disk_usage(o).free}))
"""
        placement = base.server_python("ROOT=" + repr(base.SERVER_ROOT) + "\nOVERFLOW=" + repr(OVERFLOW_ROOT) + "\n" + placement_code)
        if placement["archives"] != 13000:
            raise AssertionError("Unexpected receiver archive count")
        receipt = dict(status="complete", verified_images=len(rows),
                       verified_bytes=sum(m["compressed_bytes"] for m in self.receipts.values()),
                       completed_at=base.now(), fingerprint=complete["fingerprint"],
                       producer_run_id=self.args.producer_run_id, run_id=self.run_id,
                       storage_policy_sha256=self.policy_sha256, storage_placement=placement,
                       historical_budget_bytes=47 * 1024**3, effective_budget_bytes=49 * 1024**3,
                       no_model_computation=True, destination_F_audit="still required before training")
        base.dump(self.out / "transfer.json", receipt)
        base.dump(self.out / "COMPLETE.json", receipt)
        base.server_put(self.out / "transfer.json", base.SERVER_ROOT + "/assets/transfer.json", timeout=120)
        base.server_put(self.out / "transfer.json", base.SERVER_ROOT + "/ASSETS_TRANSFER_COMPLETE.json.tmp", timeout=120)
        base.server_ssh("mv " + base.SERVER_ROOT + "/ASSETS_TRANSFER_COMPLETE.json.tmp " + base.SERVER_ROOT + "/ASSETS_TRANSFER_COMPLETE.json", timeout=60)
        self.run.update(status="completed", completed_at=base.now(), exit_code=0)
        base.dump(self.out / "run.json", self.run)
        self.event("completed", verified_bytes=receipt["verified_bytes"], storage_placement=placement)
        return True


def publish_failure(relay, exc):
    error = dict(at=base.now(), error=repr(exc), traceback=traceback.format_exc())
    base.dump(relay.out / ("FAILURE_" + uuid.uuid4().hex + ".json"), error)
    relay.run.update(status="failed", exit_code=1, completed_at=base.now(), error=str(exc))
    base.dump(relay.out / "run.json", relay.run)
    receipt = dict(status="failed", run_id=relay.run_id, verified_images=len(relay.receipts), **error)
    base.dump(relay.out / "transfer.json", receipt)
    target = relay.out / "ASSETS_TRANSFER_FAILURE.json"
    base.dump(target, receipt)
    try:
        base.server_put(target, base.SERVER_ROOT + "/ASSETS_TRANSFER_FAILURE.json.tmp", timeout=120)
        code = "from pathlib import Path\nimport json,os\nr=Path(" + repr(base.SERVER_ROOT) + ")\np=r/'ASSETS_TRANSFER_COMPLETE.json'\ncomplete=p.exists() and json.loads(p.read_text()).get('status')=='complete'\nif not complete:os.replace(r/'ASSETS_TRANSFER_FAILURE.json.tmp',r/'ASSETS_TRANSFER_FAILURE.json')\nprint(json.dumps({'failure_published':not complete}))"
        base.server_python(code, timeout=120)
    except Exception as upload_error:
        base.dump(relay.out / "FAILURE_RECEIPT_UPLOAD_ERROR.json", dict(at=base.now(), error=str(upload_error)))
    relay.event("failed", error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--producer-run-id", required=True)
    parser.add_argument("--out")
    parser.add_argument("--resume-from", action="append", default=[])
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--poll-seconds", type=int, default=45)
    parser.add_argument("--max-hours", type=float, default=24)
    args = parser.parse_args()
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if any(not value.startswith("RUN_") or any(c not in safe for c in value) for value in (args.run_id, args.producer_run_id)):
        parser.error("Run IDs must be safe RUN_ identifiers")
    if not 1 <= args.batch_size <= 64 or not 1 <= args.poll_seconds <= 60 or args.max_hours <= 0:
        parser.error("batch 1..64, polling 1..60, max-hours positive")
    if args.limit_per_split is not None and (args.limit_per_split < 1 or not args.once):
        parser.error("--limit-per-split requires positive value and --once")
    relay = StorageRelay(args)
    try:
        relay.execute()
    except Exception as exc:
        publish_failure(relay, exc)
        raise


if __name__ == "__main__":
    main()
