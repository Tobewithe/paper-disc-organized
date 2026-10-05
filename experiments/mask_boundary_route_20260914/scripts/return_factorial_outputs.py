"""Finite collector for three named remote runs; never launches experiments."""
import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile


def write_json(path, data):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,indent=2),encoding="utf-8")
    os.replace(temp,path)


def remote_powershell(script, timeout=60):
    encoded=base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=8","28358lan",
        "powershell -NoProfile -EncodedCommand "+encoded],capture_output=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8",errors="replace"))
    return result.stdout.decode("utf-8-sig",errors="replace").strip()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--runs-root",required=True)
    p.add_argument("--source",required=True)
    p.add_argument("--dependents",nargs="*",default=[])
    p.add_argument("--timeout-hours",type=float,default=3)
    args=p.parse_args()
    root=Path(args.runs_root).resolve()
    source=root/args.source
    receipt=source/"return_status.json"
    pending=[args.source,*args.dependents]
    completed=[]
    deadline=time.monotonic()+args.timeout_hours*3600
    terminal_failed=False
    while pending and time.monotonic()<deadline:
        run_id=pending[0]
        if not run_id.startswith("RUN_") or not all(c in "RUN_0123456789abcdef" for c in run_id):
            raise ValueError("Invalid run ID")
        remote_dir="D:/coco_wire/runs/mask_boundary_route_20260914/"+run_id
        try:
            raw=remote_powershell("$p = '"+remote_dir+"/run.json'; if (Test-Path -LiteralPath $p) { Get-Content -LiteralPath $p -Raw }")
            status=json.loads(raw) if raw else {}
            if run_id==args.source and status:
                write_json(source/"run.json",status)
            if status.get("status") not in ("completed","failed","cancelled","interrupted"):
                time.sleep(30)
                continue
            remote_zip="D:/coco_wire/transfers/factorial_v3_20260915/"+run_id+".zip"
            packing="\n".join([
                "import os,zipfile", "from pathlib import Path",
                "root=Path("+repr(remote_dir)+")", "target=Path("+repr(remote_zip)+")",
                "target.parent.mkdir(parents=True,exist_ok=True)",
                "temporary=target.with_suffix('.tmp')",
                "with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:",
                " for f in root.rglob('*'):",
                "  if f.is_file(): z.write(f,f.relative_to(root).as_posix())",
                "os.replace(temporary,target)", "print(target.stat().st_size)"])
            remote_powershell("$ErrorActionPreference = 'Stop'\n$code = @'\n"+packing+
                "\n'@\n$code | & 'C:/Users/28358/anaconda3/envs/pytorch/python.exe' -\nif ($LASTEXITCODE -ne 0) { throw 'Packing failed' }",timeout=600)
            target=root/run_id
            target.mkdir(parents=True,exist_ok=True)
            archive=target/"remote_transfer.receiving.zip"
            subprocess.run(["scp","-o","BatchMode=yes","-o","ConnectTimeout=8","28358lan:"+remote_zip,str(archive)],check=True,timeout=900)
            copied=0
            byte_count=0
            with zipfile.ZipFile(archive) as z:
                entries=sorted(z.infolist(),key=lambda info:info.filename=="run.json")
                for info in entries:
                    relative=PurePosixPath(info.filename)
                    if info.is_dir():
                        continue
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError("Unsafe archive member")
                    destination=(target/Path(*relative.parts)).resolve()
                    if not destination.is_relative_to(target.resolve()):
                        raise RuntimeError("Archive outside target")
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    temporary=destination.with_suffix(destination.suffix+".receiving.tmp")
                    temporary.write_bytes(z.read(info))
                    os.replace(temporary,destination)
                    copied+=1
                    byte_count+=info.file_size
            archive.unlink()
            manifest=dict(run_id=run_id,source=remote_dir,files=copied,bytes=byte_count,
                transferred_at=datetime.now(timezone.utc).isoformat(),remote_originals_preserved=True,
                archive_integrity="ZIP CRC verified while reading every file",scientific_status=status.get("status"))
            write_json(target/"transfer_manifest.json",manifest)
            completed.append(manifest)
            pending.pop(0)
            terminal_failed |= status.get("status")!="completed"
            write_json(receipt,dict(status="collecting" if pending else "completed",collected=completed,pending=pending))
            print(json.dumps(manifest),flush=True)
            if run_id==args.source and terminal_failed:
                raise RuntimeError("Evaluation failed; dependent analyses were not started")
        except (subprocess.SubprocessError,OSError,json.JSONDecodeError) as exc:
            write_json(receipt,dict(status="retrying_transfer",error=str(exc),collected=completed,pending=pending))
            time.sleep(30)
    if pending:
        write_json(receipt,dict(status="timeout",collected=completed,pending=pending))
        raise TimeoutError("Remote outputs not all available before deadline")
    if terminal_failed:
        raise RuntimeError("Collected at least one failed run; inspect scientific logs")


if __name__=="__main__":
    main()
