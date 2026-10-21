"""Export actual laptop run evidence; heavy feature tensors stay on the executor."""
from pathlib import Path
import importlib.metadata
import platform
import sys
import zipfile
from .common import read_json,write_json,sha256,now,setup_paths


def export(root):
    setup_paths(root)
    import torch
    packages={}
    for name in ["torch","torchvision","opencv-python","numpy","scipy","fastapi","uvicorn"]:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    write_json(root/"ENVIRONMENT.json",dict(recorded_at=now(),snapshot_timing="after fixed pilot",host=platform.node(),python=sys.version,
               executable=sys.executable,packages=packages,ultralytics_source="vendor/ultralytics_8_4_100",cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
               offline_laptop=True,transfer="Public staging + authenticated SCP over 192.168.10.x wired LAN"))
    files=[p for p in root.glob("*") if p.suffix in {".json",".md"}]
    files += [p for p in (root/"runs").rglob("*") if p.is_file() and p.suffix in {".json",".md",".txt",".pt",".py"}]
    files += [p for p in (root/"annotations").rglob("*") if p.is_file() and p.suffix in {".json",".md"}]
    receipt=[]
    with zipfile.ZipFile(root/"delivery.zip","w",zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(files):
            rel=p.relative_to(root).as_posix();receipt.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha256(p)));z.write(p,rel)
        z.writestr("TRANSFER_MANIFEST.json",__import__('json').dumps(receipt,indent=2))
    print(dict(file=str(root/"delivery.zip"),bytes=(root/"delivery.zip").stat().st_size,sha256=sha256(root/"delivery.zip"),files=len(receipt)),flush=True)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();export(a.root)
