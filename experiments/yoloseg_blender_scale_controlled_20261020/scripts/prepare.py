"""CPU-only server preflight and source/input freeze, before sbatch."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")

assert not (ROOT/"FREEZE.json").exists(), "do not overwrite frozen source"
server=Path("/srv/gpu-lab/users/a25023104/projects")
original=server/"lcf-server-20261020/paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json"
profile=read(original)
profile["executor"]={"host":"master-01","environment":"linux_slurm","python":"3.12.3",
                     "gpu":"one Slurm-allocated RTX A6000","source_profile_sha256":sha(original)}
profile["budget"]={"smoke_images":16,"full_images":576,"training_steps":0,
                   "scope":"YB2; historical COCO budget does not apply to synthetic scenes"}
save(ROOT/"PROFILE.json",profile)
cfg={"profile":str(ROOT/"PROFILE.json"),
     "weights":str(server/"paper-disc-organized/assets/models/coco_clean_20260911/yolo26m-seg.pt"),
     "val_annotations":"/srv/gpu-lab/users/a25023104/datasets/coco2017/annotations/instances_val2017.json",
     "blender_installation":str(server/"dense-video-occlusion/INSTALLATION.json"),
     "source_profile":str(original),"source_profile_sha256":sha(original),
     "scope":"YB2 fixed projection synthetic video; no COCO AP replication claim"}
save(ROOT/"CONFIG.json",cfg)
assert sha(cfg["weights"])==profile["model_sha256"]
assert sha(cfg["val_annotations"])==profile["annotations_sha256"]
assetroot=server/"dense-video-occlusion-v2/assets"
manifest=read(assetroot/"ASSET_MANIFEST.json")
inputs=[]
for row in manifest:
    p=assetroot/row["path"].replace("\\","/")
    assert sha(p)==row["sha256"],str(p)
    inputs.append({"path":str(p),"sha256":row["sha256"],"role":"reused_read_only_asset"})
install=read(cfg["blender_installation"])
for name in [cfg["weights"],cfg["val_annotations"],cfg["blender_installation"],install["blender"],str(original),
             str(assetroot/"ASSET_MANIFEST.json")]:
    inputs.append({"path":name,"sha256":sha(name),"role":"pinned_external_input"})
assert sha(install["blender"])==install["blender_executable_sha256"]
files=[{"path":str(p.relative_to(ROOT)),"sha256":sha(p)} for p in sorted(ROOT.rglob("*"))
       if p.is_file() and p.name not in ["study.json"] and "__pycache__" not in p.parts]
save(ROOT/"FREEZE.json",{"protocol_version":1,"files":files,"external_inputs":inputs,
     "all_science_and_execution_sources_frozen_before_gpu":True})
print(json.dumps({"status":"prepared_and_frozen","files":len(files),"inputs":len(inputs),
     "freeze_sha256":sha(ROOT/"FREEZE.json"),"images":576,"smoke_images":16,"training_steps":0}))
