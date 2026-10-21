"""One-off bounded Slurm stages. No scheduler/monitor/auto-resume service."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

from .common import ROOT,read,save,sha,now


def command(stage,*args):
    subprocess.run([sys.executable,"-m","shared.video_occlusion_pretraining.runner",stage,*args],
                   cwd=ROOT,check=True)


def freeze():
    target=ROOT/"TRAIN_FREEZE.json"
    assert not target.exists(),"Frozen contracts cannot be overwritten"
    assert read(ROOT/"E0_PASS.json")["passed"]
    smoke=read(ROOT/"SMOKE_PASS.json")
    assert smoke["passed"]
    for item in read(ROOT/"INPUT_MANIFEST.json"):
        assert sha(item["path"])==item["sha256"],item["path"]
    for item in read(ROOT/"BACKBONE_DOWNLOAD.json"):
        assert sha(ROOT/item["file"])==item["sha256"],item["file"]
    config=read(ROOT/"EXECUTION_CONFIG.json")
    config["formal_training_allowed"]=True
    config["design"]="v0.2-E1-frozen"
    save(ROOT/"EXECUTION_CONFIG.json",config)
    code=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
          for p in sorted((ROOT/"shared/video_occlusion_pretraining").glob("*.py"))]
    protocol=ROOT/"experiments/occlusion_representation_pretraining_20261019/EXECUTION_E1_V02.md"
    frozen=dict(at=now(),formal_training_allowed=True,steps_per_arm=2000,total_steps=8000,
        seed=0,code=code,config_sha256=sha(ROOT/"EXECUTION_CONFIG.json"),
        manifest_sha256=sha(ROOT/"pilot_manifest.json"),
        protocol_sha256=sha(protocol),input_manifest_sha256=sha(ROOT/"INPUT_MANIFEST.json"),
        backbone_download_manifest_sha256=sha(ROOT/"BACKBONE_DOWNLOAD.json"),
        e0_pass_sha256=sha(ROOT/"E0_PASS.json"),smoke_pass_sha256=sha(ROOT/"SMOKE_PASS.json"))
    save(target,frozen)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["prepare_pilot","fit_evaluate"])
    args=p.parse_args()
    if args.stage=="prepare_pilot":
        freeze()
        command("render","--mode","pilot")
        pointer=read(ROOT/"dataset_pilot_train_0000.json")
        command("validate_data","--mode","pilot","--dataset",pointer["dataset"])
        command("features","--mode","pilot","--dataset",pointer["dataset"])
    else:
        command("train","--mode","pilot","--arm","all")
        command("evaluate","--mode","pilot")
        subprocess.run([sys.executable,"-m","shared.video_occlusion_pretraining.visualize"],
                       cwd=ROOT,check=True)


if __name__=="__main__":
    main()
