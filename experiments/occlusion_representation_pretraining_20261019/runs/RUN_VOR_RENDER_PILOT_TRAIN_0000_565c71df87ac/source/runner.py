import argparse
import os
from pathlib import Path
import subprocess
import sys

from .common import ROOT, Run, read, save, sha


def generate(args):
    if args.mode == "pilot":
        assert read(ROOT/"E0_PASS.json")["passed"], "E0 must pass before pilot generation"
        assert (ROOT/"TRAIN_FREEZE.json").exists(), "Pilot protocol must be frozen"
    manifest = ROOT / ("engineering_manifest.json" if args.mode == "engineering" else "pilot_manifest.json")
    rows = read(manifest)
    ids = args.ids.split(",") if args.ids else [x["id"] for x in rows]
    assert set(ids).issubset({x["id"] for x in rows})
    with Run("render_"+args.mode+"_"+ids[0], dict(
        manifest_sha256=sha(manifest), ids=ids, frames_per_variant=48,
        variants=3, training_steps=0, input_hashes=read(ROOT/"INPUT_MANIFEST.json"))) as run:
        installation = read("/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion/INSTALLATION.json")
        assert sha(installation["blender"]) == installation["blender_executable_sha256"]
        out = run.path/"dataset"
        command = [installation["blender"], "--background", "--factory-startup",
                   "--python-exit-code", "1", "--python",
                   str(ROOT/"shared/video_occlusion_pretraining/generate.py"), "--",
                   "--manifest", str(manifest), "--output", str(out),
                   "--ids", ",".join(ids)]
        save(run.path/"COMMAND.json", command)
        env = dict(os.environ, LD_LIBRARY_PATH=installation["private_runtime"]+":"+
                   os.environ.get("LD_LIBRARY_PATH", ""))
        with open(run.path/"blender.log", "w") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        assert result.returncode == 0, f"Blender exited {result.returncode}; see blender.log"
        for name in ids:
            assert read(out/name/"COMPLETE.json")["frames"] == 144
        save(run.path/"SUMMARY.json", dict(families=len(ids), clips=len(ids)*3,
             rgb_frames=len(ids)*144, dataset=str(out), scientific_result=False))
        save(ROOT/("dataset_"+args.mode+"_"+ids[0]+".json"),
             dict(dataset=str(out), run_id=run.id, ids=ids))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["render", "validate_data", "features", "smoke", "train", "evaluate"])
    p.add_argument("--mode", default="engineering")
    p.add_argument("--ids")
    p.add_argument("--dataset")
    p.add_argument("--arm", default="all")
    args = p.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "All GPU work must use Slurm"
    if args.stage == "render":
        generate(args)
    else:
        mod = __import__("shared.video_occlusion_pretraining."+args.stage, fromlist=["main"])
        mod.main(args)


if __name__ == "__main__":
    main()
