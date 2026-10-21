"""One Slurm allocation, no training or automatic restart."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"sources"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    p=Path(path); tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)); os.replace(tmp,p)


def verify_freeze():
    manifest=read(ROOT/"FREEZE.json")
    for r in manifest["files"]:
        assert sha(ROOT/r["path"]) == r["sha256"], r["path"]
    for r in manifest["external_inputs"]:
        assert sha(r["path"]) == r["sha256"], r["path"]


def inference(run, cfg):
    import torch
    from native_stream import NativeBatches, build_dataset, dataset_plan, batch_identity
    prepared=run/"prepared"; prepared.mkdir()
    files=sorted((run/"dataset/images").glob("*.png"))
    assert len(files)==384
    (prepared/"val.txt").write_text("".join(str(p)+"\n" for p in files))
    _, dataset=build_dataset(cfg,prepared/"val.txt",prepared/"dataset")
    plan=dataset_plan(dataset)
    contract={"inputs":{k:{"sha256":sha(cfg[k])} for k in ["profile","weights","val_annotations"]},
              "val_batches":plan,"val_image_hashes":{str(int(p.stem)):sha(p) for p in files}}
    save(prepared/"DATA_CONTRACT.json",contract)
    del dataset
    stream=NativeBatches(cfg,prepared,"val",run/"native")
    save(run/"ENVIRONMENT.json",dict(native=stream.environment,
        gpu=str(torch.cuda.get_device_properties(0)), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        packages={p:importlib.metadata.version(p) for p in ["scipy","torch","torchvision"]}))
    output=[]; count=0
    for bi in range(len(plan)):
        caches, rows, features=stream.batch(bi,masks=True)
        identity=batch_identity(caches,rows,features)
        with (run/"NATIVE_IDENTITY.jsonl").open("a") as f:f.write(json.dumps(identity)+"\n")
        for group in rows:
            for row in group:
                row["output_detection_id"]=len(output)+1
                output.append(row)
        count+=len(caches)
        save(run/"PROGRESS.json",dict(stage="native_inference",frames=count,planned_frames=384,
                                     exact_zero_delta_RLE_checked=True,predictions=len(output)))
        print("INFERENCE_DONE",count,len(output),flush=True)
        del caches,rows,features
    save(run/"PREDICTIONS.json",output)
    save(run/"NATIVE_REPLAY.json",dict(status="passed",frames=count,predictions=len(output),
                                     all_exact_zero_delta_RLE=True,weights_sha256=sha(cfg["weights"])))


def main():
    verify_freeze()
    assert os.environ.get("SLURM_JOB_ID"), "GPU stages require Slurm"
    lock=ROOT/"ACTIVE.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    rid="RUN_YB_SMOKE_"+uuid.uuid4().hex
    run=ROOT/"runs"/rid; run.mkdir(parents=True)
    meta=dict(study_id="STUDY_YOLOSEG_BLENDER_STRESS_20261020",run_id=rid,stage_id="engineering_smoke",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),host=socket.gethostname(),
        slurm_job_id=os.environ["SLURM_JOB_ID"],execution_status="running",training_steps=0,
        command=sys.argv,protocol_sha256=sha(ROOT/"PROTOCOL.md"),freeze_sha256=sha(ROOT/"FREEZE.json"))
    save(run/"run.json",meta);save(ROOT/"EXECUTION_STATUS.json",meta)
    try:
        cfg=read(ROOT/"CONFIG.json")
        for package, expected in read(cfg["profile"])["packages"].items():
            assert importlib.metadata.version(package)==expected, ("package_drift",package)
        assert {c["name"]:c["id"] for c in read(cfg["val_annotations"])["categories"]}["cow"] == 21
        install=read(cfg["blender_installation"])
        assert sha(install["blender"])==install["blender_executable_sha256"]
        cmd=[install["blender"],"--background","--factory-startup","--python-exit-code","1",
             "--python",str(ROOT/"scripts/render.py"),"--",str(run/"dataset")]
        env=dict(os.environ)
        env.pop("PYTHONPATH",None)
        env["LD_LIBRARY_PATH"]=install["private_runtime"]+":"+env.get("LD_LIBRARY_PATH","")
        save(run/"RENDER_COMMAND.json",cmd)
        with (run/"blender.log").open("w") as f:
            completed=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        assert completed.returncode==0, f"Blender exit {completed.returncode}; preserve blender.log"
        inference(run,cfg)
        from evaluate import evaluate
        evaluate(run)
        verify_freeze()
        artifacts=[dict(path=str(p.relative_to(run)),size=p.stat().st_size,sha256=sha(p))
                   for p in sorted(run.rglob("*")) if p.is_file() and p.name!="run.json"]
        save(run/"ARTIFACT_MANIFEST.json",artifacts)
        meta.update(execution_status="completed",exit_code=0)
    except BaseException:
        (run/"ERROR.txt").write_text(traceback.format_exc())
        meta.update(execution_status="failed",exit_code=1)
        raise
    finally:
        meta["ended_at"]=time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save(run/"run.json",meta);save(ROOT/"EXECUTION_STATUS.json",meta)
        lock.unlink()


if __name__=="__main__":
    main()
