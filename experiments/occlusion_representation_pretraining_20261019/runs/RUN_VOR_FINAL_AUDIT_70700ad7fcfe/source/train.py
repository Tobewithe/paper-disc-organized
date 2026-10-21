import copy
import json
from pathlib import Path
import random
import time

import torch

from .common import ROOT, Run, read, save, sha
from .data import FamilyData
from .losses import base_losses, paired_content_loss
from .model import VideoRepresentation
from .smoke import weight_hash


def verify_freeze():
    freeze = read(ROOT/"TRAIN_FREEZE.json")
    assert freeze["formal_training_allowed"]
    assert read(ROOT/"E0_PASS.json")["passed"]
    assert read(ROOT/"SMOKE_PASS.json")["passed"]
    for row in freeze["code"]:
        assert sha(ROOT/row["path"]) == row["sha256"], row["path"]
    assert sha(ROOT/"EXECUTION_CONFIG.json") == freeze["config_sha256"]
    assert sha(ROOT/"pilot_manifest.json") == freeze["manifest_sha256"]
    assert sha(ROOT/"experiments/occlusion_representation_pretraining_20261019/EXECUTION_E1_V02.md") == freeze["protocol_sha256"]
    assert sha(ROOT/"INPUT_MANIFEST.json") == freeze["input_manifest_sha256"]
    assert sha(ROOT/"BACKBONE_DOWNLOAD.json") == freeze["backbone_download_manifest_sha256"]
    return freeze


def main(args):
    freeze=verify_freeze()
    pointer=read(ROOT/"features_pilot.json")
    validation=read(ROOT/"validation_pilot.json")
    assert validation["valid"]
    assert validation["dataset"] == pointer["dataset"]
    features_run=Path(pointer["manifest"]).parent
    assert read(features_run/"EVAL_CONTRACT.json")["backbone_manifest_sha256"]==freeze["backbone_download_manifest_sha256"]
    cfg=read(ROOT/"EXECUTION_CONFIG.json")
    assert cfg["steps_per_arm"] == 2000 and cfg["effective_batch_families"] == 4
    manifest=read(ROOT/"pilot_manifest.json")
    train_rows=[x for x in manifest if x["split"] == "train"]
    assert len(train_rows) == 160
    groups={}
    for row in train_rows:
        groups.setdefault(row["event_type"],[]).append(row)
    group_names=sorted(groups)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    template=VideoRepresentation()
    initial=copy.deepcopy(template.state_dict())
    initial_hash=weight_hash(template)
    schedule=[]
    rng=random.Random(20261020)
    for step in range(2000):
        schedule.append([(rng.choice(groups[group_names[(step*4+k)%len(group_names)]])["id"],
                          1+(step+k)%2) for k in range(4)])
    schedule_hash=__import__("hashlib").sha256(json.dumps(schedule).encode()).hexdigest()
    arms=cfg["arms"] if args.arm == "all" else [args.arm]
    for p in (ROOT/"experiments/occlusion_representation_pretraining_20261019/runs").glob("*/run.json"):
        assert read(p)["stage_id"] not in {"train_"+a for a in arms}, (
            "A formal attempt already exists; no automatic restart or budget duplication",str(p))
    for arm in arms:
        assert arm in cfg["arms"]
        with Run("train_"+arm,dict(arm=arm,freeze_sha256=sha(ROOT/"TRAIN_FREEZE.json"),
            feature_manifest_sha256=pointer["manifest_sha256"],
            data_manifest_sha256=validation["manifest_sha256"],
            initial_state_sha256=initial_hash,batch_schedule_sha256=schedule_hash,
            training_steps=2000,seed=0,optimizer="AdamW",lr=1e-4,weight_decay=1e-4,
            precision="bfloat16_autocast_fp32_parameters",clip_grad_norm=1,
            gradient_accumulation=4,views_per_family=4,checkpoint_selection="last")) as run:
            data=FamilyData(pointer["dataset"],pointer["features"])
            model=VideoRepresentation().cuda()
            model.load_state_dict(initial)
            assert weight_hash(model) == initial_hash
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            save(run.path/"BATCH_SCHEDULE.json",schedule)
            torch.save(initial,run.path/"initial.pt")
            started=time.perf_counter()
            with open(run.path/"loss.jsonl","w") as log:
                for step,batch in enumerate(schedule,1):
                    optimizer.zero_grad(set_to_none=True)
                    rows=[]
                    for family,variant in batch:
                        inputs,targets=data.four_views(family,variant)
                        with torch.autocast("cuda",dtype=torch.bfloat16):
                            pred=model(**inputs,arm=arm)
                            loss,parts=base_losses(pred,targets,arm)
                            cf,support=paired_content_loss(pred,targets,arm)
                            total=(loss+.1*cf)/4
                        assert torch.isfinite(total), (step,family)
                        total.backward()
                        rows.append(dict(loss=float(total.detach())*4,cf=float(cf.detach()),
                                         cf_support=support,**{k:float(v.detach()) for k,v in parts.items()}))
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                    optimizer.step()
                    record=dict(step=step,gradient_norm=float(norm),seconds=time.perf_counter()-started,
                                loss=sum(x["loss"] for x in rows)/4,
                                cf_support=sum(x["cf_support"] for x in rows))
                    log.write(json.dumps(record)+"\n")
                    log.flush()
                    if step % 25 == 0 or step == 1:
                        print(arm,record,flush=True)
                        run.update(training_steps_completed=step)
                    # Recovery snapshots are recorded, never validation-selected.
                    if step % 500 == 0:
                        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                                        step=step,arm=arm,initial_sha256=initial_hash),
                                   run.path/f"step_{step:04d}.pt")
            run.update(training_steps_completed=2000)
            final=run.path/"step_2000.pt"
            save(ROOT/("trained_"+arm+".json"),dict(arm=arm,run_id=run.id,
                checkpoint=str(final),sha256=sha(final),steps=2000))
            save(run.path/"SUMMARY.json",dict(steps=2000,arm=arm,
                 effective_parameters=sum(p.numel() for p in model.parameters()),
                 checkpoint=str(final),checkpoint_sha256=sha(final),
                 seconds=time.perf_counter()-started,scientific_metrics_pending=True,
                 maximum_memory_bytes=torch.cuda.max_memory_allocated()))
            del model,optimizer,data
            torch.cuda.empty_cache()
