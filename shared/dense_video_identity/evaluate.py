from __future__ import annotations

import importlib
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import Run, iou_matrix, load_gt, read_json, setup_paths, write_json
from .learn import embed_array, load_cache, load_identity, load_utility
from .tracking import run_sequence


def official_metrics(root):
    # The pinned reference still uses removed NumPy aliases. Compatibility aliases
    # do not alter metric algorithms. Avoid importing unrelated optional datasets.
    for name,value in {"float":float,"int":int,"bool":bool}.items():
        if name not in np.__dict__:
            setattr(np,name,value)
    folder=root/"vendor/TrackEval-master/trackeval"
    for name,p in [("trackeval",folder),("trackeval.metrics",folder/"metrics")]:
        if name not in sys.modules:
            mod=types.ModuleType(name);mod.__path__=[str(p)];sys.modules[name]=mod
    hota=importlib.import_module("trackeval.metrics.hota").HOTA()
    identity=importlib.import_module("trackeval.metrics.identity").Identity({"PRINT_CONFIG":False})
    clear=importlib.import_module("trackeval.metrics.clear").CLEAR({"PRINT_CONFIG":False})
    return dict(HOTA=hota,Identity=identity,CLEAR=clear)


def metric_data(gt, outputs):
    gids=sorted({int(r[1]) for a in gt.values() for r in a})
    tids=sorted({int(t["identity"]) for f in outputs for t in f["tracks"] if t["identity"] is not None})
    gi={v:i for i,v in enumerate(gids)};ti={v:i for i,v in enumerate(tids)}
    data=dict(num_timesteps=len(outputs),num_gt_ids=len(gids),num_tracker_ids=len(tids),
              gt_ids=[],tracker_ids=[],similarity_scores=[],num_gt_dets=0,num_tracker_dets=0)
    for out in outputs:
        g=gt[out["frame"]];tracks=[t for t in out["tracks"] if t["identity"] is not None]
        b=np.asarray([t["box"] for t in tracks],np.float32).reshape(-1,4)
        data["gt_ids"].append(np.asarray([gi[int(r[1])] for r in g],np.int64))
        data["tracker_ids"].append(np.asarray([ti[int(t["identity"])] for t in tracks],np.int64))
        data["similarity_scores"].append(iou_matrix(g[:,2:6],b))
        data["num_gt_dets"]+=len(g);data["num_tracker_dets"]+=len(tracks)
    return data


def reappearance_events(gt, outputs):
    per_id={};matches={};seen={}
    for frame,g in gt.items():
        for row in g:
            per_id.setdefault(int(row[1]),[]).append(frame)
    for out in outputs:
        f=out["frame"];g=gt[f];tracks=[t for t in out["tracks"] if t["identity"] is not None]
        sim=iou_matrix(g[:,2:6],np.asarray([t["box"] for t in tracks]).reshape(-1,4))
        if sim.size:
            rr,cc=linear_sum_assignment(-sim)
            for r,c in zip(rr,cc):
                if sim[r,c]>=.5:
                    gid=int(g[r,1]);pid=int(tracks[c]["identity"])
                    matches[f,gid]=pid;seen.setdefault(pid,[]).append((f,gid))
    events=[]
    for gid,frames in per_id.items():
        for i in range(1,len(frames)):
            before,after=frames[i-1],frames[i];gap=after-before-1
            if not 2<=gap<=60:
                continue
            left=frames[max(0,i-3):i];right=frames[i:i+3]
            boundary=len(left)<3 or len(right)<3
            previous=[matches[f,gid] for f in left if (f,gid) in matches]
            prior=Counter(previous).most_common(1)[0][0] if previous else None
            following=[(f,matches[f,gid]) for f in frames[i:] if f<=after+12 and (f,gid) in matches]
            correct=[f for f,pid in following if pid==prior] if prior is not None else []
            if boundary:
                outcome="boundary_insufficient"
            elif prior is None or not following:
                outcome="unrecovered"
            elif correct:
                outcome="same_id"
            else:
                new=following[0][1]
                belonged_other=any(f<after and old!=gid for f,old in seen.get(new,[]))
                outcome="wrong_reuse" if belonged_other else "new_id"
            events.append(dict(gt_id=gid,before=before,after=after,gap_frames=gap,outcome=outcome,
                               prior_id=prior,delay_frames=min(correct)-after if correct else None,
                               source="GT temporal gap; physical cause unlabelled"))
    counts=Counter(e["outcome"] for e in events)
    eligible=[e for e in events if e["outcome"]!="boundary_insufficient"]
    return dict(events=events,total=len(events),eligible=len(eligible),counts=dict(counts),
                correct_rate=counts["same_id"]/len(eligible) if eligible else None,
                wrong_rate=counts["wrong_reuse"]/len(eligible) if eligible else None)


def compact(results):
    h,i,c=results["HOTA"],results["Identity"],results["CLEAR"]
    return dict(IDF1=float(i["IDF1"]*100),HOTA=float(np.mean(h["HOTA"])*100),
                AssA=float(np.mean(h["AssA"])*100),DetA=float(np.mean(h["DetA"])*100),
                IDSW=int(c["IDSW"]),FP=int(c["CLR_FP"]),FN=int(c["CLR_FN"]),
                IDTP=int(i["IDTP"]),IDFP=int(i["IDFP"]),IDFN=int(i["IDFN"]))


def evaluate(root:Path,config:dict,split="dev"):
    setup_paths(root)
    manifest=read_json(root/"input_manifest.json")
    seqs=[s for s in manifest["sequences"] if s["split"]==split]
    identity,ck=load_identity(root);utility=load_utility(root)
    metrics=official_metrics(root)
    contract=dict(config=config,split=split,sequences=[s["name"] for s in seqs],
                  identity=read_json(root/"identity_pointer.json"),utility=read_json(root/"utility_pointer.json"),
                  features=read_json(root/("demo_features_pointer.json" if split=="demo" and (root/"demo_features_pointer.json").exists() else "features_pointer.json")),metric_source="TrackEval 12c8791b303e0a0b50f753af204249e622d0281a",
                  gt_permissions="evaluation only",prediction_geometry="original cached detection boxes",
                  provisional_policy="not emitted as confirmed identity; counted as missed in strict MOT; no retroactive relabel")
    with Run(root,"evaluate_"+split,contract,[contract["identity"]["run_id"],contract["utility"]["run_id"]]) as run:
        raw_results={a:{} for a in config["arms"]};table=[];all_events={}
        for seq in seqs:
            records,raw=load_cache(root,seq);z=embed_array(identity,ck,raw)
            # Deliberately load evaluation GT only after the shared inference inputs exist.
            gt=load_gt(root,seq) if seq["gt"] else None
            for arm in config["arms"]:
                start=time.perf_counter()
                outputs,events=run_sequence(arm,records,z,config,utility)
                duration=time.perf_counter()-start
                dest=run.path/arm/seq["name"]
                write_json(dest/"tracks.json",outputs);write_json(dest/"policy_events.json",events)
                n=sum(len(f["tracks"]) for f in outputs)
                stat=dict(arm=arm,sequence=seq["name"],frames=len(records),tracking_seconds=duration,
                          tracking_ms_per_frame=duration/len(records)*1000,observations=n,
                          pending=sum(t["identity"] is None for f in outputs for t in f["tracks"]),
                          recovery_attempts=len(events["recovery_events"]),
                          memory_writes=sum(e["write"] for e in events["memory_events"]),memory_decisions=len(events["memory_events"]))
                if gt is not None:
                    data=metric_data(gt,outputs)
                    rs={k:m.eval_sequence(data) for k,m in metrics.items()}
                    raw_results[arm][seq["name"]]=rs
                    stat.update(compact(rs));stat["gt_detections"]=data["num_gt_dets"]
                    re=reappearance_events(gt,outputs);write_json(dest/"reappearance_events.json",re)
                    all_events[arm,seq["name"]]=re
                    stat.update(reappearance_events=re["eligible"],correct_recoveries=re["counts"].get("same_id",0),wrong_recoveries=re["counts"].get("wrong_reuse",0))
                    write_json(dest/"metric_details.json",rs)
                table.append(stat);print("EVAL",stat,flush=True)
        combined={}
        if split!="demo":
            for arm in config["arms"]:
                rs={k:m.combine_sequences({s:raw_results[arm][s][k] for s in raw_results[arm]}) for k,m in metrics.items()}
                combined[arm]=compact(rs)
                ev=[all_events[arm,s["name"]] for s in seqs]
                combined[arm].update(events=sum(e["eligible"] for e in ev),correct_recoveries=sum(e["counts"].get("same_id",0) for e in ev),wrong_recoveries=sum(e["counts"].get("wrong_reuse",0) for e in ev))
            rng=np.random.default_rng(config["seed"]);boot={a:[] for a in ("bytetrack","deepocsort","freeze_recovery")}
            names=[s["name"] for s in seqs]
            for _ in range(500):
                sample=rng.choice(names,size=len(names),replace=True)
                ids={a:metrics["Identity"].combine_sequences({str(j):raw_results[a][s]["Identity"] for j,s in enumerate(sample)})["IDF1"]*100 for a in ["utility_recovery"]+list(boot)}
                for a in boot:boot[a].append(ids["utility_recovery"]-ids[a])
            uncertainty={a:dict(delta_IDF1=combined["utility_recovery"]["IDF1"]-combined[a]["IDF1"],sequence_bootstrap_95=np.percentile(boot[a],[2.5,97.5]).tolist(),clusters=len(seqs),scope="descriptive small-sample") for a in boot}
        else:
            uncertainty={}
        write_json(run.path/"SUMMARY.json",dict(split=split,by_sequence=table,combined=combined,comparisons=uncertainty))
        if split=="test":
            m=combined["utility_recovery"];r=combined["deepocsort"];n=combined["bytetrack"]
            gates=dict(vs_native=m["IDF1"]-n["IDF1"]>=1,vs_reference=m["IDF1"]-r["IDF1"]>=1,
                       per_sequence=all(next(t for t in table if t["arm"]=="utility_recovery" and t["sequence"]==s)["IDF1"]>=next(t for t in table if t["arm"]=="deepocsort" and t["sequence"]==s)["IDF1"]-3 for s in names),
                       event_guard=None if m["events"]<20 else m["wrong_recoveries"]/m["events"]-r["wrong_recoveries"]/r["events"]<=.05)
            status="failed" if any(v is False for v in gates.values()) else "insufficient" if any(v is None for v in gates.values()) else "passed"
            decision=dict(status=status,gates=gates,mechanism_increment_vs_freeze=uncertainty["freeze_recovery"],
                          next_action="expand independent annotated sequences" if status=="passed" else "prioritize identity discriminability and independent occlusion labels; stop this fixed configuration",
                          claim_scope="5-sequence pilot; no whole-benchmark or universal robustness claim")
            write_json(run.path/"DECISION.json",decision);run.meta["scientific_status"]=status
        write_json(root/("evaluation_"+split+"_pointer.json"),dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
    return run.id
