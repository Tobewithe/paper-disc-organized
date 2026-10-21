"""Unmodified pinned TrackEval adapters and prior gap proxy definition."""
import importlib,sys,types
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
def iou_matrix(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(br - tl, 0).prod(-1)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)
    ba = np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)
    return inter / np.maximum(aa[:, None] + ba[None] - inter, 1e-8)


def official_metrics(root):
    # The pinned reference still uses removed NumPy aliases. Compatibility aliases
    # do not alter metric algorithms. Avoid importing unrelated optional datasets.
    for name,value in {"float":float,"int":int,"bool":bool}.items():
        if name not in np.__dict__:
            setattr(np,name,value)
    folder=root/"vendor/trackeval"
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

