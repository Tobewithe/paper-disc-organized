from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .common import ROOT, Run, read, save, sha
from .data import FamilyData
from .model import VideoRepresentation, grid_xy


def native(features, points):
    t,n,d=features.shape
    maps=features[0].float().T.reshape(1,d,20,32)
    anchor=F.grid_sample(maps,points[None,:,None]*2-1,align_corners=False)[0,:,:,0].T
    sim=torch.einsum("qd,tnd->tqn",F.normalize(anchor,dim=-1),F.normalize(features.float(),dim=-1))
    score,idx=sim.max(-1)
    return dict(xy=grid_xy(20,32,features.device)[idx],confidence=(score+1)/2,
                eligible=torch.ones_like(score,dtype=torch.bool))


def infer(model,arm,feat,lab,intervention="none"):
    if arm == "N":
        return {k:v.cpu().numpy() for k,v in native(feat.cuda(),lab["initial_points"].cuda()).items()}
    inputs=dict(features=feat[None].cuda(),timestamps=lab["timestamps"][None].cuda(),
                initial_points=lab["initial_points"][None].cuda(),
                initial_boxes=lab["initial_boxes"][None].cuda())
    with torch.autocast("cuda",dtype=torch.bfloat16):
        out=model(**inputs,arm=arm,memory_intervention=intervention)
    p=out["logits"][0,:,:24].float()
    return dict(xy=out["xy"][0,:,:24].float().cpu().numpy(),
        confidence=out["confidence"][0,:,:24].float().sigmoid().cpu().numpy(),
        eligible=(p.argmax(-1) != 640).cpu().numpy())


def observations(pred,lab):
    a={k:v.numpy() for k,v in lab.items()}
    xy=pred["xy"]
    x=(xy[...,0]*128).astype(int).clip(0,127)
    y=(xy[...,1]*80).astype(int).clip(0,79)
    owner=a["owner_maps"][np.arange(48)[:,None],y,x]
    valid=a["valid"] & a["active"][None]
    valid[0]=False  # initial prompt is not a recovery decision
    visible=a["state"] == 0
    correct=(np.linalg.norm((xy-a["xy"])*256,axis=-1)<=4) & visible & (owner==a["owners"][None])
    return valid,visible,correct


def threshold(validation):
    candidates=np.linspace(0,1,101)
    best=None
    for threshold_ in candidates:
        tp=accepted=visible_count=0
        for pred,lab in validation:
            valid,visible,correct=observations(pred,lab)
            accept=pred["eligible"] & (pred["confidence"] >= threshold_) & valid
            tp+=int((accept & correct).sum())
            accepted+=int(accept.sum())
            visible_count+=int((visible & valid).sum())
        f=2*tp/max(1,visible_count+accepted)
        # Tie -> higher threshold (fewer unsupported links). Same rule for all arms.
        key=(f,threshold_)
        if best is None or key>best[0]:
            best=(key,dict(threshold=float(threshold_),f1=f,tp=tp,
                          accepted=accepted,visible_count=visible_count))
    return best[1]


def recovery_groups(events,ok):
    groups=defaultdict(list)
    for owner in (1,2,3):
        eligible=[e for e in events if e["owner"]==owner and e["gap"]>=3 and
                  e["physical_external_only"] and e["reappear"] is not None]
        eligible.sort(key=lambda e:(e["start"],e["end"],e["query"]))
        last_end=-1
        group=-1
        for e in eligible:
            if e["start"]>last_end:
                group+=1
            last_end=max(last_end,e["end"])
            groups[(owner,group)].append(bool(ok[e["reappear"],e["query"]]))
    return [float(np.mean(x)) for x in groups.values()]


def clip_metrics(pred,lab,events,thr):
    valid,visible,correct=observations(pred,lab)
    accept=valid & pred["eligible"] & (pred["confidence"]>=thr)
    ok=accept & correct
    wrong=accept & ~correct
    rec=recovery_groups(events,ok)
    # U: queries that stay physically visible throughout the full clip.
    continuous=(lab["state"].numpy()==0).all(0) & lab["active"].numpy()
    control=valid & continuous[None]
    xy=pred["xy"]
    ix=(xy[...,0]*128).astype(int).clip(0,127)
    iy=(xy[...,1]*80).astype(int).clip(0,79)
    hit_owner=lab["owner_maps"].numpy()[np.arange(48)[:,None],iy,ix]
    target_owner=lab["owners"].numpy()[None]
    cross_identity=accept & (hit_owner>=1) & (hit_owner<=3) & (hit_owner!=target_owner)
    invalid_surface=accept & ((hit_owner<1) | (hit_owner>3))
    object_recovery=[]
    ratio=lab["ratio"].numpy()
    inframe=lab["inframe"].numpy()
    # Conditional object relocalization through the predicted point positions.
    # GT ownership is used ONLY to evaluate those positions, not to select links.
    for owner in (1,2,3):
        qs=lab["owners"].numpy()==owner
        support=(accept[:,qs] & (hit_owner[:,qs]==owner)).sum(1)
        total_accept=accept[:,qs].sum(1)
        identity_ok=(support>=2) & (support*2>total_accept)
        visible_object=inframe[:,owner-1] & (ratio[:,owner-1]>0)
        if not lab["region_active"].numpy()[owner-1]:
            continue
        f=1
        while f<48:
            if not(inframe[f,owner-1] and ratio[f,owner-1]==0):
                f+=1
                continue
            start=f
            while f<48 and inframe[f,owner-1] and ratio[f,owner-1]==0:
                f+=1
            if f-start>=3 and f<48 and visible_object[f] and visible_object[start-1]:
                end=min(f+6,48)
                # At least three actual visible frames must be present.
                observations_=np.flatnonzero(visible_object[f:end])+f
                success=False
                for j in range(max(0,len(observations_)-2)):
                    trio=observations_[j:j+3]
                    if identity_ok[trio].all():
                        # Any accepted majority belonging to a competitor earlier
                        # in the window invalidates a claimed clean restoration.
                        earlier=np.arange(f,trio[-1]+1)
                        bad=(cross_identity[earlier][:,qs].sum(1)>support[earlier]).any()
                        success=not bad
                        break
                object_recovery.append(float(success))
    return dict(recovery_event_values=rec,
        R1=float(np.mean(rec)) if rec else None,
        W=float(wrong.sum()/max(1,valid.sum())),
        U=float(ok[control].mean()) if control.any() else None,
        coverage=float(accept.sum()/max(1,valid.sum())),
        accepted_error=float(wrong.sum()/accept.sum()) if accept.any() else None,
        proposals=int(valid.sum()),accepted=int(accept.sum()),wrong=int(wrong.sum()),
        continuous_proposals=int(control.sum()),recovery_events=len(rec),
        W_other_instance=float(cross_identity.sum()/max(1,valid.sum())),
        W_background_or_barrier=float(invalid_surface.sum()/max(1,valid.sum())),
        conditional_object_recovery_values=object_recovery,
        R_object_conditional=float(np.mean(object_recovery)) if object_recovery else None,
        full_object_occlusion_events=len(object_recovery),
        recovery_queries=sum(e["physical_external_only"] and e["gap"]>=3 and
                             e["reappear"] is not None for e in events))


def aggregate_family(clips):
    rec=[value for c in clips for value in c["recovery_event_values"]]
    output=dict(R1=float(np.mean(rec)) if rec else None)
    for k in ("W","U","coverage","accepted_error","W_other_instance","W_background_or_barrier",
              "R_object_conditional"):
        values=[c[k] for c in clips if c[k] is not None]
        output[k]=float(np.mean(values)) if values else None
    for k in ("proposals","accepted","wrong","continuous_proposals","recovery_events","recovery_queries",
              "full_object_occlusion_events"):
        output[k]=sum(c[k] for c in clips)
    return output


def compare(rows,left,right,seed=20261020):
    rng=np.random.default_rng(seed)
    result={}
    for metric in ("R1","W","U","coverage","W_other_instance","R_object_conditional"):
        shared=[name for name in rows[left] if rows[left][name][metric] is not None
                and rows[right][name][metric] is not None]
        values=np.asarray([rows[left][x][metric]-rows[right][x][metric] for x in shared])
        if not len(values):
            result[metric]=dict(delta_pp=None,ci95_pp=None,families=0)
            continue
        boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
        result[metric]=dict(delta_pp=float(values.mean()*100),
            ci95_pp=(np.quantile(boot,[.025,.975])*100).tolist(),families=len(values))
    return result


def judge(comparisons):
    p= comparisons["P-B"]
    i= comparisons["P-I"]
    if any(p[x]["ci95_pp"] is None for x in ("R1","W","U")):
        return dict(status="insufficient_event_denominator",
            next_action="resolve_missing_validation_population_before_effect_claim")
    benefit=p["R1"]["delta_pp"]>=3 and p["R1"]["ci95_pp"][0]>0
    guards=p["W"]["ci95_pp"][1]<=1 and p["U"]["ci95_pp"][0]>=-1
    temporal=i["R1"]["ci95_pp"] is not None and i["R1"]["ci95_pp"][0]>0
    if benefit and guards and temporal:
        return dict(status="synthetic_E1_pass",next_action="same_checkpoint_frozen_real_transfer_E2",
                    claim="paired_training_effect_on_asset_heldout_synthetic_events_only")
    if p["W"]["ci95_pp"][0]>1 or p["U"]["ci95_pp"][1]<-1:
        return dict(status="guardrail_harm",next_action="stop_this_configuration_preserve_negative")
    if p["R1"]["ci95_pp"][1]<3:
        return dict(status="no_practical_pair_increment",
                    next_action="prioritize_ordinary_video_representation_transfer_if_B_has_utility")
    return dict(status="inconclusive_at_frozen_budget",
                next_action="close_pilot_without_extra_steps_reconcile_event_level_evidence")


def plot(run,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms=["N","B","P","I","Hpos"]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
    for ax,metric,title in zip(axes,["R1","W","coverage"],
        ["Recovery at reappearance (%)","Wrong acceptance / all proposals (%)","Accepted coverage (%)"]):
        values=[summary["means"][a][metric] for a in arms]
        ax.bar(arms,[100*v if v is not None else 0 for v in values],
               color=["#87949b","#397bad","#e98939","#90b3c9","#77845d"])
        ax.set_title(title,fontsize=10)
        for n,v in enumerate(values):
            if v is not None:ax.text(n,100*v,f"{100*v:.2f}",ha="center",va="bottom",fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Held-out synthetic video events — fixed final checkpoints",fontsize=12)
    fig.savefig(run.path/"synthetic_results.png",dpi=180)
    plt.close(fig)


def main(args):
    pointer=read(ROOT/"features_pilot.json")
    configs=read(ROOT/"pilot_manifest.json")
    with Run("evaluate_pilot",dict(feature_manifest_sha256=pointer["manifest_sha256"],
        split_counts=dict(train=160,validation=40,test=80),threshold_split="validation_only",
        threshold_grid=list(np.linspace(0,1,101)),bootstrap="paired_scene_family_2000",
        metrics=["R1","W","U","coverage"],correct_radius_at_256=4,
        all_three_physical_variants=True,physical_external_gap_min=3,
        real_video_metrics=False)) as run:
        torch.set_num_threads(4)
        results,thresholds,predictions={}, {}, {}
        for result_name in ["N","B","P","I","Hpos","P_other_query","P_matched_norm"]:
            arm = "P" if result_name.startswith("P_") else result_name
            intervention = result_name[2:] if result_name.startswith("P_") else "none"
            data=FamilyData(pointer["dataset"],pointer["features"])
            model=None
            if arm!="N":
                source=read(ROOT/("trained_"+arm+".json"))
                assert sha(source["checkpoint"])==source["sha256"]
                model=VideoRepresentation().cuda().eval()
                model.load_state_dict(torch.load(source["checkpoint"],map_location="cuda",weights_only=True)["model"])
            valid=[]
            with torch.inference_mode():
                if intervention == "none":
                    for cfg in configs:
                        if cfg["split"]!="validation":continue
                        for variant in range(3):
                            feat,lab=data.load(cfg["id"],variant,0)
                            pred=infer(model,arm,feat,lab)
                            valid.append((pred,lab))
                    selected=threshold(valid)
                else:
                    selected=thresholds["P"]  # no threshold retuning after intervention
                thresholds[result_name]=selected
                save(run.path/"VALIDATION_THRESHOLDS.json",thresholds)
                families={}
                for cfg in configs:
                    if cfg["split"]!="test":continue
                    clips=[]
                    for variant in range(3):
                        feat,lab=data.load(cfg["id"],variant,0)
                        pred=infer(model,arm,feat,lab,intervention)
                        events=read(Path(pointer["dataset"])/cfg["id"]/str(variant)/"events.json")
                        metrics=clip_metrics(pred,lab,events,selected["threshold"])
                        clips.append(metrics)
                        out=run.path/"predictions"/result_name/cfg["id"]
                        out.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(out/f"{variant}.npz",**pred)
                    families[cfg["id"]]=aggregate_family(clips)
                results[result_name]=families
            del data,model
            torch.cuda.empty_cache()
            print("evaluated",result_name,flush=True)
        means={}
        for arm,rows in results.items():
            means[arm]={k:float(np.mean([v[k] for v in rows.values() if v[k] is not None]))
                        if any(v[k] is not None for v in rows.values()) else None
                        for k in ["R1","W","U","coverage","accepted_error",
                                  "W_other_instance","W_background_or_barrier","R_object_conditional"]}
        pairs=[("P","B"),("P","I"),("Hpos","P"),("P","N"),("B","N"),
               ("P","P_other_query"),("P_matched_norm","P_other_query")]
        comparisons={a+"-"+b:compare(results,a,b) for a,b in pairs}
        decision=judge(comparisons)
        summary=dict(means=means,paired_comparisons=comparisons,thresholds=thresholds,
                     decision=decision,population="heldout_synthetic_assets_not_real_video",
                     mature_tracker_comparison="not_measured_in_E1",
                     seed_count=1,checkpoints="final_2000_only")
        save(run.path/"FAMILY_METRICS.json",results)
        save(run.path/"SUMMARY.json",summary)
        save(run.path/"DECISION.json",decision)
        plot(run,summary)
        run.update(scientific_status=decision["status"])
        save(ROOT/"E1_RESULT.json",dict(run_id=run.id,path=str(run.path),decision=decision))
        print(summary,flush=True)
