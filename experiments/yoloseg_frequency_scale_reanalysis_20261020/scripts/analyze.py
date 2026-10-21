"""Exploratory, immutable-input secondary analysis; no model or image inference."""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import csv, datetime, hashlib, json, socket, sys, time, traceback, uuid
from pathlib import Path
from collections import defaultdict
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
BANDS = ["low", "mid", "high"]
STUDY_ID = "STUDY_FREQ_SCALE_REANALYSIS_20261020"

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()

def decompose(p, g, e):
    c = (p + g - e) / 2
    prod = np.sqrt(p) * np.sqrt(g)
    amplitude = (np.sqrt(p)-np.sqrt(g))**2
    direction = e - amplitude
    tol = 2e-9 * np.maximum(1, np.maximum(p, np.maximum(g, e)))
    assert np.all(direction >= -tol), "negative direction term"
    assert np.all(np.abs(c) <= prod + tol), "Cauchy violation"
    direction = np.maximum(direction, 0)
    rho = np.divide(c, prod, out=np.full_like(c, np.nan), where=prod > 0)
    return amplitude, direction, rho

def synthetic():
    rng = np.random.default_rng(1729)
    x = rng.normal(size=(32, 32))
    x -= x.mean()
    answers = {}
    for name, y in [("scale", 2*x), ("shift", np.roll(x, 3, axis=1))]:
        fx = np.fft.fft2(x, norm="ortho")
        fy = np.fft.fft2(y, norm="ortho")
        p = np.array([np.sum(np.abs(fy)**2)])
        g = np.array([np.sum(np.abs(fx)**2)])
        e = np.array([np.sum(np.abs(fy-fx)**2)])
        a, d, _ = decompose(p, g, e)
        assert np.allclose(a+d, e, rtol=1e-12)
        assert np.allclose(d if name == "scale" else a, 0, atol=1e-9)
        answers[name] = {"amplitude_fraction":float(a[0]/e[0]), "direction_fraction":float(d[0]/e[0])}
    return answers

def macro(values, im, mask, weights):
    values = np.atleast_2d(values)
    if values.shape[0] == 1 and values.shape[1] == len(im):
        values = values.T
    arr = np.full((weights.shape[1], values.shape[1]), np.nan)
    for j in range(values.shape[1]):
        ok = mask & np.isfinite(values[:, j])
        count = np.bincount(im[ok], minlength=len(arr))
        sums = np.bincount(im[ok], weights=values[ok, j], minlength=len(arr))
        np.divide(sums, count, out=arr[:, j], where=count > 0)
    valid = np.isfinite(arr)
    point = np.nanmean(arr, axis=0)
    boots = (weights @ np.nan_to_num(arr)) / (weights @ valid.astype(float))
    return {"mean":point.tolist(),"ci95":np.quantile(boots,[.025,.975],axis=0).T.tolist(),
            "valid_images":valid.sum(0).tolist()}, boots

def adjusted(size, size_group, category, area, fail, image, values, weights):
    selected = np.flatnonzero(size_group == size)
    cells = defaultdict(lambda: [[], []])
    for i in selected:
        cells[(int(category[i]), int(np.floor(np.log2(area[i])/.5)))][int(fail[i])].append(i)
    common = [v for _,v in sorted(cells.items()) if v[0] and v[1]]
    nw = np.array([min(len(a),len(b)) for a,b in common], dtype=float)
    total_weight = nw.sum()
    pooled = np.zeros(values.shape[1])
    boots = np.zeros((len(weights), values.shape[1]))
    covered = [[], []]
    for w, (good, bad) in zip(nw, common):
        good, bad = np.array(good), np.array(bad)
        covered[0].extend(good.tolist())
        covered[1].extend(bad.tolist())
        pooled += w * (values[bad].mean(0)-values[good].mean(0))
        bg = weights[:,image[good]]
        bb = weights[:,image[bad]]
        boots += w * ((bb @ values[bad])/bb.sum(1)[:,None] - (bg @ values[good])/bg.sum(1)[:,None])
    if not common:
        return {"status":"no_common_support"}
    base_failure = int(np.sum((size_group == size) & fail))
    return {"status":"computed", "contrast":"failure_minus_success",
        "aggregation":"common category x half-octave input-area strata; fixed min(n0,n1) weights",
        "means":(pooled/total_weight).tolist(),
        "ci95":np.quantile(boots/total_weight,[.025,.975],axis=0).T.tolist(),
        "strata":len(common),"overlap_weight":float(total_weight),
        "failure_pairs":len(covered[1]),"success_pairs":len(covered[0]),
        "failure_images":len(set(image[covered[1]].tolist())),
        "failure_coverage":len(covered[1])/base_failure,
        "failure_total":base_failure}

def render(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    colors = ["#315A9B","#24A0A7","#EF9951"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    for j, band in enumerate(BANDS):
        for status, style in [("success","--"),("failure","-")]:
            y = [100*result["groups"][status+"_"+s]["error_fraction"]["mean"][j] for s in "SML"]
            axes[0].plot(range(3),y,style,marker="o",color=colors[j],label=band.title()+" / "+status)
    axes[0].set(xticks=range(3),xticklabels=["Small","Medium","Large"],ylabel="Share of residual AC energy (%)",
                title="Failure is relatively coarser within every COCO size",ylim=(0,100))
    axes[0].legend(fontsize=9,ncol=2,loc="upper center")
    for k,s in enumerate("SML"):
        raw=result["contrasts"][s]["error_fraction"]["mean"][0]*100
        adj=result["adjusted"][s]
        val=adj["means"][0]*100
        ci=np.array(adj["ci95"][0])*100
        axes[1].bar(k-.18,raw,.34,color="#BBC7D8",label="Raw image-macro" if k==0 else None)
        axes[1].bar(k+.18,val,.34,color=colors[0],label="Area + category conditioned" if k==0 else None)
        axes[1].errorbar(k+.18,val,yerr=[[val-ci[0]],[ci[1]-val]],color="#172C49",capsize=4)
        axes[1].text(k+.18,ci[1]+1.0,f"{val:.1f}",ha="center",fontsize=10)
    axes[1].set(xticks=range(3),xticklabels=["Small","Medium","Large"],
                ylabel="Failure - success low-band share (percentage points)",title="Does area/category composition explain the gap?")
    axes[1].axhline(0,color="#657080",lw=.7)
    axes[1].legend(loc="upper left",fontsize=9)
    fig.suptitle("YOLO26m-seg | COCO | Same fixed detections, exploratory server reanalysis",fontsize=14)
    for ext in ["png","svg"]:
        fig.savefig(out/("01_scale_conditioned."+ext),dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13,5.6),constrained_layout=True)
    xs=np.arange(9)
    amp=[]; direct=[]
    for s in "SML":
        g=result["groups"]["failure_"+s]
        amp.extend(np.array(g["amplitude_error_fraction"]["mean"])*100)
        direct.extend(np.array(g["direction_error_fraction"]["mean"])*100)
    axes[0].bar(xs,amp,color="#EF9951",label="Band total-amplitude difference")
    axes[0].bar(xs,direct,bottom=amp,color="#315A9B",label="Spectral-vector direction mismatch")
    axes[0].set(xticks=xs,xticklabels=[s+"\n"+b for s in "SML" for b in ["Low","Mid","High"]],
                ylim=(0,118),ylabel="Within-band error decomposition (%)",title="Failure masks: the exact energy identity")
    axes[0].legend(fontsize=9,loc="upper center")
    for x,y in zip(xs,direct):
        axes[0].text(x,104,f"{y:.0f}%",ha="center",fontsize=9)
    for j,b in enumerate(BANDS):
        for status,style in [("success","--"),("failure","-")]:
            y=[result["groups"][status+"_"+s]["spectral_cosine"]["mean"][j] for s in "SML"]
            axes[1].plot(range(3),y,style,marker="o",color=colors[j],label=b.title()+" / "+status)
    axes[1].set(xticks=range(3),xticklabels=["Small","Medium","Large"],ylim=(-.05,1.05),
                ylabel="Real normalized spectral inner product",title="Energy can be present while its pattern is wrong")
    axes[1].legend(fontsize=9,ncol=2)
    fig.suptitle("Direction mismatch includes phase AND within-band amplitude distribution; it is not proof of translation",fontsize=11)
    for ext in ["png","svg"]:
        fig.savefig(out/("02_amplitude_direction."+ext),dpi=180)
    plt.close(fig)

def main():
    stage = sys.argv[1]
    assert stage in ("smoke","full")
    assert os.environ.get("SLURM_JOB_ID"), "server analysis requires Slurm"
    freeze = read(ROOT/"FREEZE.json")
    for rel,expected in freeze["files"].items():
        assert sha(ROOT/rel) == expected, "frozen drift: "+rel
    if stage == "full":
        assert read(ROOT/"SMOKE_POINTER.json")["status"] == "completed"
    lock=ROOT/"stage.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps({"pid":os.getpid(),"job":os.environ["SLURM_JOB_ID"]}).encode())
    os.close(fd)
    run_id="RUN_FREQ_S2_"+stage.upper()+"_"+uuid.uuid4().hex
    out=ROOT/"runs"/run_id
    out.mkdir(parents=True)
    start=time.monotonic()
    meta={"study_id":STUDY_ID,"run_id":run_id,"stage":stage,"started_at":now(),
          "execution_status":"running","executor":socket.gethostname(),"pid":os.getpid(),
          "slurm_job_id":os.environ["SLURM_JOB_ID"],"command":sys.argv,
          "protocol_sha256":sha(ROOT/"PROTOCOL.md"),"freeze_sha256":sha(ROOT/"FREEZE.json"),
          "code_commit":freeze["code_commit"],"source_run_id":freeze["source_run_id"]}
    write(out/"run.json",meta)
    write(ROOT/"LATEST_RUN.json",{"run_id":run_id,"stage":stage})
    try:
        check={"synthetic":synthetic()}
        with (ROOT/"inputs/PER_INSTANCE.csv").open(encoding="utf-8-sig",newline="") as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==28377
        geometry={int(r["image_id"]):r for r in read(ROOT/"inputs/PER_IMAGE.json")}
        assert len(geometry)==5000
        if stage=="smoke":
            selected=set(sorted(geometry)[:16])
            rows=[r for r in rows if int(r["image_id"]) in selected]
        n=len(rows)
        def arr(key,dtype=float):
            return np.array([r[key] for r in rows],dtype=dtype)
        image_ids=arr("image_id",int)
        unique,im=np.unique(image_ids,return_inverse=True)
        fail=arr("own_mask75",int)==0
        size=arr("size_group",str)
        category=arr("category_id",int)
        area=arr("gt_area")*np.array([np.prod(geometry[int(i)]["ratio_pad"][0]) for i in image_ids])
        p=np.array([[float(r["pred_"+b+"_energy"]) for b in BANDS] for r in rows])
        g=np.array([[float(r["gt_"+b+"_energy"]) for b in BANDS] for r in rows])
        e=np.array([[float(r["error_"+b+"_energy"]) for b in BANDS] for r in rows])
        dc=arr("error_dc"); pixels=arr("error_pixels")
        assert np.all(p>=0) and np.all(g>=0) and np.all(e>=0)
        err=np.max(np.abs(e.sum(1)+dc-pixels))
        assert err<1e-6
        a,d,rho=decompose(p,g,e)
        frac=e/e.sum(1)[:,None]
        amp=np.divide(a,e,out=np.full_like(a,np.nan),where=e>0)
        direct=np.divide(d,e,out=np.full_like(d,np.nan),where=e>0)
        hann=np.array([[float(r["error_hann_"+b+"_fraction"]) for b in BANDS] for r in rows])
        identities=[(r["annotation_id"],r["output_detection_id"]) for r in rows]
        assert len(set(identities))==n
        check.update(pairs=n,images=len(unique),parseval_max_abs=float(err),
            decomposition_max_abs=float(np.max(np.abs(a+d-e))),
            min_direction_fraction=float(np.nanmin(direct)),max_direction_fraction=float(np.nanmax(direct)))
        import importlib.metadata as md
        write(out/"ENVIRONMENT.json",{"python":sys.version,"executable":sys.executable,
            "numpy":np.__version__,"matplotlib":md.version("matplotlib"),"psutil":psutil.__version__,
            "host":socket.gethostname(),"job_id":os.environ["SLURM_JOB_ID"],"cpu_only":True})
        contract=read(ROOT/"inputs/EVAL_CONTRACT.json")
        contract["secondary_analysis"]={"source_run_id":freeze["source_run_id"],"stage":stage,
            "inputs":"verified spectral exports only; no image/cache reread, no forward",
            "protocol_sha256":meta["protocol_sha256"],"parameters_changed":[]}
        write(out/"EVAL_CONTRACT.json",contract)
        boot_n=2000 if stage=="full" else 40
        weights=np.random.default_rng(1729).exponential(1,(boot_n,len(unique)))
        result={"groups":{},"contrasts":{},"adjusted":{},"n":n,"images":len(unique),
                "bootstrap":{"kind":"image-cluster Exp(1) multiplier","replicates":boot_n,"seed":1729}}
        stored={}
        old=read(ROOT/"inputs/SUMMARY.json")
        max_replay=0.
        for s in ["all","S","M","L"]:
            for status,ff in [("failure",fail),("success",~fail)]:
                mask=ff & ((size==s) if s!="all" else np.ones(n,dtype=bool))
                if not mask.any():
                    continue
                name=status+"_"+s
                group={"pairs":int(mask.sum()),"images":len(set(im[mask].tolist()))}
                for field,vals in [("error_fraction",frac),("hann_fraction",hann),
                    ("amplitude_error_fraction",amp),("direction_error_fraction",direct),("spectral_cosine",rho),
                    ("absolute_error_energy",e),("error_over_gt_area",e/arr("gt_area")[:,None])]:
                    group[field],stored[(name,field)]=macro(vals,im,mask,weights)
                group["pooled_pred_gt_energy_ratio"]=(p[mask].sum(0)/g[mask].sum(0)).tolist()
                group["pooled_direction_fraction"]=(d[mask].sum(0)/e[mask].sum(0)).tolist()
                if stage=="full":
                    diff=np.max(np.abs(np.array(group["error_fraction"]["mean"])-old["groups"][name]["error"]["image_macro_fraction"]))
                    max_replay=max(max_replay,float(diff))
                result["groups"][name]=group
            if stage=="full":
                result["contrasts"][s]={}
                for field in ["error_fraction","hann_fraction","spectral_cosine"]:
                    diff=np.array(result["groups"]["failure_"+s][field]["mean"])-np.array(result["groups"]["success_"+s][field]["mean"])
                    bs=stored[("failure_"+s,field)]-stored[("success_"+s,field)]
                    result["contrasts"][s][field]={"mean":diff.tolist(),"ci95":np.quantile(bs,[.025,.975],axis=0).T.tolist()}
            print("group complete",s,"elapsed",round(time.monotonic()-start,1),flush=True)
        if stage=="full":
            assert max_replay<1e-12
            values=np.column_stack([frac,hann])
            result["adjusted_fields"]=["low","mid","high","hann_low","hann_mid","hann_high"]
            for s in "SML":
                result["adjusted"][s]=adjusted(s,size,category,area,fail,im,values,weights)
                print("conditioned",s,result["adjusted"][s],flush=True)
            primary=result["adjusted"]["L"]
            enough=primary["failure_coverage"]>=.7 and primary["failure_pairs"]>=200 and primary["failure_images"]>=100
            ci=primary["ci95"][0]
            decision="positive_association" if enough and ci[0]>0 else ("negative_association" if enough and ci[1]<=0 else "inconclusive")
            result["primary_decision"]={"status":decision,"coverage_sufficient":enough,
                "claim_scope":"exploratory same-data association, not independent replication or causal proof"}
            render(result,out)
        write(out/"SUMMARY.json",result)
        check["original_summary_max_abs"]=max_replay
        check["rss_GiB"]=psutil.Process().memory_info().rss/2**30
        assert check["rss_GiB"]<4
        assert time.monotonic()-start<1700
        fields=["image_id","annotation_id","output_detection_id","size_group","category_id","gt_area","iou","own_mask75"]
        extra=["input_area"]+[b+"_"+suffix for b in BANDS for suffix in ["amplitude_error","direction_error","spectral_cosine"]]
        with (out/"DERIVED.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields+extra)
            writer.writeheader()
            for i,r in enumerate(rows):
                record={k:r[k] for k in fields}
                record["input_area"]=float(area[i])
                for j,b in enumerate(BANDS):
                    for suffix,val in [("amplitude_error",a[i,j]),("direction_error",d[i,j]),("spectral_cosine",rho[i,j])]:
                        record[b+"_"+suffix]=float(val) if np.isfinite(val) else ""
                writer.writerow(record)
        for rel,expected in freeze["files"].items():
            assert sha(ROOT/rel)==expected,"input changed during run"
        check["frozen_files_before_after_verified"]=len(freeze["files"])
        write(out/"VERIFICATION.json",check)
        write(out/"DECISION.json",result.get("primary_decision",{"status":"smoke_passed","scientific_claim":False}))
        meta.update(execution_status="completed",ended_at=now(),exit_code=0,seconds=time.monotonic()-start)
        write(out/"run.json",meta)
        if stage=="smoke":
            write(ROOT/"SMOKE_POINTER.json",{"run_id":run_id,"status":"completed"})
        write(out/"ARTIFACT_MANIFEST.json",{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
        print(json.dumps(meta),flush=True)
    except BaseException:
        (out/"ERROR.txt").write_text(traceback.format_exc(),encoding="utf-8")
        meta.update(execution_status="failed",ended_at=now(),exit_code=1)
        write(out/"run.json",meta)
        raise
    finally:
        lock.unlink()

if __name__=="__main__":
    main()
