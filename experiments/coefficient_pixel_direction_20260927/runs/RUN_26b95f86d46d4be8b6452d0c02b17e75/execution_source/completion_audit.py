"""Requirement-level completion checks against actual 7Q artifacts."""
import argparse,json
import numpy as np
from common import *

def main(a):
    setup()
    a.out.mkdir(parents=True,exist_ok=True)
    queue=json.loads((ROOT/"QUEUE.json").read_text())
    completed=json.loads((ROOT/"COMPLETE.json").read_text())
    manifest=json.loads((PRIOR/"MANIFEST.json").read_text())
    fitdir=ROOT/"runs"/queue["runs"]["fit_dev_data"]
    testdir=Path(completed["test"]);evaldir=Path(completed["evaluation"])
    summarydir=Path(completed["summary"])
    fits=set();fit_count=0
    for p in sorted((fitdir/"fit").glob("PART_*.pt")):
        d=load(p);fit_count+=len(d["keys"]);fits.update(k[0] for k in d["keys"])
    dev=load(fitdir/"DEV.pt");test=load(testdir/"TEST.pt")
    devs=set(k[0] for k in dev["keys"]);tests=set(k[0] for k in test["keys"])
    assert fits.isdisjoint(devs) and fits.isdisjoint(tests) and devs.isdisjoint(tests)
    assert fits<=set(manifest["train_images"])
    assert devs<=set(manifest["old_development_images"])
    assert tests<=set(manifest["independent_test_images"])
    assert len(set(test["keys"]))==len(test["keys"])
    fit_info=json.loads((fitdir/"COMPLETE.json").read_text())
    test_info=json.loads((testdir/"COMPLETE.json").read_text())
    assert fit_count==fit_info["fit_candidates"]==71269
    assert len(test["keys"])==test_info["instances"]
    assert fit_info["max_h_error"]==fit_info["max_c_error"]==fit_info["max_box_error"]==0
    source_fit=json.loads((TARGETS/"COMPLETE.json").read_text())
    source_dev=json.loads(DEV_TARGETS.with_name("COMPLETE.json").read_text())
    assert source_fit["penalty"]==source_dev["penalty"]==.003
    assert test_info["max_stationary"]<=1e-3
    models=[]
    for seed in (0,1):
        for mode in MODES:
            for kind in KINDS:
                name=f"train_{mode}_{kind}_s{seed}"
                out=ROOT/"runs"/queue["runs"][name]
                rec=json.loads((out/"run.json").read_text())
                cp=load(out/"BEST.pt");history=json.loads((out/"TRAINING.json").read_text())
                assert rec["status"]=="completed" and rec["return_code"]==0
                assert (cp["mode"],cp["kind"],cp["seed"])==(mode,kind,seed)
                minimum=min(r["dev"]["loss"] for r in history["history"])
                selected=[r for r in history["history"] if r["epoch"]==cp["selected_epoch"]][0]
                assert abs(cp["dev"]["loss"]-selected["dev"]["loss"])<1e-9
                assert abs(cp["dev"]["loss"]-minimum)<=1.1e-5
                assert all(np.isfinite([r["fit"]["loss"],r["dev"]["loss"]]).all() for r in history["history"])
                models.append(dict(run_id=out.name,mode=mode,kind=kind,seed=seed,
                    parameters=cp["parameters"],selected_epoch=cp["selected_epoch"]))
    assert max(r["parameters"] for r in models)/min(r["parameters"] for r in models)<1.02
    predictions=load(evaldir/"PREDICTIONS.pt")
    assert predictions["keys"]==test["keys"]
    assert len(predictions["predictions"])==20
    assert all(torch.isfinite(t).all() for t in predictions["predictions"].values())
    metrics=np.load(evaldir/"METRICS.npz")
    assert np.array_equal(metrics["keys"],np.array(test["keys"]))
    values=metrics["values"];columns=metrics["metrics"].tolist()
    assert values.shape==(len(test["keys"]),34,10)
    iou=values[:,:,columns.index("iou")]
    assert np.isfinite(iou).all() and (iou>=0).all() and (iou<=1).all()
    cosine=values[:,:,columns.index("effect_cos")]
    assert np.nanmin(cosine)>=-1.0001 and np.nanmax(cosine)<=1.0001
    for key in ("grid_nmse","native_nmse"):
        assert np.nanmin(values[:,:,columns.index(key)] )>=0
    evaluation=json.loads((evaldir/"COMPLETE.json").read_text())
    assert evaluation["max_decode_error"]==0
    results=json.loads((summarydir/"RESULTS.json").read_text())
    assert results["candidates"]==len(test["keys"])
    assert set(results["results"])=={"all","failure","good_box_failure","original_success"}
    assert len(results["primary_contrasts"])==14
    for group,r in results["results"].items():
        assert len(r)==19
        for arm,score in r.items():
            assert score["delta_iou"]["ci95"][0]<=score["delta_iou"]["ci95"][1]
            if "_spatial" in arm or "_coefficient" in arm:
                assert len(score["repairs_per_seed"])==len(score["damage_per_seed"])==2
    audit_id=json.loads((ROOT/"AUDIT_RUN.json").read_text())["run_id"]
    numerical=json.loads((ROOT/"runs"/audit_id/"COMPLETE.json").read_text())
    assert numerical["decode_max_pixel_error"]==0 and numerical["oracle_full_projection_relative_error_max"]<1e-3
    result=dict(status="passed",fit_images_planned=10000,fit_images_effective=len(fits),fit_candidates=fit_count,
        dev_images_planned=200,dev_images_effective=len(devs),dev_candidates=len(dev["keys"]),
        test_images_planned=2000,test_images_effective=len(tests),test_candidates=len(test["keys"]),
        image_intersections=[len(fits&devs),len(fits&tests),len(devs&tests)],
        models=models,evaluation=str(evaldir),summary=str(summarydir),numerical_audit=numerical,
        missing_auc_candidates=int(np.isnan(values[:,0,columns.index("auc")]).sum()),
        missing_direction_candidates=int(np.isnan(values[:,0,columns.index("effect_cos")]).sum()),
        limitation="Reused held-out diagnostic test; two fitted seeds; official GT-conditioned candidates; not complete inference AP.")
    write(a.out/"AUDIT.json",result);write(a.out/"COMPLETE.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"models"}}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args())

