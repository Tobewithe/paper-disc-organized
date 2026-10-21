"""Zero-update final source compatibility, semantic metric and E0 coverage audit."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

from .common import ROOT, Run, read, save, sha
from .data import FamilyData
from .evaluate import clip_metrics
from .model import VideoRepresentation
from .validate_data import validate
from .smoke import weight_hash


def main():
    with Run("final_audit",dict(training_updates=0,scope="source_compatibility_and_E0_coverage")) as run:
        torch.set_num_threads(4)
        first=read(ROOT/"dataset_engineering_engineering_0000.json")
        rest=read(ROOT/"dataset_engineering_engineering_0001.json")
        reports,files=[],[]
        for source in (first,rest):
            rows,manifest=validate(source["dataset"])
            reports.extend(rows)
            files.extend(manifest)
        assert len(reports)==8
        assert sum(r["external_return"] for r in reports)>0
        assert sum(r["external_long_return"] for r in reports)>0
        assert sum(r["censored"] for r in reports)>0
        assert any(r["event_type"]=="no_occlusion" for r in reports)
        assert any(r["event_type"]=="turn" for r in reports)
        pointer=read(ROOT/"features_engineering_first.json")
        data=FamilyData(pointer["dataset"],pointer["features"])
        inputs,targets=data.four_views("engineering_0000",2)
        smoke=read(ROOT/"SMOKE_PASS.json")
        old_path=Path(smoke["summary"]).parent/"source/model.py"
        spec=importlib.util.spec_from_file_location("smoke_frozen_model",old_path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(0)
        old=module.VideoRepresentation().cuda().eval()
        current=VideoRepresentation().cuda().eval()
        current.load_state_dict(old.state_dict())
        assert weight_hash(old)==weight_hash(current)
        with torch.inference_mode():
            error={}
            for arm in ("B","P","I","Hpos"):
                a=old(**inputs,arm=arm)
                b=current(**inputs,arm=arm)
                error[arm]=max(float((a[k]-b[k]).abs().max()) for k in a)
                assert error[arm]<=1e-6,error
            future=dict(inputs)
            future["features"]=inputs["features"].clone()
            future["features"][:,24:]=torch.randn_like(future["features"][:,24:])
            p=current(**inputs,arm="P")["logits"]
            other=current(**future,arm="P")["logits"]
            assert torch.equal(p[:,:24],other[:,:24])
            # Controlled interventions must be finite and actually affect predictions.
            interventions={}
            for name in ("other_query","matched_norm"):
                changed=current(**inputs,arm="P",memory_intervention=name)["logits"]
                assert torch.isfinite(changed).all()
                interventions[name]=float((p[:,1:]-changed[:,1:]).abs().max())
                assert interventions[name]>0
        # Event semantics: a wrong first reappearance followed by a correct match
        # must not receive R1 credit; a hidden-frame acceptance is a wrong link.
        _,lab=data.load("engineering_0000",0,0)
        pred=dict(xy=lab["xy"].numpy().copy(),confidence=np.ones((48,24)),
                  eligible=np.ones((48,24),bool))
        events=read(Path(pointer["dataset"])/"engineering_0000/0/events.json")
        eligible=[x for x in events if x["physical_external_only"] and x["gap"]>=3 and x["reappear"] is not None]
        assert eligible
        target=eligible[0]
        initial=clip_metrics(pred,lab,[target],.5)
        pred["xy"][target["reappear"],target["query"]]=[-1,-1]
        changed=clip_metrics(pred,lab,[target],.5)
        assert changed["R1"]==0 and changed["wrong"]>=initial["wrong"]
        hidden=((lab["state"].numpy()!=0) & lab["valid"].numpy() & lab["active"].numpy()[None])
        hidden[0]=False
        assert initial["wrong"]>=int(hidden.sum())
        save(run.path/"E0_DATA_MANIFEST.json",files)
        summary=dict(passed=True,families=reports,rgb_frames=1152,
            independent_ray_checks=1152*24,forward_equivalence_max_abs=error,
            memory_intervention_effect=interventions,engineering_updates=16,
            metric_first_reappearance_check=True,real_world_effect_tested=False)
        save(run.path/"SUMMARY.json",summary)
        save(ROOT/"E0_PASS.json",dict(passed=True,run_id=run.id,
            summary=str(run.path/"SUMMARY.json"),sha256=sha(run.path/"SUMMARY.json"),
            data_manifest=str(run.path/"E0_DATA_MANIFEST.json"),
            data_manifest_sha256=sha(run.path/"E0_DATA_MANIFEST.json")))
        print(summary,flush=True)


if __name__ == "__main__":
    main()
