import copy
import hashlib
import io
import time
import torch

from .common import ROOT, Run, read, save, sha
from .data import FamilyData
from .losses import base_losses, paired_content_loss
from .model import VideoRepresentation


def weight_hash(model):
    digest = hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(args):
    pointer = read(ROOT/("features_"+args.mode+".json"))
    with Run("smoke_"+args.mode,dict(feature_manifest_sha256=pointer["manifest_sha256"],
        steps_per_arm=4,arms=["B","P","I","Hpos"],test_data_used=False)) as run:
        torch.set_num_threads(4)
        torch.manual_seed(0)
        dataset = FamilyData(pointer["dataset"],pointer["features"])
        family = sorted(dataset.dataset.iterdir())[0].name
        inputs,targets = dataset.four_views(family,2)
        # Compare explicit supervision firewall with implementation signature.
        assert set(inputs) == {"features","timestamps","initial_points","initial_boxes"}
        model = VideoRepresentation().cuda()
        initial = copy.deepcopy(model.state_dict())
        initial_hash = weight_hash(model)
        results = []
        for arm in ["B","P","I","Hpos"]:
            model.load_state_dict(initial)
            assert weight_hash(model) == initial_hash
            optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for step in range(4):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    pred = model(**inputs,arm=arm)
                    loss,components = base_losses(pred,targets,arm)
                    cf,support = paired_content_loss(pred,targets,arm)
                    loss = loss+.1*cf
                assert torch.isfinite(loss), (arm,step)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                gradient_names = [n for n,p in model.named_parameters()
                                  if p.grad is not None and p.grad.abs().max().item() > 0]
                assert any(n.startswith("adapter.") for n in gradient_names)
                assert any(n.startswith("blocks.") for n in gradient_names)
                assert any(n.startswith("content.") for n in gradient_names), "No contrastive support"
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()
                run.update(training_steps_completed=len(results)*4+step+1)
            torch.cuda.synchronize()
            results.append(dict(arm=arm,loss=float(loss),support=support,
                initial_hash=initial_hash,total_parameters=sum(p.numel() for p in model.parameters()),
                gradient_parameters=sum(p.numel() for n,p in model.named_parameters() if n in gradient_names),
                inactive_parameter_names=[n for n,p in model.named_parameters() if n not in gradient_names],
                seconds_per_family_update=(time.perf_counter()-start)/4,
                max_memory_bytes=torch.cuda.max_memory_allocated(),
                components={k:float(v.detach()) for k,v in components.items()}))
        # Causality: changing only future RGB features cannot alter the prefix.
        model.eval()
        with torch.no_grad():
            before = model(**inputs,arm="P")["logits"]
            changed = dict(inputs)
            changed["features"] = inputs["features"].clone()
            changed["features"][:,24:] = torch.randn_like(changed["features"][:,24:])
            after = model(**changed,arm="P")["logits"]
            prefix_error = (before[:,:24]-after[:,:24]).abs().max().item()
            assert prefix_error <= 1e-6,prefix_error
            # I must not carry actual visual history into later outputs.
            original_i = model(**inputs,arm="I")["logits"]
            changed["features"] = inputs["features"].clone()
            changed["features"][:,1:24] = torch.randn_like(changed["features"][:,1:24])
            reset_i = model(**changed,arm="I")["logits"]
            history_error_i = (original_i[:,24:]-reset_i[:,24:]).abs().max().item()
            assert history_error_i <= 1e-6,history_error_i
        model.zero_grad(set_to_none=True)
        pred = model(**inputs,arm="P")
        pred["confidence"].square().mean().backward()
        leaked = [n for n,p in model.named_parameters() if not n.startswith("confidence.")
                  and p.grad is not None and p.grad.abs().max().item()>0]
        assert not leaked,leaked
        assert len({r["gradient_parameters"] for r in results}) == 1
        assert not any(r["inactive_parameter_names"] for r in results),results
        summary = dict(passed=True,initial_hash=initial_hash,arms=results,
            future_prefix_max_error=prefix_error,I_history_max_error=history_error_i,
            confidence_gradient_leak=leaked,steps=16,
            point_of_scope="engineering_only_no_effect_claim")
        save(run.path/"SUMMARY.json",summary)
        save(ROOT/"SMOKE_PASS.json",dict(passed=True,run_id=run.id,
             summary=str(run.path/"SUMMARY.json"),sha256=sha(run.path/"SUMMARY.json")))
        print(summary,flush=True)
