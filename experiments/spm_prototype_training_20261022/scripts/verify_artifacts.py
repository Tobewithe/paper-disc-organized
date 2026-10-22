"""Post-run CPU validation; reads trained artifacts without changing them."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True); args = p.parse_args()
    root = Path(args.root); pointer = json.loads((root/"TRAIN_POINTER.json").read_text())
    run = root/"runs"/pointer["run_id"]; torch.set_num_threads(4)
    record = json.loads((run/"run.json").read_text()); summary = json.loads((run/"SUMMARY.json").read_text())
    assert record["execution_status"] == "completed" and record["optimizer_steps"] == 2000
    memory = torch.load(run/"memory.pt", map_location="cpu", weights_only=True)
    projector = torch.load(run/"projector.pt", map_location="cpu", weights_only=True)
    assert sha(run/"memory.pt") == pointer["memory_sha256"] == summary["memory_sha256"]
    assert sha(run/"projector.pt") == memory["projector_sha256"]
    assert projector["optimizer_steps"] == 2000 and memory["dimension"] == 128
    data = torch.load(run/"FEATURES.pt", map_location="cpu", weights_only=False)
    cards = json.loads((run/"PROTOTYPE_CARDS.json").read_text()); rows = data["records"]
    train_sources = {r["source_id"] for r in rows if r["split"] == "train"}
    dev_sources = {r["source_id"] for r in rows if r["split"] == "dev"}
    assert not train_sources & dev_sources
    projected = F.linear((data["a"]-memory["input_mean"])/memory["input_std"], **memory["projector_state"])
    q = projected/torch.linalg.vector_norm(projected, dim=1, keepdim=True).clamp_min(1e-12)
    results = {}
    for label, name, partition in [(1, "foreground", "fg"), (0, "background", "bg")]:
        ix = [i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == label]
        centers = memory["keys_"+partition]
        assert centers.shape[1] == 128 and torch.isfinite(centers).all()
        assert torch.allclose(centers.norm(dim=1), torch.ones(len(centers)), atol=1e-5)
        scores = q[ix]@centers.T; assignments = scores.argmax(1)
        recount = [int((assignments == k).sum()) for k in range(len(centers))]
        stored = [c["tokens"] for c in cards[name]]
        assert recount == stored and sum(recount) == len(ix)
        domains = {"real": 0, "synthetic": 0}
        for card in cards[name]:
            for domain, count in card["domain_counts"].items(): domains[domain] += count
            assert all(r["source_id"] in train_sources and r["source_id"] not in dev_sources for r in card["examples"])
        assert domains == {d: sum(rows[i]["domain"] == d for i in ix) for d in domains}
        results[name] = {"K": len(centers), "tokens": len(ix), "domains": domains,
                         "card_reassignment_counts_exact": True, "unit_norm_passed": True}
    # Reimplement retrieval directly, independent of memory_query.forward.
    dev = [i for i, r in enumerate(rows) if r["split"] == "dev"][:512]
    cosine = q[dev]@memory["keys_fg"].T
    val, indices = torch.topk(cosine, k=min(4, len(memory["keys_fg"])), dim=1)
    response = (torch.softmax(val/.1, dim=1)[..., None]*memory["values_fg"][indices]).sum(1)
    from memory_query import FrozenPrototypeMemory
    with torch.no_grad(): loaded = FrozenPrototypeMemory(memory)(data["a"][dev])["foreground"]
    assert torch.allclose(response, loaded, atol=1e-6, rtol=1e-5)
    receipt = {"run_id": run.name, "slurm_job_id": record["slurm_job_id"], "projector_steps": 2000,
        "memory_sha256": sha(run/"memory.pt"), "projector_sha256": sha(run/"projector.pt"),
        "input_channels": memory["input_channels"], "dimension": 128,
        "projector_parameter_count": sum(v.numel() for v in projector["projector_state"].values()),
        "class_partitions": results, "heldout_sources_excluded_from_memory": True,
        "independent_retrieval_max_abs_error": float((response-loaded).abs().max()),
        "scientific_scope": "trained_artifact_validation_not_YOLO_AP_effectiveness"}
    (root/"ARTIFACT_ACCEPTANCE.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__": main()
