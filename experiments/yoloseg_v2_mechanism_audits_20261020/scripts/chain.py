"""Trace actual selected raw rows; factorial P/c swaps are GT-assisted diagnostics."""
import collections
import copy
from common import *


@torch.inference_mode()
def execute(run, smoke):
    inp = Inputs()
    prep = inp.prepared(run)
    stream = NativeBatches(inp.cfg, prep, "val", run / "native")
    save(run / "ENVIRONMENT.json", stream.environment)
    before_args = read(run / "native/NATIVE_ARGS.json")
    assert before_args == read(inp.base / "native/NATIVE_ARGS.json")
    held = {}
    def observer(module, args, output):
        held["raw"] = output[1]["one2one"]
        held["final"] = output[0][0]
    hook = stream.model.register_forward_hook(observer)
    indices = [0, 3, 6, 9] if smoke else list(range(36))
    caches_by_id = {}
    stage_rows = []
    replay_count = 0
    for bi in indices:
        caches, rows, features = stream.batch(bi, masks=True)
        append(run / "IDENTITY.jsonl", inp.check_base(bi, caches, rows, features))
        raw = held["raw"]
        raw_boxes = stream.head._get_decode_boxes(raw).transpose(1, 2)
        raw_scores = raw["scores"].sigmoid().transpose(1, 2)
        for pos, (c, group) in enumerate(zip(caches, rows)):
            iid = c["image_id"]
            ids = c["raw_ids"]
            assert torch.equal(raw_boxes[pos, ids], c["boxes_input"])
            assert torch.equal(raw_scores[pos, ids, c["classes"].long()], c["scores"])
            final = held["final"][pos]
            first = raw_scores[pos].amax(1).topk(300).indices.cpu().numpy()
            assert set(ids) <= set(first.tolist())
            # Retain native coordinate boxes; GT coordinates have fixed +16 letterbox.
            assert c["ratio_pad"] == ((1.0, 1.0), (16, 16)) or list(c["ratio_pad"]) == [[1.0, 1.0], [16, 16]]
            boxes = raw_boxes[pos].cpu().numpy() - np.array([16, 16, 16, 16])
            scores = raw_scores[pos].cpu().numpy()
            matches = inp.matches(iid, group)
            for g in inp.frames[iid]["records"]:
                tid = g["track_id"]
                geo = boxes_iou([g["bbox"]], boxes)[0] >= .5
                cand = geo & (scores[:, 19] > .001)
                first_mask = np.zeros(len(geo), bool); first_mask[first] = True
                final_cow = [j for j in range(len(group)) if group[j]["category_id"] == 21
                             and geo[ids[j]]]
                flags = [bool(geo.any()), bool(cand.any()), bool((cand & first_mask).any()),
                         bool(final_cow), any(group[j]["score"] > .25 for j in final_cow)]
                j = matches[tid]
                correct = j is not None and group[j]["category_id"] == 21
                flags.append(correct)
                assert all(not flags[k] or flags[k-1] for k in range(1, 5)), ("non_nested", iid, tid)
                old = inp.timeline[iid, tid]
                assert correct == old["class_correct"]
                if j is not None:
                    assert inp.pred[iid][j]["output_detection_id"] == old["output_detection_id"]
                failure = next((k+1 for k, value in enumerate(flags) if not value), None)
                best = int(np.where(geo)[0][scores[geo, 19].argmax()]) if geo.any() else None
                value = dict(image_id=iid, track_id=tid, seed=inp.frames[iid]["seed"],
                    variant=inp.frames[iid]["variant"], stages=flags, first_failure=failure,
                    matched_rank=j, matched_class=None if j is None else group[j]["category_id"],
                    best_geometric_raw=best,
                    best_cow_score=None if best is None else float(scores[best, 19]),
                    geometric_raw_count=int(geo.sum()), correct_class=correct,
                    source_output_id=None if j is None else inp.pred[iid][j]["output_detection_id"])
                append(run / "CHAIN_ROWS.jsonl", value); stage_rows.append(value)
            cc = cpu_cache(c)
            caches_by_id[iid] = cc
            cachepath = run / "cache" / f"{iid:012d}.pt"
            cachepath.parent.mkdir(exist_ok=True)
            torch.save(cc, cachepath)
            replay_count += len(group)
        save(run / "PROGRESS.json", dict(stage="native_trace", frames=len(caches_by_id), replay_rows=replay_count))
        print("TRACE", bi, len(caches_by_id), flush=True)
        held.clear()
        del caches, rows, features, raw, raw_boxes, raw_scores
        resource_check()
    hook.remove()
    pair_rows = []
    for seed_index in range(3):
        for control_variant, pressure_variant in [(0, 1), (2, 3)]:
            for frame in range(48):
                a = (seed_index*4+control_variant)*1000+frame+1
                b = (seed_index*4+pressure_variant)*1000+frame+1
                if a not in caches_by_id or b not in caches_by_id: continue
                ta, tb = inp.timeline[a, 1], inp.timeline[b, 1]
                value = dict(control_image=a, pressure_image=b, seed=20261020+seed_index,
                             contrast="near_clear" if pressure_variant == 1 else "front_rear",
                             eligible=ta["class_correct"] and tb["class_correct"])
                if not value["eligible"]:
                    append(run / "FACTORIAL_ROWS.jsonl", value); pair_rows.append(value); continue
                ca, cb = caches_by_id[a], caches_by_id[b]
                ja, jb = inp.id_to_rank[ta["output_detection_id"]], inp.id_to_rank[tb["output_detection_id"]]
                value.update(control_output=ta["output_detection_id"], pressure_output=tb["output_detection_id"])
                label_a, label_b = inp.labels(a), inp.labels(b)
                context = Context(label_b == 1, [(i, label_b == i) for i in [2, 3]])
                common_gt = (label_a == 1) & (label_b == 1)
                assert common_gt.any()
                results = {}
                for box_name, box in [("pressure", cb["boxes_input"][jb]), ("control", ca["boxes_input"][ja])]:
                    results[box_name] = {}
                    for p_idx, p in enumerate([ca["proto"], cb["proto"]]):
                        for c_idx, coeff in enumerate([ca["c0"][ja], cb["c0"][jb]]):
                            hybrid = {**cb, "proto": p, "boxes_input": cb["boxes_input"].clone()}
                            hybrid["boxes_input"][jb] = box
                            matrix = cb["c0"].clone(); matrix[jb] = coeff
                            decoded = decode_native(hybrid, matrix)[jb].cpu().numpy().astype(bool)
                            key = f"P{p_idx}c{c_idx}"
                            results[box_name][key] = context.measure(decoded)
                            append(run / "HYBRID_MASKS.jsonl", dict(control_image=a, pressure_image=b,
                                   box=box_name, combination=key, segmentation=rle(decoded)))
                # Affine map to original coordinates makes the bilinear decomposition exact
                # up to frozen FP32 rounding; this is not the nonlinear native binary mask.
                p0, p1 = ca["proto"].cuda(), cb["proto"].cuda()
                c0, c1 = ca["c0"][ja:ja+1].cuda(), cb["c0"][jb:jb+1].cuda()
                l0 = continuous(cb, c0, p0)[0].cpu().numpy()
                l1 = continuous(cb, c1, p1)[0].cpu().numpy()
                terms = [continuous(cb, c0, p1-p0)[0].cpu().numpy(),
                         continuous(cb, c1-c0, p0)[0].cpu().numpy(),
                         continuous(cb, c1-c0, p1-p0)[0].cpu().numpy()]
                error = float(np.abs(l1-l0-sum(terms)).max())
                assert error <= 1e-4, ("factorial_identity", a, b, error)
                value.update(results=results, common_gt_pixels=int(common_gt.sum()),
                    logit_decomposition_max_abs_error=error,
                    logit_term_rms={k:float(np.sqrt(np.mean(t[common_gt].astype(float)**2)))
                        for k,t in zip(["P", "c", "interaction"], terms)},
                    logit_total_rms=float(np.sqrt(np.mean((l1-l0)[common_gt].astype(float)**2))))
                append(run / "FACTORIAL_ROWS.jsonl", value); pair_rows.append(value)
        print("FACTORIAL_SEED", seed_index, len(pair_rows), flush=True)
    assert len(stage_rows) == (192 if smoke else 1728)
    save(run / "ENGINEERING.json", dict(status="passed", frames=len(caches_by_id),
         replay_rows=replay_count, exact_base_identity=True, raw_gather_exact=True,
         native_args_exact=True, training_steps=0, resources=resource_check()))
