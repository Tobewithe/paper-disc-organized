"""Integer phase intervention, original batch layout, exact geometric null controls."""
import gc
from common import *

OFFSETS = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(4,0),(-4,0),(0,4),(0,-4)]


@torch.inference_mode()
def execute(run, smoke):
    inp = Inputs()
    indices = [0,3,6,9] if smoke else list(range(36))
    baseline = {}
    total = replay_rows = 0
    for dx,dy in OFFSETS:
        prep = inp.prepared(run, (dx,dy))
        stream = NativeBatches(inp.cfg, prep, "val", run / f"native_{dx}_{dy}")
        assert read(run / f"native_{dx}_{dy}/NATIVE_ARGS.json") == read(inp.base/"native/NATIVE_ARGS.json")
        save(run / f"ENVIRONMENT_{dx}_{dy}.json", stream.environment)
        for bi in indices:
            caches, rows, features = stream.batch(bi, masks=True)
            if (dx,dy)==(0,0):
                append(run / "ZERO_IDENTITY.jsonl", inp.check_base(bi,caches,rows,features))
            else:
                append(run / "SHIFT_IDENTITY.jsonl", dict(offset=[dx,dy],batch=bi,identity=batch_identity(caches,rows,features)))
            for c,group in zip(caches,rows):
                iid=c["image_id"]; label=inp.labels(iid)
                h,w=label.shape; interior=np.zeros((h,w),bool);interior[4:-4,4:-4]=True
                matches=inp.matches(iid,group,(dx,dy))
                masks=decode_native(c).cpu().numpy().astype(bool)
                logits=continuous(c).cpu().numpy()
                for g in inp.frames[iid]["records"]:
                    tid=g["track_id"]; bbox=g["bbox"]
                    eligible=bbox[0]>=8 and bbox[1]>=8 and bbox[2]<=w-8 and bbox[3]<=h-8
                    j=matches[tid]
                    correct=j is not None and group[j]["category_id"]==21
                    gt=(label==tid)&interior
                    pred=translate(masks[j],-dx,-dy).astype(bool)&interior if correct else np.zeros_like(gt)
                    metric=Context(gt,[(t,(label==t)&interior) for t in [1,2,3] if t!=tid]).measure(pred)
                    op_iou=metric["iou"] if correct else 0.
                    row=dict(image_id=iid,track_id=tid,seed=inp.frames[iid]["seed"],
                        variant=inp.frames[iid]["variant"],offset=[dx,dy],eligible=eligible,
                        correct=correct,operational_iou=op_iou,success=correct and op_iou>=.75,
                        matched_rank=j,raw_id=None if j is None else c["raw_ids"][j],
                        raw_location=None if j is None else c["raw_locations"][j],
                        metrics=metric)
                    x0,y0,x1,y1=bbox
                    if (dx,dy)==(0,0):
                        old=inp.timeline[iid,tid]; assert correct==old["class_correct"]
                        if correct:
                            assert abs(iou(masks[j],label==tid)-old["operational_iou"])<=1e-12
                        baseline[iid,tid]=dict(row=row,mask=pred,logits=None if not correct else logits[j,y0:y1,x0:x1].copy(),
                            box=None if j is None else c["boxes_input"][j].cpu().clone())
                        row["source_output_id"]=old.get("output_detection_id")
                    else:
                        base=baseline[iid,tid]; br=base["row"]
                        row.update(base_correct=br["correct"],base_success=br["success"],
                            base_operational_iou=br["operational_iou"],
                            detection_flip=correct!=br["correct"], success_flip=row["success"]!=br["success"],
                            either_flip=correct!=br["correct"] or row["success"]!=br["success"],
                            delta_iou=op_iou-br["operational_iou"])
                        shifted_null=translate(base["mask"],dx,dy)
                        assert np.array_equal(translate(shifted_null,-dx,-dy)[4:-4,4:-4],base["mask"][4:-4,4:-4])
                        if correct and br["correct"]:
                            aligned=translate(logits[j],-dx,-dy)[y0:y1,x0:x1]
                            oldlogit=base["logits"]
                            rms=float(np.sqrt(np.mean((aligned.astype(float)-oldlogit.astype(float))**2)))
                            relative=rms/max(1e-12,float(np.sqrt(np.mean(oldlogit.astype(float)**2))))
                            fixed={**c,"boxes_input":c["boxes_input"].clone()}
                            fixed["boxes_input"][j]=base["box"].cuda()+torch.tensor([dx,dy,dx,dy],device="cuda")
                            pm=decode_native(fixed)[j].cpu().numpy().astype(bool)
                            pm=translate(pm,-dx,-dy)&interior
                            row.update(mask_consistency_iou=iou(pred,base["mask"]),
                                fixed_box_consistency_iou=iou(pm,base["mask"]),
                                logit_rms=rms,logit_relative_rms=relative,
                                logit_sign_flip_fraction=float(np.mean((aligned>0)!=(oldlogit>0))),
                                bridge_flip=metric["material_bridge"]!=br["metrics"]["material_bridge"],
                                split_flip=metric["material_split"]!=br["metrics"]["material_split"])
                        append(run/"ALIGNED_MASKS.jsonl",dict(image_id=iid,track_id=tid,offset=[dx,dy],segmentation=rle(pred)))
                    append(run/"SHIFT_ROWS.jsonl",row)
                total+=1; replay_rows+=len(group)
            save(run/"PROGRESS.json",dict(offset=[dx,dy],frames=total,replay_rows=replay_rows))
            print("SHIFT",dx,dy,bi,total,flush=True)
            del caches,rows,features,masks,logits
            resource_check()
        del stream
        gc.collect();torch.cuda.empty_cache()
    assert total==(576 if smoke else 5184)
    save(run/"ENGINEERING.json",dict(status="passed",frames=total,offsets=OFFSETS,
        zero_base_exact=True,geometric_null_exact=True,native_args_exact=True,
        training_steps=0,replay_rows=replay_rows,resources=resource_check()))
