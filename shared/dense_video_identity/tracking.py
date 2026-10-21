"""Causal memory policies and bounded tracklet recovery on a fixed tracker backend."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import iou_matrix, unit


def quality_vector(memory, feature, quality, others, length, gap):
    similarity = float(np.dot(memory, feature))
    competitor = max((float(np.dot(m, feature)) for m in others), default=0.0)
    q = np.asarray(quality, np.float32)
    return np.asarray([q[0], q[1], q[2], np.sqrt(max(q[3], 0)), similarity,
                       similarity - competitor, min(np.log1p(length) / 6, 1), min(gap / 60, 1)], np.float32)


def make_tracker(arm, config, utility=None):
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.deep_oc_sort import DeepOCSORT, DeepOCSortTrack
    from ultralytics.trackers.utils.stracks import parse_bboxes
    args = SimpleNamespace(**deepcopy(config))
    if arm == "bytetrack":
        args.with_reid = False
        return BYTETracker(args)
    if arm == "deepocsort":
        return DeepOCSORT(args)

    class ProtectedTrack(DeepOCSortTrack):
        def __init__(self, *a, quality=None, owner=None, **kw):
            self.write_enabled = True
            self.quality = quality
            self.owner = owner
            self.update_record = None
            super().__init__(*a, **kw)

        def update_features(self, feat, score=None):
            if self.write_enabled or self.smooth_feat is None:
                super().update_features(feat, score)
            else:
                self.curr_feat = unit(feat)

        def decide(self, new, frame):
            if self.smooth_feat is None or new.curr_feat is None:
                self.write_enabled = True
                return
            others = [m for tid,m in self.owner.memory_start.items() if tid != self.track_id]
            x = quality_vector(self.smooth_feat, new.curr_feat, new.quality, others,
                               frame-self.start_frame, frame-self.end_frame)
            pred = None
            if arm in {"utility_memory", "utility_recovery"}:
                assert utility is not None
                pred = float(utility(x))
                self.write_enabled = pred > 0
            else:
                self.write_enabled = bool(new.quality[1] <= .35 and new.quality[2] <= .25)
            self.update_record = dict(frame=frame, track=int(self.track_id), write=bool(self.write_enabled),
                                      expected_benefit=pred, overlap=float(new.quality[1]))
            self.owner.policy_events.append(self.update_record)

        def update(self, new_track, frame_id):
            self.decide(new_track, frame_id)
            super().update(new_track, frame_id)
            self.quality = new_track.quality

        def re_activate(self, new_track, frame_id, new_id=False):
            self.decide(new_track, frame_id)
            super().re_activate(new_track, frame_id, new_id)
            self.quality = new_track.quality

    class ProtectedTracker(DeepOCSORT):
        def __init__(self):
            super().__init__(args)
            self.policy_events = []
            self.memory_start = {}

        def update(self, results, img=None, feats=None, **kw):
            self.memory_start = {t.track_id:t.smooth_feat.copy() for t in self.tracked_stracks+self.lost_stracks if t.smooth_feat is not None}
            return super().update(results, img=img, feats=feats, **kw)

        def init_track(self, results, img=None):
            if not len(results):
                return []
            assert img is not None and img.shape[1] == 68, "64D embeddings + 4 visible-quality inputs"
            return [ProtectedTrack(xywh, s, c, self.delta_t, feat=f[:64], quality=f[64:], owner=self,
                                   alpha_fixed_emb=self.alpha_fixed_emb, det_thresh=self.args.track_high_thresh)
                    for xywh,s,c,f in zip(parse_bboxes(results), results.conf, results.cls, img)]
    return ProtectedTracker()


class BoundedRecovery:
    """Join reappearance tracklets without future access or rewriting earlier output.

    The native tracker creates short tracklets. An old identity remains eligible only
    while absent, spatially plausible and within the same 60-frame budget. Joint
    Hungarian assignment is accepted only after row AND column competitor margins.
    Uncertain observations remain provisional and count as omissions in strict MOT.
    """
    def __init__(self, config, buffer_frames=60):
        self.cfg, self.buffer = config, buffer_frames
        self.mapping, self.states, self.pending = {}, {}, {}
        self.next_id = 1
        self.events = []

    def _new_id(self):
        identity = self.next_id
        self.next_id += 1
        return identity

    @staticmethod
    def distance(a,b):
        ca,cb=(a[:2]+a[2:])/2,(b[:2]+b[2:])/2
        scale=max(np.linalg.norm(a[2:]-a[:2]), np.linalg.norm(b[2:]-b[:2]),1)
        return float(np.linalg.norm(ca-cb)/scale)

    def update(self, frame, observations):
        present = {o["base_id"] for o in observations}
        # Deadlines advance on every video frame, including frames with no observation.
        # Otherwise a disappeared provisional track could wait arbitrarily long.
        for bid,p in list(self.pending.items()):
            if bid not in present and frame-p["first"] >= self.cfg["max_delay_frames"]:
                self.mapping[bid]=self._new_id()
                self.events.append(dict(frame=frame,type="expired_unobserved",identity=self.mapping[bid],base_id=bid,
                                        delay_frames=frame-p["first"]))
                del self.pending[bid]
        used = {self.mapping[b] for b in present if b in self.mapping}
        eligible = {i:s for i,s in self.states.items() if i not in used and 1 < frame-s["frame"] <= self.buffer}
        result = []
        for o in observations:
            bid = o["base_id"]
            if bid in self.mapping:
                continue
            if bid not in self.pending:
                possible = [i for i,s in eligible.items() if self.distance(o["box"],s["box"]) <= self.cfg["max_normalized_distance"]]
                if not possible:
                    self.mapping[bid] = self._new_id()
                    continue
                self.pending[bid] = dict(first=frame, count=0, features=[], candidates=possible)
            p = self.pending[bid]
            p["count"] += 1
            p["features"].append(o["feature"].copy())
            p["features"] = p["features"][-self.cfg["max_delay_frames"]:]
            p["box"] = o["box"]
        candidates = [(bid,p) for bid,p in self.pending.items() if bid in present and p["count"] >= self.cfg["min_tracklet_frames"]]
        ids = sorted(eligible)
        if candidates and ids:
            sim = np.full((len(candidates),len(ids)),-2.,np.float32)
            for r,(bid,p) in enumerate(candidates):
                f = unit(np.mean(p["features"],axis=0))
                for c,identity in enumerate(ids):
                    if identity in p["candidates"]:
                        dist = self.distance(p["box"],eligible[identity]["box"])
                        if dist <= self.cfg["max_normalized_distance"]:
                            sim[r,c] = float(f@eligible[identity]["memory"]) - .05*dist
            rr,cc=linear_sum_assignment(-sim)
            for r,c in zip(rr,cc):
                row_other=np.delete(sim[r],c);col_other=np.delete(sim[:,c],r)
                margin=float(sim[r,c]-max(row_other.max(initial=-1),col_other.max(initial=-1)))
                if sim[r,c] >= self.cfg["min_similarity"] and margin >= self.cfg["min_margin"]:
                    bid,p=candidates[r];identity=ids[c]
                    self.mapping[bid]=identity
                    self.events.append(dict(frame=frame, type="recovered", identity=identity, base_id=bid,
                                            delay_frames=frame-p["first"], similarity=float(sim[r,c]), margin=margin))
                    del self.pending[bid]
        for o in observations:
            bid=o["base_id"]
            status="tracked"
            if bid not in self.mapping:
                p=self.pending[bid]
                if frame-p["first"] >= self.cfg["max_delay_frames"]:
                    self.mapping[bid]=self._new_id()
                    self.events.append(dict(frame=frame,type="abstained_new_id",identity=self.mapping[bid],base_id=bid,delay_frames=frame-p["first"]))
                    del self.pending[bid]
                else:
                    result.append(dict(o, identity=None,status="pending"))
                    continue
            identity=self.mapping[bid]
            if any(e["frame"]==frame and e["base_id"]==bid and e["type"]=="recovered" for e in self.events[-len(observations):]):
                status="recovered"
            # One canonical identity can never be emitted twice in the same frame.
            if any(x["identity"]==identity for x in result):
                identity=self._new_id();self.mapping[bid]=identity;status="collision_new_id"
            self.states[identity]=dict(frame=frame,box=o["box"].copy(),memory=o["memory"].copy())
            result.append(dict(o,identity=identity,status=status))
        self.states={i:s for i,s in self.states.items() if frame-s["frame"] <= self.buffer}
        self.pending={i:p for i,p in self.pending.items() if frame-p["first"] <= self.buffer}
        return result


def frame_input(record, embeddings, protected=False):
    from ultralytics.engine.results import Boxes
    boxes=np.asarray(record["boxes"],np.float32).reshape(-1,4)
    n=len(boxes)
    data=np.column_stack([boxes,np.asarray(record["scores"],np.float32),np.asarray(record["classes"],np.float32)]) if n else np.empty((0,6),np.float32)
    start=record["feature_start"]
    feat=embeddings[start:start+n]
    if protected:
        feat=np.column_stack([feat,np.asarray(record["quality"],np.float32).reshape(-1,4)])
    return Boxes(data,orig_shape=(1,1)),feat


def run_sequence(arm, records, embeddings, config, utility=None, on_before=None, on_after=None):
    tracker=make_tracker(arm,config["tracker"],utility)
    resolver=BoundedRecovery(config["recovery"],config["tracker"]["track_buffer"]) if arm.endswith("recovery") else None
    outputs=[]
    protected=arm not in {"bytetrack","deepocsort"}
    for rec in records:
        frame=rec["frame"]
        if on_before:
            on_before(frame,tracker)
        boxes,feats=frame_input(rec,embeddings,protected)
        rows=tracker.update(boxes,img=None,feats=feats)
        tracks={t.track_id:t for t in tracker.tracked_stracks}
        observations=[]
        for row in rows:
            bid,idx=int(row[4]),int(row[-1])
            t=tracks[bid]
            start=rec["feature_start"]+idx
            feature=embeddings[start]
            memory=getattr(t,"smooth_feat",None)
            observations.append(dict(base_id=bid,det_id=idx,box=np.asarray(rec["boxes"][idx],np.float32),
                                     feature=feature,memory=memory.copy() if memory is not None else feature.copy(),
                                     score=float(rec["scores"][idx]),write=getattr(t,"write_enabled",True)))
        if on_after:
            on_after(frame,tracker,observations,rec)
        assigned=resolver.update(frame,observations) if resolver else [dict(o,identity=o["base_id"],status="tracked") for o in observations]
        outputs.append(dict(frame=frame, tracks=[{k:v for k,v in o.items() if k not in {"feature","memory"}} for o in assigned]))
    return outputs,dict(memory_events=getattr(tracker,"policy_events",[]),recovery_events=resolver.events if resolver else [])
