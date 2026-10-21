from __future__ import annotations
from pathlib import Path
import unittest
import numpy as np
from .common import Run, read_json, setup_paths, sha256, write_json
from .evaluate import official_metrics, metric_data, compact
from .tracking import run_sequence


def audit(root:Path,config):
    setup_paths(root)
    manifest=read_json(root/"input_manifest.json")
    with Run(root,"input_and_implementation_audit",dict(scope="input hashes, causal invariants and metric identity check; no model fitting")) as run:
        checks=[]
        for s in manifest["sequences"]:
            assert sha256(root/s["video"])==s["video_sha256"],s["name"]
            if s["gt"]:
                assert sha256(root/s["gt"])==s["gt_sha256"],s["name"]
                a=np.loadtxt(root/s["gt"],delimiter=",",ndmin=2)
                assert a[:,0].min()>=1 and a[:,0].max()<=s["frames"]
                assert (a[:,4:6]>0).all()
                assert len(set(map(tuple,a[:,:2])))==len(a),"Duplicate (frame, GT identity)"
            checks.append(s["name"])
        for kind,p in manifest["models"].items():
            assert sha256(root/p)==manifest["model_sha256"][kind]
        import shared.dense_video_identity.test_invariants as tests
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        assert result.wasSuccessful()
        g={1:np.asarray([[1,7,10,10,30,30,1,8,-1]],float),2:np.asarray([[2,7,11,10,31,30,1,8,-1]],float)}
        pred=[dict(frame=i,tracks=[dict(identity=55,box=g[i][0,2:6])]) for i in (1,2)]
        ms=official_metrics(root);d=metric_data(g,pred);s=compact({k:m.eval_sequence(d) for k,m in ms.items()})
        assert s["IDF1"]==100 and abs(s["HOTA"]-100)<1e-8 and s["IDSW"]==0,s
        switched=[pred[0],dict(frame=2,tracks=[dict(identity=56,box=g[2][0,2:6])])]
        d2=metric_data(g,switched);bad=compact({k:m.eval_sequence(d2) for k,m in ms.items()})
        assert bad["IDF1"]<100 and bad["IDSW"]==1,bad
        records=[dict(frame=i,boxes=[[10+i,10,40+i,40]],scores=[.95],classes=[0],quality=[[.95,.1,.0,.1]],feature_start=i-1,feature_count=1) for i in range(1,6)]
        z=np.tile(np.eye(1,64,dtype=np.float32),(5,1))
        for arm in config["arms"]:
            out,ev=run_sequence(arm,records,z,config,lambda _: -1.)
            assert len(out)==5 and all(len(f["tracks"])==1 for f in out),arm
            if arm.startswith("utility"):
                assert all(not e["write"] for e in ev["memory_events"])
        write_json(run.path/"SUMMARY.json",dict(input_hashes="passed",sequences=len(checks),metric_perfect=s,metric_switch=bad,
                   causal_tests=result.testsRun,tracker_arms="all six passed",status="engineering_smoke_passed"))
    return run.id
