"""Small checks for measurement semantics, not an experiment."""
import json
from pathlib import Path
from evaluate import associate
from topology import tests

root=Path(__file__).resolve().parents[1]
gt=[{"bbox":[0,0,10,10]},{"bbox":[20,0,30,10]}]
pred=[{"bbox":[20,0,10,10],"category_id":99},{"bbox":[0,0,10,10],"category_id":1}]
assert associate(gt,pred)=={0:1,1:0}
assert associate(gt,[])=={} and associate([],pred)=={}
assert associate(gt,[{"bbox":[100,100,10,10]}])=={}
scenes=json.loads((root/"SCENES.json").read_text())
assert len(scenes)==8 and sum(s["frames"] for s in scenes)==384
for a,b in zip(scenes[::2],scenes[1::2]):
    assert a["scenario"]==b["scenario"] and a["seed"]==b["seed"] and a["asset"]==b["asset"]
    assert a["id"]//2==b["id"]//2
result={"status":"passed","topology":tests(),"association":"class-independent one-to-one geometry",
        "scene_pairs":4,"training_steps":0,"gpu_operations":0}
(root/"ENGINEERING_TESTS.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
