import json,sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import generate as g
scenes=json.loads((g.R/'scenes.json').read_text())
for cfg,frame in [(scenes[33],3),(scenes[51],31)]:
 s,roots,assets,devices=g.setup(cfg);g.pose(s,roots,cfg,frame);rays,focal,origin=g.raygrid(s)
 try:
  g.geometry(s,rays,focal,origin,frame);print('DIAG_PASSED',cfg['id'],frame,flush=True)
 except AssertionError as e:print('DIAG_ERROR',cfg['id'],frame,e,flush=True)
