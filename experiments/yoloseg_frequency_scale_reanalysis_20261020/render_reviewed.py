"""Presentation-only repair: move labels clear of the decomposition legend."""
import ast, json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
run=root/"runs"/json.loads((root/"LATEST_RUN.json").read_text())["run_id"]
source=(root/"scripts/analyze.py").read_text(encoding="utf-8")
tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="render")
text=ast.get_source_segment(source,fn)
text=text.replace("ylim=(0,118)", "ylim=(0,130)")
text=text.replace('axes[0].text(x,104,f"{y:.0f}%",ha="center",fontsize=9)',
                  'axes[0].text(x,96,f"{y:.0f}%",ha="center",va="top",fontsize=9,color="white")')
ns={"np":np,"BANDS":["low","mid","high"]}
exec(text,ns)
out=root/"figures"
out.mkdir(exist_ok=True)
ns["render"](json.loads((run/"SUMMARY.json").read_text()),out)
print(out)
