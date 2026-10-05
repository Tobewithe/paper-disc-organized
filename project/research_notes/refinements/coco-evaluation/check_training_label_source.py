"""Read-only SSH inspection using the already configured project connection.

Only literal connection fields are read by AST; the source script is never run.
Credentials are not emitted or persisted in the audit.
"""
import ast
import json
from pathlib import Path
import paramiko

root = Path(__file__).resolve().parents[2]
tree = ast.parse((root/"gemini/scripts/download_data_logs.py").read_text(encoding="utf-8-sig"))
connect = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == "connect")
args = [ast.literal_eval(n) for n in connect.args]
kwargs = {k.arg: ast.literal_eval(k.value) for k in connect.keywords}
kwargs.update(timeout=12, banner_timeout=12, auth_timeout=12)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.WarningPolicy())
out = {"connection_source": "gemini/scripts/download_data_logs.py", "mode": "read-only"}
try:
    client.connect(*args, **kwargs)
    sftp = client.open_sftp()
    out["connected"] = True
    paths = ["/root/autodl-tmp/datasets/coco",
             "/root/autodl-tmp/datasets/coco/labels",
             "/root/autodl-tmp/datasets/coco/labels/train2017",
             "/root/autodl-tmp/datasets/coco/labels/val2017"]
    out["paths"] = {}
    for path in paths:
        try:
            entries = sftp.listdir_attr(path)
            out["paths"][path] = {"entry_count": len(entries),
                                 "examples": [{"name": e.filename, "bytes": e.st_size} for e in entries[:8]]}
        except OSError as error:
            out["paths"][path] = {"error_type": type(error).__name__}
    sftp.close()
except Exception as error:
    out["connected"] = False
    out["error_type"] = type(error).__name__
finally:
    client.close()
dest=root/"refine-logs/coco-evaluation/TRAINING_LABEL_CONNECTION_CHECK_20260911.json"
dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
