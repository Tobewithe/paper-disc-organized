"""Interactive read-only SFTP inspector. Password stays in memory."""
import argparse
import getpass
import hashlib
import json
import stat
from pathlib import Path
import paramiko

p=argparse.ArgumentParser()
p.add_argument("--host",required=True)
p.add_argument("--port",type=int,required=True)
p.add_argument("--user",default="root")
a=p.parse_args()
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.WarningPolicy())
password=getpass.getpass("SSH password: ")
client.connect(a.host,port=a.port,username=a.user,password=password,
               timeout=20,banner_timeout=20,auth_timeout=20,allow_agent=False,look_for_keys=False)
password=None
sftp=client.open_sftp()
print(json.dumps({"connected":True,"host":a.host,"port":a.port,
                  "host_key_sha256":hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()}),flush=True)
try:
    while True:
        try:
            raw=input("READ_ONLY_REQUEST> ")
        except EOFError:
            break
        try:
            request=json.loads(raw)
            action=request["action"]
            if action=="close":
                break
            if action=="list":
                result={}
                for path in request["paths"]:
                    try:
                        entries=sftp.listdir_attr(path)
                        result[path]={"count":len(entries),"entries":[{"name":e.filename,"bytes":e.st_size,
                                      "directory":stat.S_ISDIR(e.st_mode)} for e in entries[:request.get("limit",50)]]}
                    except OSError as error:
                        result[path]={"error":type(error).__name__}
                print(json.dumps(result,ensure_ascii=False),flush=True)
            elif action=="read":
                with sftp.open(request["path"],"rb") as f:
                    value=f.read(request.get("limit",10000))
                print(json.dumps({"path":request["path"],"text":value.decode("utf-8",errors="replace")},ensure_ascii=False),flush=True)
            elif action=="download":
                dest=Path(request["local"])
                if dest.exists():
                    raise FileExistsError("Refusing to overwrite local file")
                dest.parent.mkdir(parents=True,exist_ok=True)
                sftp.get(request["remote"],str(dest))
                print(json.dumps({"downloaded":str(dest),"bytes":dest.stat().st_size,
                                  "sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}),flush=True)
            else:
                raise ValueError("Only list/read/download/close allowed")
        except Exception as error:
            print(json.dumps({"error":type(error).__name__,"detail":str(error)}),flush=True)
finally:
    sftp.close()
    client.close()
