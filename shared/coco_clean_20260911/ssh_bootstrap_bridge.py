"""Interactive SSH operations; credentials are never persisted."""
import argparse
import getpass
import hashlib
import json
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
client.connect(a.host,port=a.port,username=a.user,password=password,timeout=20,
               banner_timeout=20,auth_timeout=20,allow_agent=False,look_for_keys=False)
password=None
sftp=client.open_sftp()
print("SSH_CONNECTED",flush=True)
try:
    while True:
        try:
            r=json.loads(input("REMOTE_REQUEST> "))
        except EOFError:
            break
        try:
            if r["action"]=="close":
                break
            if r["action"]=="exec":
                stdin,stdout,stderr=client.exec_command(r["command"],timeout=r.get("timeout",55))
                stdin.close()
                output=stdout.read().decode("utf-8",errors="replace")
                error=stderr.read().decode("utf-8",errors="replace")
                result={"exit_code":stdout.channel.recv_exit_status(),"stdout":output,"stderr":error}
            elif r["action"]=="put":
                sftp.put(r["local"],r["remote"])
                result={"uploaded":r["remote"],"local_sha256":hashlib.sha256(Path(r["local"]).read_bytes()).hexdigest()}
            elif r["action"]=="get":
                dest=Path(r["local"])
                dest.parent.mkdir(parents=True,exist_ok=True)
                sftp.get(r["remote"],str(dest))
                result={"downloaded":str(dest),"sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}
            elif r["action"]=="host_key":
                k=client.get_transport().get_remote_server_key()
                result={"known_hosts_line":f"[{a.host}]:{a.port} {k.get_name()} {k.get_base64()}"}
            else:
                raise ValueError("Unknown action")
            print(json.dumps(result,ensure_ascii=False),flush=True)
        except Exception as error:
            print(json.dumps({"error":type(error).__name__,"detail":str(error)}),flush=True)
finally:
    sftp.close()
    client.close()
