"""Small SSH transport; password comes from the process environment only."""
import argparse, os
from pathlib import Path
import paramiko

p = argparse.ArgumentParser()
p.add_argument('--command-file', type=Path)
p.add_argument('--get', nargs=2, action='append', default=[])
p.add_argument('--put', nargs=2, action='append', default=[])
a=p.parse_args()
c=paramiko.SSHClient(); c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.WarningPolicy())
c.connect('connect.cqa1.seetacloud.com', port=26822, username='root', password=os.environ['PAPER_SSH_PASSWORD'], timeout=20, banner_timeout=30)
if a.command_file:
    stdin, stdout, stderr = c.exec_command(a.command_file.read_text(encoding='utf-8'), timeout=180)
    print(stdout.read().decode('utf-8',errors='replace'))
    print(stderr.read().decode('utf-8',errors='replace'))
    status=stdout.channel.recv_exit_status()
    if status: raise SystemExit(status)
if a.get or a.put:
    s=c.open_sftp()
    for remote,local in a.get:
        Path(local).parent.mkdir(parents=True,exist_ok=True);s.get(remote,local);print('downloaded',Path(local).name)
    for local,remote in a.put:
        s.put(local,remote);print('uploaded',Path(local).name)
    s.close()
c.close()
