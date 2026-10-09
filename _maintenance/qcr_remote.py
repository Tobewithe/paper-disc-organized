"""SSH/SFTP transport for the authorized QCR server; credentials stay in env."""
import json
import os
from pathlib import Path
import sys

import paramiko

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
request = json.load(sys.stdin)
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(os.environ['SSH_HOST'], port=int(os.environ['SSH_PORT']),
               username=os.environ['SSH_USER'], password=os.environ['SSH_PASSWORD'],
               timeout=20, auth_timeout=20, banner_timeout=20)
try:
    action = request['action']
    if action == 'exec':
        _, stdout, stderr = client.exec_command(request['command'], timeout=request.get('timeout', 60))
        out = stdout.read().decode('utf-8', 'replace')
        err = stderr.read().decode('utf-8', 'replace')
        print(json.dumps({'returncode': stdout.channel.recv_exit_status(), 'stdout': out, 'stderr': err}, ensure_ascii=False))
    elif action in ('get', 'put'):
        with client.open_sftp() as sftp:
            receipts = []
            files = request['files']
            if len(files) == 2 and all(isinstance(item, str) for item in files):
                files = [files]
            for local, remote in files:
                if action == 'get':
                    Path(local).parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote, local)
                else:
                    sftp.put(local, remote)
                receipts.append({'local': local, 'remote': remote})
            print(json.dumps({'files': receipts}, ensure_ascii=False))
    else:
        raise ValueError(action)
finally:
    client.close()
