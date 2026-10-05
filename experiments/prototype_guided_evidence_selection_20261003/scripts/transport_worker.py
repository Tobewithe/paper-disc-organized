"""Key-pinned SSH/SFTP worker; no model imports or global network changes."""
import json
import os
from pathlib import Path
import socket
import sys
import paramiko

request = json.load(sys.stdin)
host, port = 'connect.bjb2.seetacloud.com', 33953
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(20)
if os.name == 'nt':
    # Windows IP_DONTFRAGMENT=14. Default DF=1 caused reproducible large-write
    # resets on this route; clearing DF on this socket passed the same transfer.
    # This does not change any adapter, routing table or another application's socket.
    sock.setsockopt(socket.IPPROTO_IP, 14, 0)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
sock.connect((host,port))
client = paramiko.SSHClient()
client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(host,port=port,username='root',key_filename=request['key'],sock=sock,
               look_for_keys=False,allow_agent=False,timeout=20,auth_timeout=20,banner_timeout=20)
client.get_transport().set_keepalive(5)
try:
    action=request['action']
    if action=='ssh':
        stdin,stdout,stderr=client.exec_command(request['command'],timeout=request['timeout'])
        stdin.close()
        output=stdout.read().decode('utf-8',errors='replace')
        error=stderr.read().decode('utf-8',errors='replace')
        result=dict(returncode=stdout.channel.recv_exit_status(),stdout=output,stderr=error)
    else:
        sftp=client.open_sftp();sftp.get_channel().settimeout(60)
        if action=='put':
            for source in request['paths']:
                destination=request['destination']
                if destination.endswith('/'):
                    destination+=Path(source).name
                sftp.put(source,destination,confirm=True)
        elif action=='get':
            sftp.get(request['source'],request['destination'])
        else:
            raise ValueError(action)
        sftp.close();result=dict(returncode=0,stdout='',stderr='')
    print(json.dumps(result,ensure_ascii=False),flush=True)
finally:
    client.close()
