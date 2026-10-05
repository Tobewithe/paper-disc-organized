import os
import subprocess
import sys

def install_deps():
    try:
        import paramiko
        from scp import SCPClient
    except ImportError:
        print("Installing paramiko and scp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko", "scp"])

install_deps()

import paramiko
from scp import SCPClient

def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def main():
    server = 'connect.westd.seetacloud.com'
    port = 17381
    user = 'root'
    password = '/eIOAILi96O8'
    local_zip = 'C:/Dpan/codexproject/paper-disc/coco_ccl_scripts.zip'
    remote_path = '/root/autodl-tmp/'

    print(f"Connecting to {server}:{port}...")
    ssh = create_ssh_client(server, port, user, password)
    
    print("Uploading zip file...")
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(local_zip, remote_path)
        
    print("Executing remote setup commands...")
    commands = [
        "cd /root/autodl-tmp && unzip -o coco_ccl_scripts.zip",
        # AutoDL base images usually have an academic mirror, but let's install ultralytics quietly
        "pip install ultralytics > /root/autodl-tmp/install.log 2>&1",
        "chmod +x /root/autodl-tmp/tools/run_coco_cluster.sh",
        # We start the cluster run in the background using nohup
        "cd /root/autodl-tmp && nohup bash tools/run_coco_cluster.sh > run_cluster.log 2>&1 &"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        exit_status = stdout.channel.recv_exit_status()
        print(f"Exit status: {exit_status}")
        
    print("Deployment and background task launched successfully on AutoDL!")
    ssh.close()

if __name__ == '__main__':
    main()
