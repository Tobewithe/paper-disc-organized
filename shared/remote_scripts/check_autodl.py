import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    # Use bash -lc to execute commands as a login shell so conda is loaded
    stdin, stdout, stderr = client.exec_command("bash -lc 'which python; python -V; conda env list'")
    print(stdout.read().decode())
    
    # We will kill the previous run if it failed
    client.exec_command("killall bash; killall python")
    
    # Now run the correct setup
    client.exec_command("bash -lc 'pip install ultralytics'")
    client.exec_command("bash -lc 'cd /root/autodl-tmp && nohup bash tools/run_coco_cluster.sh > run_cluster.log 2>&1 &'")
    
    print("Fixed and restarted!")
    client.close()

if __name__ == '__main__':
    main()
