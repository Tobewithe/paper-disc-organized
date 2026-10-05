import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    print("--- GPU Status ---")
    stdin, stdout, stderr = client.exec_command("nvidia-smi")
    print(stdout.read().decode())
    
    print("--- Processes ---")
    stdin, stdout, stderr = client.exec_command("ps aux | grep python")
    print(stdout.read().decode())
    
    print("--- Log File End ---")
    stdin, stdout, stderr = client.exec_command("tail -n 20 /root/autodl-tmp/run_cluster_fixed.log")
    print(stdout.read().decode())
    
    client.close()

if __name__ == '__main__':
    main()
