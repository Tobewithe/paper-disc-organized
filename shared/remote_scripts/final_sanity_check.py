import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    print("=== DATASET LOADING ===")
    stdin, stdout, stderr = client.exec_command("cat /root/autodl-tmp/run_cluster_fixed.log | grep 'Using'")
    print(stdout.read().decode('utf-8', errors='ignore'))
    
    print("=== GPU UTILIZATION ===")
    stdin, stdout, stderr = client.exec_command("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv")
    print(stdout.read().decode('utf-8', errors='ignore'))
    
    print("=== LATEST LOGS ===")
    # Tail the log avoiding emoji crashes by replacing them
    stdin, stdout, stderr = client.exec_command("tail -n 25 /root/autodl-tmp/run_cluster_fixed.log")
    raw = stdout.read()
    print(raw.decode('utf-8', errors='ignore').encode('ascii', 'ignore').decode('ascii'))
    
    client.close()

if __name__ == '__main__':
    main()
