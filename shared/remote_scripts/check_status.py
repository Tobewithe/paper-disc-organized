import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    except Exception as e:
        print(f"Failed to connect: {e}")
        return
        
    print("--- Checking running Python processes ---")
    stdin, stdout, stderr = client.exec_command("ps aux | grep python")
    print(stdout.read().decode())
    
    print("--- Checking run_cluster.log ---")
    stdin, stdout, stderr = client.exec_command("tail -n 20 /root/autodl-tmp/run_cluster.log")
    print(stdout.read().decode())
    
    print("--- Checking GPU usage ---")
    stdin, stdout, stderr = client.exec_command("nvidia-smi")
    print(stdout.read().decode())

    client.close()

if __name__ == '__main__':
    main()
