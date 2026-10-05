import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    print("--- Log File End ---")
    stdin, stdout, stderr = client.exec_command("tail -n 25 /root/autodl-tmp/run_cluster_fixed.log")
    print(stdout.read().decode())
    
    client.close()

if __name__ == '__main__':
    main()
