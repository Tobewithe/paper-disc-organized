import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("cat /root/autodl-tmp/run_cluster_fixed.log | grep 'Using'")
    raw = stdout.read()
    print("USING:", raw.decode('utf-8', errors='ignore'))
    
    stdin, stdout, stderr = client.exec_command("cat /root/autodl-tmp/run_cluster_fixed.log | grep -A 2 'ignoring'")
    raw = stdout.read()
    print("IGNORING:", raw.decode('utf-8', errors='ignore')[:1000]) # just print a bit to see
    
    client.close()

if __name__ == '__main__':
    main()
