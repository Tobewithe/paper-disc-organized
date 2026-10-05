import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("cat /root/autodl-tmp/run_cluster_fixed.log | grep -E 'train:|WARNING|Scanning|images'")
    
    raw = stdout.read()
    text = raw.decode('utf-8', errors='ignore').encode('ascii', 'ignore').decode('ascii')
    print(text)
    
    client.close()

if __name__ == '__main__':
    main()
