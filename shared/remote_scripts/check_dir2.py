import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("find /root/autodl-tmp/datasets/coco -maxdepth 2")
    print(stdout.read().decode('utf-8', errors='ignore'))
    
    client.close()

if __name__ == '__main__':
    main()
