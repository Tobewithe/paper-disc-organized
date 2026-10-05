import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("find /root/autodl-tmp/datasets/coco/ -type d -name '*label*'")
    print(stdout.read().decode('utf-8', errors='ignore'))
    
    stdin, stdout, stderr = client.exec_command("ls /root/autodl-tmp/datasets/coco/labels/default/ | head -n 5")
    print("Files in default:", stdout.read().decode('utf-8', errors='ignore'))

    client.close()

if __name__ == '__main__':
    main()
