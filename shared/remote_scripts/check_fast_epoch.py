import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("ls /root/autodl-tmp/datasets/coco/labels/train2017/ | wc -l")
    print("Labels in train2017:", stdout.read().decode().strip())
    
    stdin, stdout, stderr = client.exec_command("head -n 20 /root/autodl-tmp/run_cluster_fixed.log")
    print("Log head:", stdout.read().decode('utf-8', errors='ignore'))
    
    client.close()

if __name__ == '__main__':
    main()
