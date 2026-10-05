import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
        
        print("--- Disk Usage of Extracted Images ---")
        stdin, stdout, stderr = client.exec_command("du -sh /root/autodl-tmp/datasets/coco/images || echo 'Not created yet'")
        print(stdout.read().decode())
        
        print("--- File Count ---")
        stdin, stdout, stderr = client.exec_command("ls /root/autodl-tmp/datasets/coco/images/train2017/ | wc -l")
        print(stdout.read().decode())
        
        print("--- End of bootstrap.log ---")
        stdin, stdout, stderr = client.exec_command("tail -n 10 /root/autodl-tmp/bootstrap.log")
        print(stdout.read().decode())
        
        client.close()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
