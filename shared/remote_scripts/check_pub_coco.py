import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("ls -l /root/autodl-pub/COCO2017/")
    print(stdout.read().decode())
    
    stdin, stdout, stderr = client.exec_command("ls -l /root/autodl-pub/COCO2017/annotations/")
    print(stdout.read().decode())

    client.close()

if __name__ == '__main__':
    main()
