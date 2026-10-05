import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("ls /root/autodl-fs/ || ls /root/autodl-pub/")
    print("Pub datasets:", stdout.read().decode())

    stdin, stdout, stderr = client.exec_command("find /root/ -name '*coco*' -maxdepth 3")
    print("Find coco:", stdout.read().decode())

    client.close()

if __name__ == '__main__':
    main()
