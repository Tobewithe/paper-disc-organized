import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("find /root/ -name 'yolo26*.pt'")
    print(stdout.read().decode())
    
    client.close()

if __name__ == '__main__':
    main()
