import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("cat /root/miniconda3/lib/python3.12/site-packages/ultralytics/nn/modules/head.py | grep -A 20 'class Segment26'")
    print("Segment26 head.py:", stdout.read().decode('utf-8', errors='ignore'))
    
    stdin, stdout, stderr = client.exec_command("cat /root/miniconda3/lib/python3.12/site-packages/ultralytics/utils/loss.py | grep 'v26'")
    print("loss.py v26:", stdout.read().decode('utf-8', errors='ignore'))

    client.close()

if __name__ == '__main__':
    main()
