import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("cat /root/miniconda3/lib/python3.12/site-packages/ultralytics/utils/loss.py | grep -A 5 'class v8DetectionLoss' | grep 'loss_names'")
    print("DET NAMES:", stdout.read().decode('utf-8', errors='ignore'))
    
    client.close()

if __name__ == '__main__':
    main()
