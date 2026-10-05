import paramiko
import time

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    time.sleep(15) # Wait for YOLO to cache and start epoch 1
    
    stdin, stdout, stderr = client.exec_command("cat /root/autodl-tmp/run_cluster_fixed.log | grep 'CCL ACTIVE' | head -n 5")
    raw = stdout.read()
    print("CCL PRINTS:", raw.decode('utf-8', errors='ignore'))
    
    stdin, stdout, stderr = client.exec_command("tail -n 20 /root/autodl-tmp/run_cluster_fixed.log")
    raw = stdout.read()
    print("LOGS:", raw.decode('utf-8', errors='ignore').encode('ascii', 'ignore').decode('ascii'))
    
    client.close()

if __name__ == '__main__':
    main()
