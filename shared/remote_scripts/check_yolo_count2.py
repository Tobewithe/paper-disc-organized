import paramiko
import time

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    time.sleep(10) # wait for cache to build
    
    stdin, stdout, stderr = client.exec_command("cat /root/autodl-tmp/run_cluster_fixed.log | grep 'Using'")
    raw = stdout.read()
    print("USING:", raw.decode('utf-8', errors='ignore'))
    
    client.close()

if __name__ == '__main__':
    main()
