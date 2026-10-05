import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    # Check if CCL ACTIVE was ever printed
    stdin, stdout, stderr = client.exec_command("cat /root/autodl-tmp/run_cluster_fixed.log | grep 'CCL ACTIVE'")
    print("CCL PRINTS:", stdout.read().decode('utf-8', errors='ignore'))
    
    # Check which script is running
    stdin, stdout, stderr = client.exec_command("ps aux | grep train_coco_ccl")
    print("PROCESSES:", stdout.read().decode('utf-8', errors='ignore'))
    
    client.close()

if __name__ == '__main__':
    main()
