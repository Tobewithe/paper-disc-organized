import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("ls -lt /root/autodl-tmp/runs/segment/coco_dense_ablations/")
    print("RUNS FOLDERS:", stdout.read().decode('utf-8', errors='ignore'))
    
    # Read the last few lines of results.csv for both
    stdin, stdout, stderr = client.exec_command("cat /root/autodl-tmp/runs/segment/coco_dense_ablations/ccl_w0.0_m0.0*/results.csv | tail -n 2")
    print("W=0.0 results:", stdout.read().decode('utf-8', errors='ignore'))
    
    stdin, stdout, stderr = client.exec_command("cat /root/autodl-tmp/runs/segment/coco_dense_ablations/ccl_w0.1_m0.1*/results.csv | tail -n 2")
    print("W=0.1 results:", stdout.read().decode('utf-8', errors='ignore'))
    
    client.close()

if __name__ == '__main__':
    main()
