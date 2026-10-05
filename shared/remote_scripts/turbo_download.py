import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    # 1. Kill the slow process
    client.exec_command("pkill -f train_coco_ccl")
    
    # 2. Add AutoDL academic acceleration (network_turbo) to the bash script
    fix_script = """sed -i '2i source /etc/network_turbo' /root/autodl-tmp/tools/run_coco_cluster.sh"""
    client.exec_command(fix_script)
    
    # 3. Pre-download the model using the turbo proxy explicitly to make sure it's cached
    dl_cmd = "bash -lc 'source /etc/network_turbo && source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && python -c \"from ultralytics import YOLO; YOLO(\\\"yolo26m-seg.pt\\\")\"'"
    print("Starting turbo download...")
    stdin, stdout, stderr = client.exec_command(dl_cmd)
    
    # Wait for download to finish
    out = stdout.read().decode()
    err = stderr.read().decode()
    print("Download output:", out)
    print("Download err:", err)
    
    # 4. Restart training
    client.exec_command("rm /root/autodl-tmp/run_cluster_fixed.log")
    client.exec_command("cd /root/autodl-tmp && nohup bash tools/run_coco_cluster.sh > run_cluster_fixed.log 2>&1 &")
    
    print("Turbo applied and restarted!")
    client.close()

if __name__ == '__main__':
    main()
