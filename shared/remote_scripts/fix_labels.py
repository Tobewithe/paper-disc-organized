import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    # 1. Kill hanging YOLO
    client.exec_command("pkill -f train_coco_ccl")
    
    # 2. Run the converter interactively to see what happens
    stdin, stdout, stderr = client.exec_command("bash -lc 'python -c \"from ultralytics.data.converter import convert_coco; convert_coco(\\\"/root/autodl-tmp/datasets/coco/annotations/\\\", use_segments=True, dir=\\\"/root/autodl-tmp/datasets/coco/\\\")\"'")
    print(stdout.read().decode())
    print(stderr.read().decode())
    
    # 3. Add PYTHONUNBUFFERED=1 to run_coco_cluster.sh
    client.exec_command("sed -i 's/python tools/PYTHONUNBUFFERED=1 python tools/g' /root/autodl-tmp/tools/run_coco_cluster.sh")
    
    # 4. Restart training
    client.exec_command("cd /root/autodl-tmp && nohup bash tools/run_coco_cluster.sh > run_cluster_fixed.log 2>&1 &")
    
    client.close()

if __name__ == '__main__':
    main()
