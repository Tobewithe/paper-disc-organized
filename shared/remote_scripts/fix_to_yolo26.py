import paramiko
import time

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    # 1. Kill the current YOLOv8 training
    client.exec_command("pkill -f train_coco_ccl")
    
    # 2. Modify the script to use YOLO26
    client.exec_command("sed -i 's/yolov8m-seg.pt/yolo26m-seg.pt/g' /root/autodl-tmp/tools/train_coco_ccl.py")
    
    # 3. Clean up the logs
    client.exec_command("rm /root/autodl-tmp/run_cluster_fixed.log")
    
    # 4. Restart the cluster
    client.exec_command("cd /root/autodl-tmp && nohup bash tools/run_coco_cluster.sh > run_cluster_fixed.log 2>&1 &")
    
    print("Switched to YOLO26 and restarted!")
    client.close()

if __name__ == '__main__':
    main()
