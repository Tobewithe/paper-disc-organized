import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    client.exec_command("mv /root/yolo26m-seg.pt /root/autodl-tmp/yolo26m-seg.pt")
    
    client.exec_command("pkill -f train_coco_ccl")
    client.exec_command("rm /root/autodl-tmp/run_cluster_fixed.log")
    client.exec_command("cd /root/autodl-tmp && nohup bash tools/run_coco_cluster.sh > run_cluster_fixed.log 2>&1 &")
    
    client.close()

if __name__ == '__main__':
    main()
