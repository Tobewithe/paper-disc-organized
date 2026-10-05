import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    script = """#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

python tools/prepare_coco_dense.py
python tools/train_coco_ccl.py --weight 0.0 --margin 0.0
python tools/train_coco_ccl.py --weight 0.1 --margin 0.1
python tools/train_coco_ccl.py --weight 0.5 --margin 0.1

echo "All COCO Dense Ablations Finished!"
"""
    stdin, stdout, stderr = client.exec_command("cat > /root/autodl-tmp/tools/run_coco_cluster.sh")
    stdin.write(script)
    stdin.close()
    
    client.exec_command("rm /root/autodl-tmp/run_cluster_fixed.log")
    client.exec_command("cd /root/autodl-tmp && nohup bash tools/run_coco_cluster.sh > run_cluster_fixed.log 2>&1 &")
    
    print("Fixed sh and restarted!")
    client.close()

if __name__ == '__main__':
    main()
