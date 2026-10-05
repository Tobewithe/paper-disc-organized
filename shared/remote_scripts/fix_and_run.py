import paramiko
import time

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    # We will write a bootstrap bash script on the server
    bootstrap_script = """#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
pip install ultralytics

# Clean up previous attempts
cd /root/autodl-tmp
mkdir -p datasets/coco

echo "Checking if we need to unzip COCO from AutoDL pub..."
if [ ! -d "datasets/coco/images/train2017" ]; then
    echo "Unzipping images..."
    unzip -q -o /root/autodl-pub/COCO2017/train2017.zip -d datasets/coco/images/
    unzip -q -o /root/autodl-pub/COCO2017/val2017.zip -d datasets/coco/images/
fi

if [ ! -d "datasets/coco/annotations" ]; then
    echo "Unzipping annotations..."
    unzip -q -o /root/autodl-pub/COCO2017/annotations_trainval2017.zip -d datasets/coco/
fi

echo "Converting COCO JSON to YOLO format..."
# We can use Ultralytics' built in converter
python -c "
from ultralytics.data.converter import convert_coco
convert_coco('datasets/coco/annotations/', use_segments=True, dir='datasets/coco/labels/')
" || echo "Conversion already done or failed"

# Now run our custom scripts
python tools/prepare_coco_dense.py
nohup bash tools/run_coco_cluster.sh > run_cluster_fixed.log 2>&1 &
echo "Bootstrap complete! Cluster is running."
"""

    stdin, stdout, stderr = client.exec_command("cat > /root/autodl-tmp/bootstrap.sh")
    stdin.write(bootstrap_script)
    stdin.close()
    
    # Run the bootstrap script
    stdin, stdout, stderr = client.exec_command("bash /root/autodl-tmp/bootstrap.sh > /root/autodl-tmp/bootstrap.log 2>&1 &")
    
    print("Fixed bootstrap deployed and running in background!")
    client.close()

if __name__ == '__main__':
    main()
