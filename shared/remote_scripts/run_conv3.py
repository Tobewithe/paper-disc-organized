import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    script = """#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# Remove non-instance JSONs so convert_coco doesn't crash
cd /root/autodl-tmp/datasets/coco/annotations/
ls | grep -v 'instances' | xargs rm -f
cd /root/autodl-tmp/

python -c "
from ultralytics.data.converter import convert_coco
print('Starting conversion...')
convert_coco(labels_dir='/root/autodl-tmp/datasets/coco/annotations/', save_dir='/root/autodl-tmp/datasets/coco/', use_segments=True)
print('Conversion done!')
"

sed -i 's/python tools/PYTHONUNBUFFERED=1 python tools/g' /root/autodl-tmp/tools/run_coco_cluster.sh

cd /root/autodl-tmp
rm -f run_cluster_fixed.log
nohup bash tools/run_coco_cluster.sh > run_cluster_fixed.log 2>&1 &
"""
    client.exec_command("cat > /root/autodl-tmp/run_conv.sh")[0].write(script)
    
    print("Executing clean and convert...")
    # Run synchronously to see if it finishes converting correctly
    stdin, stdout, stderr = client.exec_command("bash /root/autodl-tmp/run_conv.sh")
    out = stdout.read().decode()
    err = stderr.read().decode()
    print("STDOUT:", out[:500])
    print("STDERR:", err[:500])
    
    client.close()

if __name__ == '__main__':
    main()
