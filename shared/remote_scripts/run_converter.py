import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    script = """#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
python -c "
from ultralytics.data.converter import convert_coco
print('Starting conversion...')
convert_coco(annotations_dir='/root/autodl-tmp/datasets/coco/annotations/', use_segments=True, dir='/root/autodl-tmp/datasets/coco/')
print('Conversion done!')
"
"""
    client.exec_command("cat > /root/autodl-tmp/run_conv.sh")
    stdin, stdout, stderr = client.exec_command("cat > /root/autodl-tmp/run_conv.sh")
    stdin.write(script)
    stdin.close()
    
    print("Executing converter...")
    stdin, stdout, stderr = client.exec_command("bash /root/autodl-tmp/run_conv.sh")
    print("STDOUT:", stdout.read().decode())
    print("STDERR:", stderr.read().decode())
    
    client.close()

if __name__ == '__main__':
    main()
