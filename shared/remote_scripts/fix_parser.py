import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    # Fix the train_coco_ccl.py file
    fix_cmd = "sed -i 's/args = parser.add_argument()/args = parser.parse_args()/g' /root/autodl-tmp/tools/train_coco_ccl.py"
    client.exec_command(fix_cmd)
    
    # Restart the training script directly
    client.exec_command("rm /root/autodl-tmp/run_cluster_fixed.log")
    client.exec_command("cd /root/autodl-tmp && nohup bash tools/run_coco_cluster.sh > run_cluster_fixed.log 2>&1 &")
    
    print("Fixed syntax error and restarted training!")
    client.close()

if __name__ == '__main__':
    main()
