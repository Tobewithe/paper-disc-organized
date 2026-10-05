import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    stdin, stdout, stderr = client.exec_command("ls /root/autodl-tmp/datasets/coco/labels/ | wc -l")
    print("Labels count:", stdout.read().decode())
    
    # Check if download of yolov8m is happening
    stdin, stdout, stderr = client.exec_command("ls -lh /root/autodl-tmp/")
    print(stdout.read().decode())

    client.close()

if __name__ == '__main__':
    main()
