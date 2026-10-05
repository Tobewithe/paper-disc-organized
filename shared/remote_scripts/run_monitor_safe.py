import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    # run monitor.sh for one iteration if it's a loop, or just dump it
    # monitor.sh might be a watch loop or tail -f. If it's a TUI loop, running it might hang.
    # Let's inspect the first 10 lines of the script safely.
    stdin, stdout, stderr = client.exec_command("head -n 20 /root/autodl-tmp/monitor.sh")
    raw = stdout.read()
    print("SCRIPT HEAD:", raw.decode('utf-8', errors='ignore').encode('ascii', 'ignore').decode('ascii'))
    
    client.close()

if __name__ == '__main__':
    main()
