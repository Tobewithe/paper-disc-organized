import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.westd.seetacloud.com', 17381, 'root', '/eIOAILi96O8')
    
    monitor_script = """#!/bin/bash
clear
echo "======================================================"
echo "      🚀 顶会论文打榜 - 全流程上帝视角监控台"
echo "======================================================"

# 第一阶段：环境与数据
echo -e "\n\033[33m---> [第一阶段] 正在监控：环境配置与数据集解压...\033[0m"
touch /root/autodl-tmp/bootstrap.log

if grep -q "Bootstrap complete" /root/autodl-tmp/bootstrap.log; then
    echo -e "\033[32m✔ 数据集与环境准备已完美结束！\033[0m"
else
    # 追踪直到出现关键字后自动退出
    tail -n 20 -f /root/autodl-tmp/bootstrap.log | while read LINE; do
        echo "\"
        if [[ "\" == *"Bootstrap complete"* ]]; then
            pkill -P  tail
            break
        fi
    done
fi

# 第二阶段：正式训练
echo -e "\n\033[33m---> [第二阶段] 正在切入监控：消融实验与 YOLO 训练进度...\033[0m"
while [ ! -f "/root/autodl-tmp/run_cluster_fixed.log" ]; do 
    sleep 2
done

# 无缝接管训练日志
tail -n +1 -f /root/autodl-tmp/run_cluster_fixed.log
"""

    stdin, stdout, stderr = client.exec_command("cat > /root/autodl-tmp/monitor.sh && chmod +x /root/autodl-tmp/monitor.sh")
    stdin.write(monitor_script)
    stdin.close()
    
    print("Monitor script created on AutoDL.")
    client.close()

if __name__ == '__main__':
    main()
