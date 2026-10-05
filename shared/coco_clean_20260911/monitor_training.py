#!/usr/bin/env python3
"""Read-only terminal dashboard for this experiment; Python standard library only."""
import argparse
import csv
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
TRAIN = re.compile(r'^\s*(\d+)/(\d+)\s+[\d.]+G\s+')
PROGRESS = re.compile(r':\s*(\d+)%.*?\s(\d+)/(\d+)(?:\s|$)')
DONE = {'COMPLETE', 'COMPLETE_EXISTING'}
LABELS = {'RUNNING': '运行', 'PENDING': '排队', 'COMPLETE': '完成',
          'COMPLETE_EXISTING': '完成', 'FAILED': '失败', 'UNKNOWN': '未知'}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def tail_text(path, limit=65536):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size-limit))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def clean_lines(raw):
    raw = ANSI.sub('', raw)
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
    return [line.strip() for line in raw.splitlines() if line.strip()]


def csv_rows(path):
    try:
        content = path.read_text(encoding='utf-8')
        # A concurrently written trailing row may be incomplete.
        content = content[:content.rfind('\n')+1]
        rows = []
        for row in csv.DictReader(io.StringIO(content)):
            if None in row or any(v is None for v in row.values()):
                continue
            row = {k.strip(): v.strip() for k, v in row.items()}
            if numeric(row.get('epoch')) is not None:
                rows.append(row)
        return rows
    except (OSError, csv.Error):
        return []


def last_integrity(path):
    for line in reversed(tail_text(path).splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and 'completed_epoch' in value:
                return value
        except ValueError:
            pass
    return {}


def numeric(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def process_state(pid, expected):
    if not pid:
        return '未知'
    if not sys.platform.startswith('linux'):
        return '未知（需在训练服务器运行）'
    try:
        proc = Path('/proc')/str(int(pid))
        state = (proc/'stat').read_text().rsplit(')', 1)[1].split()[0]
        if state in {'Z', 'X'}:
            return '已退出'
        arguments = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        if not any(Path(arg).name == expected for arg in arguments if arg):
            return 'PID已复用'
        return '存活'
    except (FileNotFoundError, ProcessLookupError):
        return '已退出'
    except (OSError, ValueError, IndexError):
        return '不可读'


def gpu_snapshot():
    fields = ['index', 'name', 'utilization.gpu', 'memory.used', 'memory.total',
              'temperature.gpu', 'power.draw']
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu='+','.join(fields), '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3, check=True)
        return [dict(zip(fields, [v.strip() for v in row]))
                for row in csv.reader(result.stdout.splitlines()) if len(row) == len(fields)]
    except (OSError, subprocess.SubprocessError):
        return []


def registered_checkpoints(directory):
    index = read_json(directory/'checkpoint_index.json', [])
    count = 0
    for entry in index if isinstance(index, list) else []:
        try:
            path = directory/'weights'/entry['file']
            count += bool(path.is_file() and path.stat().st_size == entry['bytes'])
        except (OSError, KeyError, TypeError):
            pass
    return count


def log_age(path):
    try:
        return max(0, time.time()-path.stat().st_mtime)
    except OSError:
        return None


def snapshot(root, stale_seconds):
    queue = read_json(root/'queue_status.json', {})
    config = read_json(root/'train_config.json', {})
    entries = queue.get('jobs') or [dict(seed=s, arm=a, status='PENDING')
                                  for s in [0, 1, 2] for a in ['baseline', 'ccl01']]
    jobs, alerts = [], []
    queue_process = process_state(queue.get('pid'), 'run_queue.py')
    if queue.get('status') == 'RUNNING' and queue_process in {'已退出', 'PID已复用'}:
        alerts.append('队列记录为 RUNNING，但队列进程已不在；状态文件可能停留在退出前。')
    if queue.get('error'):
        alerts.append(str(queue['error']))
    for entry in entries:
        name = f"{entry['arm']}_s{entry['seed']}"
        directory = root/'runs'/name
        log = root/'logs'/f'{name}.log'
        rows = csv_rows(directory/'results.csv')
        latest = rows[-1] if rows else {}
        integrity = last_integrity(directory/'epoch_integrity.jsonl')
        receipt = read_json(directory/'launch_receipt.json', {})
        lines = clean_lines(tail_text(log)) if entry['status'] in {'RUNNING', 'FAILED'} else []
        progress_line = next((line for line in reversed(lines) if PROGRESS.search(line)), '')
        match = PROGRESS.search(progress_line)
        train_match = TRAIN.match(progress_line)
        stage = '训练' if train_match else '验证/准备'
        epoch = int(train_match[1]) if train_match else integrity.get('completed_epoch')
        alive = process_state(entry.get('pid'), 'train_pair.py') if entry['status'] == 'RUNNING' else '—'
        age = log_age(log)
        if entry['status'] == 'RUNNING':
            if alive in {'已退出', 'PID已复用'}:
                alerts.append(f'{name}：训练进程已不在，等待队列状态更新或检查日志。')
            elif age is not None and age > stale_seconds:
                alerts.append(f'{name}：日志 {age/60:.1f} 分钟未更新，进程{alive}；需检查，不直接判定失败。')
        job = dict(name=name, seed=entry['seed'], arm=entry['arm'], status=entry['status'],
                   pid=entry.get('pid'), process=alive, epochs=receipt.get('epochs', config.get('epochs', 15)),
                   validated_epoch=int(float(latest['epoch'])) if latest else 0,
                   trained_epoch=integrity.get('completed_epoch', 0), current_epoch=epoch,
                   checkpoints=registered_checkpoints(directory), metrics=latest, integrity=integrity,
                   log=str(log), log_age_seconds=age, tail=lines[-6:], progress_line=progress_line,
                   phase=stage, progress=dict(percent=int(match[1]), step=int(match[2]), total=int(match[3])) if match else None)
        jobs.append(job)
    disk = shutil.disk_usage(root)
    if disk.free < 5*1024**3:
        alerts.append(f'实验盘剩余空间仅 {disk.free/1024**3:.1f} GiB，请检查 checkpoint 存储。')
    return dict(time=datetime.now().astimezone().isoformat(timespec='seconds'), root=str(root),
                queue_status=queue.get('status', 'UNKNOWN'), queue_pid=queue.get('pid'), queue_process=queue_process,
                active=queue.get('active'), jobs=jobs, gpu=gpu_snapshot(),
                disk_free_gib=disk.free/1024**3, disk_total_gib=disk.total/1024**3, alerts=alerts)


def number(value, scale=1, digits=2):
    value = numeric(value)
    return f'{value*scale:.{digits}f}' if value is not None else '—'


def dashboard(data, interval, log_lines):
    jobs = data['jobs']
    finished = sum(job['status'] in DONE for job in jobs)
    ckpts = sum(job['checkpoints'] for job in jobs)
    total = sum(job['epochs'] for job in jobs)
    lines = [f"COCO 三种子训练监控  {data['time']}  刷新 {interval:g}s",
             f"队列 {data['queue_status']} | 完成 {finished}/{len(jobs)} 组 | 已登记 checkpoint {ckpts}/{total}",
             f"队列 PID {data['queue_pid'] or '—'}（{data['queue_process']}） | 磁盘剩余 {data['disk_free_gib']:.1f}/{data['disk_total_gib']:.1f} GiB"]
    for gpu in data['gpu']:
        lines.append(f"GPU {gpu['index']} {gpu['name']} | 利用率 {gpu['utilization.gpu']}% | 显存 {gpu['memory.used']}/{gpu['memory.total']} MiB | {gpu['temperature.gpu']}℃ | {gpu['power.draw']}W")
    if not data['gpu']:
        lines.append('GPU: nvidia-smi 暂不可用')
    lines += ['', '任务            状态   验证轮次  保存轮数  Mask AP  AP50   Box AP  Seg loss']
    for job in jobs:
        m = job['metrics']
        lines.append(f"{job['name']:<15} {LABELS.get(job['status'], job['status']):<4} "
                     f"{job['validated_epoch']:>2}/{job['epochs']:<2}    {job['checkpoints']:>2}/{job['epochs']:<2}   "
                     f"{number(m.get('metrics/mAP50-95(M)'), 100):>6}  "
                     f"{number(m.get('metrics/mAP50(M)'), 100):>5}  "
                     f"{number(m.get('metrics/mAP50-95(B)'), 100):>6}  "
                     f"{number(m.get('train/seg_loss'), digits=4):>8}")
    lines.append('AP 按 0–100 显示；每行是最新已验证轮次。来自训练验证，最终原始 COCOeval 另算。')
    for job in jobs:
        if job['status'] not in {'RUNNING', 'FAILED'}:
            continue
        age = job['log_age_seconds']
        age_text = f'{age:.0f}s' if age is not None else '未知'
        lines += ['', f"当前 {job['name']} | PID {job['pid'] or '—'}（{job['process']}） | 日志距今 {age_text}"]
        if job['progress']:
            p = job['progress']
            epoch = job['current_epoch'] or '?'
            lines.append(f"{job['phase']} | 轮次 {epoch}/{job['epochs']} | {p['step']}/{p['total']} ({p['percent']}%)")
            lines.append('最近进度：'+job['progress_line'])
        record = job['integrity']
        if record:
            lines.append(f"最近训练完成轮次 {record['completed_epoch']} | 累计优化器步数 {record.get('optimizer_steps', '—')} | 最近梯度范数 {number(record.get('last_grad_norm'))}")
            if job['arm'] == 'ccl01':
                for branch in ['o2m', 'o2o']:
                    stats = record.get(branch, {})
                    pairs = stats.get('pairs', 0)
                    mean = stats.get('raw_sum', 0)/pairs if pairs else None
                    lines.append(f"CCL {branch}: 配对 {pairs:,} | 激活配对 {stats.get('active_pairs', 0):,} | 原始惩罚均值 {number(mean, digits=4)}")
        if log_lines:
            lines.append('最近日志：')
            # Long integrity JSON / launch JSON are saved in full in the source log.
            lines.extend('  '+line[:220] for line in job['tail'][-log_lines:])
    if data['alerts']:
        lines += ['', '需检查：']+[f'  ! {alert}' for alert in data['alerts']]
    if data['queue_status'] == 'FAILED':
        footer = '队列已停止，等待处理。Ctrl+C 只退出监控，不会恢复训练。'
    elif data['queue_status'] == 'COMPLETE':
        footer = '队列全部完成。Ctrl+C 退出监控。'
    elif data['queue_process'] != '存活':
        footer = '队列进程状态需检查。Ctrl+C 只退出监控。'
    else:
        footer = 'Ctrl+C 退出监控；训练由独立队列管理。'
    lines += ['', footer]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--interval', type=float, default=5, help='Refresh seconds, minimum 1')
    parser.add_argument('--once', action='store_true', help='Print one snapshot and exit')
    parser.add_argument('--json', action='store_true', help='Print one machine-readable snapshot and exit')
    parser.add_argument('--no-clear', action='store_true', help='Append snapshots instead of redrawing')
    parser.add_argument('--log-lines', type=int, choices=range(0, 7), default=3)
    parser.add_argument('--stale-minutes', type=float, default=10, help='Alert after log inactivity; not a failure verdict')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1 or not math.isfinite(args.stale_minutes) or args.stale_minutes <= 0:
        parser.error('interval must be >=1; stale-minutes must be positive; both must be finite')
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f'Experiment directory does not exist: {root}')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    try:
        while True:
            data = snapshot(root, args.stale_minutes*60)
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
                break
            if sys.stdout.isatty() and not args.no_clear and not args.once:
                print('\033[2J\033[H', end='')
            print(dashboard(data, args.interval, args.log_lines), flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n已退出监控，训练继续。')
    except BrokenPipeError:
        pass


if __name__ == '__main__':
    main()
