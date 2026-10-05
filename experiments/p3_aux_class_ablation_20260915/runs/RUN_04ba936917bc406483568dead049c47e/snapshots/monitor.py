"""Read-only queue monitor; Ctrl+C stops this viewer only."""
import argparse,csv,json,time
from pathlib import Path

def show(study):
    cfg=json.loads((study/'protocol.json').read_text());q=json.loads((study/'queue.json').read_text())
    print('\n',time.strftime('%Y-%m-%d %H:%M:%S'),q['status'],'completed',len(q['completed']),'/',len(cfg['runs']))
    for spec in cfg['runs']:
        if spec['kind'] not in ['smoke','training']:continue
        folder=study/'runs'/spec['run_id'];status='queued';epoch=0;ap='-'
        if (folder/'run.json').exists():status=json.loads((folder/'run.json').read_text())['status']
        if (folder/'results.csv').exists():
            with (folder/'results.csv').open() as f:rows=list(csv.DictReader(f))
            if rows:
                row={k.strip():v for k,v in rows[-1].items()};epoch=row.get('epoch','?')
                ap=row.get('metrics/mAP50-95(M)','-')
        print(f'{spec["arm"]:18} {status:10} epoch={epoch} native_mask_AP={ap}')
    if q.get('current'):
        print('current:',q['current'])
        folder=study/'runs'/q['current']['run_id']
        for filename in ['stdout.log','stderr.log']:
            path=folder/filename
            if path.exists():
                with path.open('rb') as f:f.seek(max(0,path.stat().st_size-5000));tail=f.read().decode(errors='replace')
                lines=[v for v in tail.replace('\r','\n').splitlines() if v.strip()]
                print(filename, '\n'.join(lines[-2:]))
    if q.get('error'):print(q['error'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--watch',type=int,default=0);a=p.parse_args()
    while True:
        show(a.study)
        if not a.watch:break
        time.sleep(max(1,a.watch))

if __name__=='__main__':main()
