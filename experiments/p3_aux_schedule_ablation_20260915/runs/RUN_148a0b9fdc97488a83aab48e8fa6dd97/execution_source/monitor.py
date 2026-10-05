"""Read queue and recent logs; never changes a training process."""
import argparse,csv,json,time
from pathlib import Path

def show(study):
    q=json.loads((study/'queue.json').read_text());cfg=json.loads((study/'protocol.json').read_text())
    print(time.strftime('%Y-%m-%d %H:%M:%S'),q['status'],len(q['completed']),'/',len(cfg['runs']))
    for spec in cfg['runs']:
        folder=study/'runs'/spec['run_id'];status='queued'
        if (folder/'run.json').exists():status=json.loads((folder/'run.json').read_text())['status']
        print(spec['kind'],spec.get('checkpoint',''),status,spec['run_id'])
    if q.get('current'):
        folder=study/'runs'/q['current']['run_id']
        for name in ['stdout.log','stderr.log']:
            f=folder/name
            if f.exists():
                with f.open('rb') as h:h.seek(max(0,f.stat().st_size-4000));data=h.read().decode(errors='replace')
                print(name,'\n'.join([x for x in data.replace('\r','\n').splitlines() if x.strip()][-2:]))
    if q.get('error'):print(q['error'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--watch',type=int,default=0);a=p.parse_args()
    while True:
        show(a.study)
        if not a.watch:break
        time.sleep(max(1,a.watch))

if __name__=='__main__':main()
