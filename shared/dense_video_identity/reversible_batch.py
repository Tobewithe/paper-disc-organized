"""Execute frozen comparisons with results concealed until raw annotation is frozen."""
from pathlib import Path
import contextlib
from .common import read_json,write_json,now
from .reversible_identity import main


def execute(root):
    p=read_json(root/'reversible_smoke_pointer.json');assert read_json(root/p['path']/'SUMMARY.json')['smoke']
    meta=read_json(root/'study.json');meta.update(execution_status='running',plan_status='fixed_complete_comparison_and_blind_annotation',updated_at=now());write_json(root/'study.json',meta)
    for split in ['engineering','retrospective','independent']:
        log=root/('comparison_'+split+'.log');assert not log.exists()
        print('COMPARISON_START',split,flush=True)
        with log.open('w',encoding='utf-8') as stream,contextlib.redirect_stdout(stream):main(root,False,split)
        print('COMPARISON_COMPLETE',split,flush=True)
    print('All fixed comparisons saved; scores not printed during annotation',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();execute(a.root)
