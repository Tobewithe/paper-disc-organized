"""Collect records/results; large checkpoints and inference exports stay at recorded remote paths."""
import argparse,tarfile
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    count=0
    with tarfile.open(a.out,'w:gz',compresslevel=1) as archive:
        for top in ['runs','data','queue.json','queue.log','queue_continuation.json','queue_continuation.log','certified_receipts.json']:
            entry=a.study/top
            for f in sorted(entry.rglob('*')) if entry.is_dir() else [entry]:
                if not f.is_file():continue
                rel=f.relative_to(a.study)
                if any(part in ['weights','exports','__pycache__'] for part in rel.parts) or f.suffix in ['.pt','.cache','.pyc']:continue
                archive.add(f,arcname=rel.as_posix(),recursive=False);count+=1
    print('RECORDS_PACKED',count,a.out.stat().st_size,flush=True)
if __name__=='__main__':main()
