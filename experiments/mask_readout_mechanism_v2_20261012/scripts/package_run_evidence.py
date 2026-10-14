"""Package completed Run evidence; retain large predictions/cache/data on execution host."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import socket
import zipfile
from posttrain_run import ROOT,STUDY,read_json,require,sha256,write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    args=parser.parse_args()
    for run in args.runs:
        run=run.resolve()
        require(run.is_relative_to((STUDY/'runs').resolve()),'not an owned study Run')
        require(read_json(run/'run.json')['execution_status'] in ('completed','error'),'cannot package active Run')
        if (run/'evidence.zip').exists():
            receipt=read_json(run/'transfer.json')
            require(all(sha256(run/name)==digest for name,digest in receipt['files'].items()),'completed evidence changed')
            print('EXISTING',run/'evidence.zip',flush=True)
            continue
        selected,omitted={},[]
        for path in sorted(run.rglob('*')):
            if not path.is_file():continue
            rel=path.relative_to(run).as_posix()
            if path.name in ('transfer.json','manifest.sha256','evidence.zip'):continue
            if (rel.startswith('trace/cache/') or (rel.startswith('dataset/') and path.name not in ('coco.yaml','NATIVE_ENTRY_ALIAS.json'))
                or path.name.endswith('predictions.json') or path.stat().st_size>16*1024**2):
                omitted.append({'relative_path':rel,'size_bytes':path.stat().st_size,'availability':'retained_on_execution_host'})
            else:selected[rel]=sha256(path)
        write_new(run/'transfer.json',{'source_host':socket.gethostname(),'source_path':str(run),
                  'prepared_at':datetime.now(timezone.utc).isoformat(),'status':'prepared','scope':'lightweight_evidence_and_adapter_checkpoints',
                  'files':selected,'omitted':omitted,'complete_artifact_inventory':'ARTIFACT_MANIFEST.json'})
        with (run/'manifest.sha256').open('x',encoding='utf8') as stream:
            for name,digest in selected.items():stream.write(digest+'  '+name+'\n')
        with zipfile.ZipFile(run/'evidence.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in [*selected,'transfer.json','manifest.sha256']:archive.write(run/name,name)
        print('PACKAGED',run.name,'files',len(selected),'zip_bytes',(run/'evidence.zip').stat().st_size,
              'zip_sha256',sha256(run/'evidence.zip'),flush=True)


if __name__=='__main__':main()
