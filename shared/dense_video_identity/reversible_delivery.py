"""Verify laptop evidence, make a portable offline exhibit, export hash inventory."""
from pathlib import Path
import json,zipfile,shutil,math,platform,sys,importlib.metadata
from html.parser import HTMLParser
import cv2
from .common import Run,read_json,write_json,sha256,now,setup_paths
from .reversible_showcase import architecture

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.rows=0
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        for k in ['src','href','poster']:
            if k in d:self.links.append(d[k])
        if tag=='tr' and 'data-arm' in d:self.rows+=1

def main(root):
    setup_paths(root)
    import torch
    architecture(root)
    for name in ['REPORT.md','PROTOCOL.md','ENGINEERING_CORRECTIONS.md']:shutil.copy2(root/name,root/'showcase'/name)
    manifest=read_json(root/'input_manifest.json');ptr=read_json(root/'reversible_adjudication_pointer.json');summary=read_json(root/ptr['path']/'SUMMARY.json')
    with Run(root,'reversible_delivery_verification',dict(scope='hashes, provenance, metric arithmetic, full decoding, local links; not method efficacy')) as run:
        assets=[]
        for seq in manifest['sequences']:
            for key in ['video','gt']:
                if seq.get(key):
                    p=Path(seq[key]);actual=sha256(p);assert actual==seq[key+'_sha256'];assets.append(dict(sequence=seq['name'],kind=key,path=str(p),sha256=actual))
        for name,path in manifest['models'].items():assert sha256(Path(path))==manifest['model_sha256'][name]
        identity=read_json(root/'identity_pointer.json');assert sha256(Path(identity['checkpoint']))==identity['sha256']
        snapshots=[]
        for rp in sorted((root/'runs').glob('*/run.json')):
            meta=read_json(rp);folder=rp.parent
            assert sha256(folder/'EVAL_CONTRACT.json')==meta['contract_sha256']
            assert sha256(folder/'SOURCE_SNAPSHOT.json')==meta['source_snapshot_sha256']
            for entry in read_json(folder/'SOURCE_SNAPSHOT.json'):assert sha256(folder/'source'/entry['path'])==entry['sha256']
            for p in (folder/'inputs').rglob('*'):
                if p.is_file():assert sha256(p)==meta[p.relative_to(folder/'inputs').as_posix()+'_sha256']
            snapshots.append(dict(run_id=meta['run_id'],execution_status=meta['execution_status'],executor=meta['executor']['host'],source_files=len(read_json(folder/'SOURCE_SNAPSHOT.json'))))
        for split,s in summary['splits'].items():
            for arm,c in s['combined'].items():
                assert math.isclose(c['IDF1'],200*c['IDTP']/(2*c['IDTP']+c['IDFP']+c['IDFN']),abs_tol=1e-9)
                rows=[r for r in s['by_sequence'] if r['arm']==arm]
                for k in ['IDTP','IDFP','IDFN']:assert sum(r[k] for r in rows)==c[k]
        # Baseline/method observation parity already checked over all 19 videos in adjudication.
        assert len(summary['invariants'])==19 and all(r['geometry_count_identity_unique_latency'] for r in summary['invariants'])
        videos=[]
        for v in read_json(root/'showcase/video_catalog.json'):
            p=root/'showcase'/v['path'];assert sha256(p)==v['sha256'];cap=cv2.VideoCapture(str(p));n=0
            while True:
                ok,img=cap.read()
                if not ok:break
                assert img.shape[:2]==(900,1600);n+=1
            cap.release();assert n==v['frames'];videos.append(dict(file=v['path'],decoded_frames=n,sha256=sha256(p)))
        parser=Links();parser.feed((root/'showcase/index.html').read_text(encoding='utf-8'));assert parser.rows==19*7
        for ref in parser.links:
            if not ref.startswith(('#','https:','http:','data:')):assert (root/'showcase'/ref.split('#')[0]).is_file(),ref
        sources=[dict(path=p.relative_to(root).as_posix(),sha256=sha256(p)) for p in sorted((root/'code/shared/dense_video_identity').rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
        write_json(root/'CURRENT_SOURCE_MANIFEST.json',dict(recorded_at=now(),scope='final workspace; actual execution authority remains per-run snapshot',files=sources))
        dependency=[]
        for pattern in ['vendor/ultralytics_8_4_100/ultralytics/trackers/*.py','vendor/ultralytics_8_4_100/ultralytics/cfg/trackers/*.yaml','vendor/TrackEval-master/trackeval/metrics/*.py']:
            for p in sorted(root.glob(pattern)):dependency.append(dict(path=p.relative_to(root).as_posix(),sha256=sha256(p)))
        write_json(root/'RUNTIME_SOURCE_HASHES.json',dependency)
        write_json(root/'ENVIRONMENT.json',dict(recorded_at=now(),host=platform.node(),python=sys.version,executable=sys.executable,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,numpy=importlib.metadata.version('numpy'),ultralytics='8.4.100 pinned local source',training_steps=0,transfer='authenticated SCP over wired LAN from Public staging'))
        result=dict(status='passed',input_assets=assets,models=manifest['model_sha256'],identity=identity,run_snapshots=snapshots,
                    full_comparisons=133,video_sequences=19,total_source_frames=sum(i['frames'] for i in summary['invariants']),prediction_observations=sum(i['observations'] for i in summary['invariants']),
                    identity_geometry_count_latency='passed',metric_components='passed',video_decode=videos,html_local_links='passed',html_table_rows=parser.rows,
                    UI_validation='posters/plots/architecture visually inspected; HTML links and table checked; no browser rendering claimed')
        write_json(run.path/'SUMMARY.json',result);write_json(run.path/'DECISION.json',dict(status='engineering_verified',scientific_gate='not_passed'))
        write_json(root/'reversible_delivery_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
        shutil.copy2(run.path/'SUMMARY.json',root/'showcase/evidence/VERIFICATION.json')
        print('VERIFIED',dict(sequences=19,source_frames=result['total_source_frames'],videos=len(videos),decoded_frames=sum(v['decoded_frames'] for v in videos),run_snapshots=len(snapshots)),flush=True)
    meta=read_json(root/'study.json');meta.update(plan_status='fixed_configuration_closed',execution_status='completed',scientific_status='negative_fixed_rollback_configuration',artifact_status='offline_exhibit_verified',transfer_status='ready_to_transfer',updated_at=now(),runs=[read_json(p)['run_id'] for p in sorted((root/'runs').glob('*/run.json'))]);write_json(root/'study.json',meta)
    # Preserve full event-level evidence and all original failures; feature tensors/video sources remain on laptop.
    files=[p for p in root.glob('*') if p.is_file() and p.suffix in {'.json','.md','.log'}]
    for folder in ['runs','annotations','showcase','showcase_versions','code/shared/dense_video_identity']:
        for p in (root/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not ('cache' in p.relative_to(root).parts):files.append(p)
    files=sorted(set(files));receipt=[];dest=root/'delivery_v3.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:
            rel=p.relative_to(root).as_posix();r=dict(path=rel,bytes=p.stat().st_size,sha256=sha256(p));receipt.append(r);z.write(p,rel)
        z.writestr('TRANSFER_MANIFEST.json',json.dumps(receipt,ensure_ascii=False,indent=2))
    result=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha256(dest),files=len(receipt),created_at=now());write_json(root/'DELIVERY_EXPORT.json',result);print('EXPORT',result,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
