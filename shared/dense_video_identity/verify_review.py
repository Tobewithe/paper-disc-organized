"""Bounded, in-process verification; starts no listening service or SSH tunnel."""
from pathlib import Path
import cv2
from .common import Run,read_json,write_json,sha256,setup_paths,load_gt
from .learn import load_cache,match_gt


def main(root):
    setup_paths(root)
    with Run(root,'review_delivery_verification',dict(network_listener=False,background_server=False,checks=['media_decode','artifact_hashes','ASGI_in_process_routes','frozen_weights','annotation_counts','import_smoke'])) as run:
        manifest=read_json(root/'input_manifest.json')
        for k,p in manifest['models'].items():assert sha256(root/p)==manifest['model_sha256'][k],k
        video_checks=[]
        for v in read_json(root/'showcase/video_catalog.json'):
            p=root/'showcase'/v['path'];assert sha256(p)==v['sha256'];cap=cv2.VideoCapture(str(p));assert cap.isOpened()
            n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert n==v['frames'];assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==1600
            for idx in [0,n//2,n-1]:
                cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=cap.read();assert ok
            cap.release();video_checks.append(dict(path=v['path'],frames=n,decoded_first_mid_last=True))
        annotation=read_json(root/'annotations/field_sparse_identity_v1.json');assert len(annotation['observations'])==144;assert sum(o.get('proposal_index') is not None for o in annotation['observations'])==143
        job='52ae31fa857f4d40a424fd9d5a84dbfa';jobstat=read_json(root/'jobs'/job/'job.json');assert jobstat['status']=='completed' and jobstat['frames']==15
        from fastapi.testclient import TestClient
        from .review_server import create_app
        with TestClient(create_app(root)) as client:
            routes={path:client.get(path).status_code for path in ['/', '/results.json','/architecture.svg','/api/import/'+job,'/api/import/'+job+'/video']}
            assert all(v==200 for v in routes.values()),routes
        seq=next(s for s in manifest['sequences'] if s['name']=='pig_1');records,_=load_cache(root,seq);gt=load_gt(root,seq);maps=match_gt(records,gt)
        ceiling=dict(sequence='pig_1',gt_observations=sum(len(g) for g in gt.values()),raw_detected_matches=sum(len(m) for m in maps.values()),definition='fixed raw detections >=.1; one-to-one GT IoU>=.5; posthoc diagnostic')
        artifacts=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted((root/'showcase').rglob('*')) if p.is_file()]
        write_json(run.path/'SUMMARY.json',dict(status='passed',videos=video_checks,routes=routes,network_listener=False,model_hashes_passed=True,annotation_records=144,usable_annotations=143,import_job=jobstat,detection_ceiling=ceiling,
                   UI_browser_verification='blocked: file URL policy and background server startup review',live_review_service_started=False))
        write_json(root/'SHOWCASE_MANIFEST.json',artifacts);write_json(run.path/'ARTIFACT_MANIFEST.json',artifacts)
        write_json(root/'review_verification_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))
        print('REVIEW_VERIFIED',dict(artifacts=len(artifacts),routes=routes,videos=len(video_checks),detection_ceiling=ceiling),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
