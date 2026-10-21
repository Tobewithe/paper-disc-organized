import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import traceback
import uuid

root = pathlib.Path('/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion')
study = root / 'experiments/dense_video_synthetic_occlusion_20261019'
run_id = 'RUN_RESOURCE_PREFLIGHT_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_') + uuid.uuid4().hex[:8]
run = study / 'runs' / run_id
run.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
record = {'study_id':'STUDY_DENSE_VIDEO_SYNTHETIC_OCCLUSION_20261019','run_id':run_id,'stage_id':'resource_preflight','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'executor':platform.node(),'user':os.environ.get('USER'),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'command':sys.argv,'code_commit':'bc0507c5400b61935fe93f981a4ff5026964c271','source_sha256':sha(__file__),'protocol_sha256':sha(study/'PROTOCOL.md'),'execution_status':'running','training_steps':0}
(run/'run.json').write_text(json.dumps(record,indent=2))
(run/'PROTOCOL.md').write_bytes((study/'PROTOCOL.md').read_bytes())
(run/'server_preflight.py').write_bytes(pathlib.Path(__file__).read_bytes())
try:
    import torch
    assert torch.cuda.is_available(), 'Slurm GPU is not visible to PyTorch'
    assert torch.cuda.device_count()==1, 'Preflight expects exactly one allocated GPU'
    x=torch.arange(256*256,device='cuda',dtype=torch.float32).reshape(256,256)/65536
    y=x@x.T
    torch.cuda.synchronize()
    assert bool(torch.isfinite(y).all())
    smi=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True)
    result={'torch':torch.__version__,'torch_cuda':torch.version.cuda,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'cuda_device_count':torch.cuda.device_count(),'allocated_device_name':torch.cuda.get_device_name(0),'device_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'finite_matmul':True,'result_sum':float(y.sum()),'nvidia_smi':smi.stdout,'allocated_gpus':os.environ.get('SLURM_JOB_GPUS'),'cpu_count':os.environ.get('SLURM_CPUS_PER_TASK')}
    (run/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    (run/'DECISION.json').write_text(json.dumps({'status':'passed','scope':'GPU access only; no renderer or model validation','next_action':'install isolated renderer and run 2-frame smoke'},indent=2))
    record.update(execution_status='completed',exit_code=0)
except Exception:
    (run/'ERROR.txt').write_text(traceback.format_exc())
    record.update(execution_status='failed',exit_code=1)
finally:
    record['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (run/'run.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)
    if (run/'SUMMARY.json').exists(): print((run/'SUMMARY.json').read_text(),flush=True)
sys.exit(record['exit_code'])
