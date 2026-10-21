from pathlib import Path
from fastapi import FastAPI,Request,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import subprocess,sys,uuid,re
from .common import read_json,write_json


def create_app(root):
    root=root.resolve();app=FastAPI(docs_url=None,redoc_url=None);state={'process':None,'receiving':False}
    @app.post('/api/import')
    async def upload(request:Request,kind:str='pig',arm:str='trajectory'):
        if kind not in {'pig','horse','zebra'} or arm not in {'trajectory','extension'}:raise HTTPException(400,'Unsupported configuration')
        if state['receiving'] or (state['process'] and state['process'].poll() is None):raise HTTPException(409,'已有视频正在分析，请稍候。')
        state['receiving']=True;job_id=uuid.uuid4().hex;folder=root/'jobs'/job_id;folder.mkdir(parents=True)
        try:
            count=0
            with (folder/'input.mp4').open('wb') as f:
                async for chunk in request.stream():
                    count+=len(chunk)
                    if count>200*1024*1024:raise HTTPException(413,'上限200MB，最多分析前30秒。')
                    f.write(chunk)
            write_json(folder/'job.json',dict(status='queued',phase='已接收视频'))
            with (folder/'process.log').open('w',encoding='utf-8') as log:
                state['process']=subprocess.Popen([sys.executable,'-u','-m','shared.dense_video_identity.import_review','--root',str(root),'--job',job_id,'--kind',kind,'--arm',arm],cwd=root/'code',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            return dict(job_id=job_id)
        finally:state['receiving']=False
    def job_folder(job_id):
        if not re.fullmatch('[a-f0-9]{32}',job_id):raise HTTPException(404)
        return root/'jobs'/job_id
    @app.get('/api/import/{job_id}')
    def job_status(job_id:str):
        p=job_folder(job_id)/'job.json'
        if not p.exists():raise HTTPException(404)
        return read_json(p)
    @app.get('/api/import/{job_id}/video')
    def video(job_id:str):
        p=job_folder(job_id)/'comparison.mp4'
        if not p.exists():raise HTTPException(404)
        return FileResponse(p)
    app.mount('/',StaticFiles(directory=root/'showcase',html=True),name='review');return app

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8872);a=p.parse_args()
    uvicorn.run(create_app(a.root),host='127.0.0.1',port=a.port,log_level='warning')
