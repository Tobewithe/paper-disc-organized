"""Read-only local research viewer; serves only this experiment's explicit assets."""
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
import subprocess
import sys
import uuid
import re
import uvicorn
from .common import read_json,write_json


def create_app(root: Path):
    root=root.resolve()
    app=FastAPI(title="TraceLab · 密集视频个体跟踪",docs_url=None,redoc_url=None)
    current_job={"process":None,"receiving":False}
    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent/"web/index.html")
    @app.get("/api/catalog")
    def catalog():
        p=root/"showcase/catalog.json"
        if not p.exists():raise HTTPException(503,"展示资料正在生成")
        obj=read_json(p)
        for f in sorted((root/"jobs").glob("*/case.json"),reverse=True) if (root/"jobs").exists() else []:
            obj["sequences"].insert(0,read_json(f))
        return obj
    @app.post("/api/jobs")
    async def upload(request:Request,kind:str="pig"):
        if kind not in {"pig","horse"}:raise HTTPException(400,"本原型已配置猪、马两类模型。")
        if current_job["receiving"] or (current_job["process"] is not None and current_job["process"].poll() is None):raise HTTPException(409,"已有视频正在分析，请等待完成。")
        current_job["receiving"]=True
        job_id=uuid.uuid4().hex;folder=root/"jobs"/job_id;folder.mkdir(parents=True)
        try:
            count=0
            with (folder/"input.mp4").open("wb") as f:
                async for chunk in request.stream():
                    count+=len(chunk)
                    if count>200*1024*1024:raise HTTPException(413,"单个视频上限 200 MB；原型最多分析前 60 秒。")
                    f.write(chunk)
            write_json(folder/"job.json",dict(status="queued",phase="视频已接收"))
            with (folder/"process.log").open("w",encoding="utf-8") as log:
                current_job["process"]=subprocess.Popen([sys.executable,"-u","-m","shared.dense_video_identity.single","--root",str(root),"--job",job_id,"--kind",kind],cwd=root/"code",stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            return dict(job_id=job_id,status="queued")
        finally:
            current_job["receiving"]=False
    @app.get("/api/jobs/{job_id}")
    def job_status(job_id:str):
        if not re.fullmatch(r"[a-f0-9]{32}",job_id):raise HTTPException(404)
        p=root/"jobs"/job_id/"job.json"
        if not p.exists():raise HTTPException(404)
        obj=read_json(p)
        if obj["status"] in {"queued","running"} and current_job["process"] is not None and current_job["process"].poll() not in {None,0}:
            obj.update(status="failed",error="分析进程退出；详情见本任务 process.log。")
        return obj
    @app.get("/asset/{relative:path}")
    def asset(relative:str):
        p=(root/relative).resolve()
        if not p.is_relative_to(root) or not p.is_file():raise HTTPException(404)
        if p.relative_to(root).parts[0] not in {"showcase","data","runs","jobs"}:raise HTTPException(403)
        if p.suffix.lower() not in {".json",".mp4",".jpg",".png",".svg",".csv",".md",".pdf"}:raise HTTPException(403)
        return FileResponse(p)
    return app


def serve(root,port=8871):
    uvicorn.run(create_app(root),host="127.0.0.1",port=port,log_level="warning")
