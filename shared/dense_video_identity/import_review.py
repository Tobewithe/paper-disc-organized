"""Run the frozen second-round candidate on a new local upload, without GT or fitting."""
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw
from .common import Run,read_json,write_json,setup_paths,sha256,now
from .render import overlay,font


def main(root,job_id,kind,arm):
    job=root/'jobs'/job_id;status=job/'job.json'
    try:
        setup_paths(root);cfg=read_json(root/'config.json');manifest=read_json(root/'input_manifest.json')
        key='pig' if kind=='pig' else 'horse';weights=root/manifest['models'][key]
        cap=cv2.VideoCapture(str(job/'input.mp4'));assert cap.isOpened(),'Cannot decode video'
        fps=cap.get(cv2.CAP_PROP_FPS);assert 1<=fps<=120
        frames=min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),int(fps*30));assert frames>0
        seq=dict(name='imported',split='demo',model=key,classes=[{'pig':0,'horse':17,'zebra':22}[kind]],video=str(job/'input.mp4'),gt=None,frames=frames,fps=fps,width=int(cap.get(3)),height=int(cap.get(4)),video_sha256=sha256(job/'input.mp4'))
        cap.release();write_json(job/'input_manifest.json',dict(sequences=[seq],models={key:str(weights)},model_sha256={key:sha256(weights)}));write_json(job/'config.json',cfg)
        write_json(job/'study.json',dict(study_id='STUDY_DENSE_VIDEO_ASSOCIATION_V2_20261018',scope='unannotated imported-video demo'))
        ptr=read_json(root/'identity_pointer.json');write_json(job/'identity_pointer.json',dict(ptr,checkpoint=str(root/ptr['checkpoint'])))
        write_json(status,dict(status='running',phase=f'检测与特征提取 · {frames}帧',kind=kind,arm=arm))
        from .infer import extract
        extract(job,cfg['detector'])
        from .learn import load_cache,load_identity,embed_array
        from .trajectory_selector import load_trajectory
        from .conflict_selector import sequence
        identity,ck=load_identity(job);records,x=load_cache(job,seq);z=embed_array(identity,ck,x)
        pred,threshold=load_trajectory(root)
        with Run(job,'import_tracking',dict(arm=arm,baseline='motion',GT=False,new_training=0,parent_model=read_json(root/'trajectory_selector_pointer.json'))) as run:
            write_json(status,dict(status='running',phase='身份关联 · 同视频基线对照',kind=kind,arm=arm))
            left,_=sequence('motion',records,z,cfg);right,events=sequence('net_benefit' if arm=='trajectory' else 'extension',records,z,cfg,pred if arm=='trajectory' else None,threshold)
            write_json(run.path/'baseline_tracks.json',left);write_json(run.path/'candidate_tracks.json',right);write_json(run.path/'decisions.json',events)
            write_json(status,dict(status='running',phase='生成可视化视频',kind=kind,arm=arm))
            cap=cv2.VideoCapture(str(job/'input.mp4'));dest=job/'comparison.mp4';n=max(1,int(frames/fps*15))
            proc=subprocess.Popen([str(root/'tools/ffmpeg.exe'),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r','15','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
            base=Image.new('RGB',(1600,900),'#f2f6fb');d=ImageDraw.Draw(base)
            d.text((40,28),'TraceLab  /  新视频 · 冻结模型推理',font=font(27),fill='#167a8c')
            d.text((40,86),'ByteTrack 基线',font=font(28),fill='#37566f');d.text((822,86),'轨迹收益 CGS-T' if arm=='trajectory' else '保护式补接 PEA',font=font(28),fill='#167a8c')
            d.text((40,662),'新视频没有真实ID标注：本页展示算法行为，不计算准确率。',font=font(28),fill='#607990')
            d.text((40,726),'同一检测、同一身份特征；没有新增训练或人工修改轨迹。',font=font(25),fill='#607990')
            source_idx=-1
            for j in range(n):
                target=min(frames-1,int(j*fps/15))
                while source_idx<target:
                    ok,img=cap.read();source_idx+=1
                    if not ok:raise RuntimeError('Unexpected video end')
                canvas=base.copy();canvas.paste(overlay(img,records[target],left,target,736,414),(40,172));canvas.paste(overlay(img,records[target],right,target,736,414),(822,172))
                ImageDraw.Draw(canvas).text((40,815),f'原帧 {target+1}/{frames} · {j/15:.1f}秒',font=font(24),fill='#42657d');proc.stdin.write(np.asarray(canvas).tobytes())
            proc.stdin.close();assert proc.wait()==0;cap.release()
            write_json(run.path/'SUMMARY.json',dict(frames=frames,seconds=frames/fps,GT=False,video_sha256=sha256(dest),arm=arm))
        write_json(status,dict(status='completed',phase='分析完成',frames=frames,seconds=frames/fps,kind=kind,arm=arm,video='/api/import/'+job_id+'/video',ended_at=now(),run_id=run.id))
    except Exception as exc:
        write_json(status,dict(status='failed',phase='分析失败',error=str(exc),ended_at=now()));raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True);p.add_argument('--kind',choices=['pig','horse','zebra'],required=True);p.add_argument('--arm',choices=['trajectory','extension'],default='trajectory');a=p.parse_args();main(a.root,a.job,a.kind,a.arm)
