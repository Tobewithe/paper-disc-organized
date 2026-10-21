"""Render an offline evidence viewer from unmodified RGB and geometry labels."""
import argparse
import collections
import json
import pathlib
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAMES={'front_occlusion':'前方遮挡','rear_control':'后方通过对照','similar_crossing':'相似个体交叉','partial_occlusion':'局部遮挡','out_of_view':'离开视野','mixed_occlusion_exit':'遮挡后离开视野'}
STATES={'visible':'可见','partially_occluded':'部分遮挡','fully_occluded':'完全遮挡','out_of_view':'出视野'}
PALETTE={0:(15,23,42),1:(45,212,191),2:(251,191,36),100:(244,114,182),900:(51,65,85)}
STATE_COLORS={'visible':'#2dd4bf','partially_occluded':'#fbbf24','fully_occluded':'#f472b6','out_of_view':'#94a3b8'}
FONT=pathlib.Path('C:/Windows/Fonts/msyh.ttc')


def font(size):
    return ImageFont.truetype(str(FONT),size)


def text(draw,xy,s,size=24,color='#e2e8f0'):
    draw.text(xy,str(s),font=font(size),fill=color)


def color_ids(ids):
    out=np.zeros((*ids.shape,3),np.uint8)
    for key,color in PALETTE.items():
        out[ids==key]=color
    return Image.fromarray(out)


def display_frame(folder,data,frame):
    image=Image.new('RGB',(1600,900),'#0b1220')
    draw=ImageDraw.Draw(image)
    name=data['scene_id']
    text(draw,(24,20),'三维生成 · 遮挡状态与关系标签',34)
    text(draw,(24,70),NAMES[name]+'   |   帧 %02d / 23'%frame,25,'#94a3b8')
    text(draw,(1190,28),'生成数据工程验证',24,'#67e8f9')
    text(draw,(1190,65),'尚未训练遮挡模型',22,'#94a3b8')
    with np.load(folder/'labels'/('%04d.npz'%frame)) as arrays:
        ids=arrays['instance_id']
        amodal=arrays['amodal_1']>0
        visible=ids==1
        explain=np.zeros((*ids.shape,3),np.uint8)
        explain[:]=[15,23,42]
        explain[amodal]=[80,90,110]
        explain[amodal&visible]=PALETTE[1]
        for owner in [2,100,900]:
            explain[amodal&(ids==owner)]=PALETTE[owner]
    panels=[Image.open(folder/'rgb'/('%04d.png'%frame)).convert('RGB'),color_ids(ids),Image.fromarray(explain)]
    titles=['生成画面','可见实例真值：青色 ID1 / 黄色 ID2','ID1 完整投影中的可见与被挡部分']
    for x,panel,title in zip([16,544,1072],panels,titles):
        text(draw,(x,120),title,21)
        image.paste(panel,(x,158))
    records={r['track_id']:r for r in data['records'] if r['frame']==frame}
    for track,x in [(1,24),(2,550)]:
        r=records[track]
        text(draw,(x,515),'个体 ID%d'%track,28,PALETTE[track])
        text(draw,(x,562),STATES[r['state']],32,STATE_COLORS[r['state']])
        occ='—' if r['occlusion_fraction'] is None else '%.1f%%'%(100*r['occlusion_fraction'])
        trunc='—' if r['truncation_fraction'] is None else '%.1f%%'%(100*r['truncation_fraction'])
        text(draw,(x,620),'外部遮挡比例   '+occ,24)
        text(draw,(x,660),'视野截断比例   '+trunc,24)
        parts=[]
        for k,v in r['occluder_pixels'].items():
            label={'100':'挡板','900':'地面','1':'ID1','2':'ID2'}.get(k,k)
            parts.append(label+' '+str(v)+'像素')
        text(draw,(x,704),'遮挡者：'+(' / '.join(parts) if parts else '无'),20,'#94a3b8')
    text(draw,(1080,515),'ID1 状态变化',27)
    for r in [r for r in data['records'] if r['track_id']==1]:
        x=1080+r['frame']*20
        draw.rectangle((x,565,x+17,605),fill=STATE_COLORS[r['state']])
    x=1080+frame*20
    draw.rectangle((x-2,560,x+19,610),outline='white',width=2)
    text(draw,(1080,630),'青：可见   黄：部分遮挡',22)
    text(draw,(1080,668),'粉：完全遮挡   灰：出视野',22)
    text(draw,(24,810),'遮挡比例在原视野内计算；视野截断单独计算。遮挡与出画可以同时发生。',25,'#cbd5e1')
    return image


def make_curves(dataset,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fp=FontProperties(fname=str(FONT))
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True,sharey=True)
    for ax,(name,title) in zip(axes.flat,NAMES.items()):
        j=json.loads((dataset/name/'annotations.json').read_text())
        r=[x for x in j['records'] if x['track_id']==1]
        occ=[np.nan if x['occlusion_fraction'] is None else 100*x['occlusion_fraction'] for x in r]
        trunc=[100*x['truncation_fraction'] for x in r]
        ax.plot(range(24),occ,color='#0f766e',lw=2.5,label='外部遮挡')
        ax.plot(range(24),trunc,color='#be185d',lw=2.5,ls='--',label='视野截断')
        ax.set_title(title,fontproperties=fp,fontsize=14)
        ax.set_ylim(-3,103)
        ax.grid(alpha=.2)
        ax.set_ylabel('比例 / %',fontproperties=fp)
        ax.set_xlabel('帧',fontproperties=fp)
    axes[0,0].legend(prop=fp)
    fig.suptitle('个体 ID1：遮挡与出视野的分解（几何真值）',fontproperties=fp,fontsize=19)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(output/'visibility_curves.png',dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--reuse-media',action='store_true')
    args=p.parse_args()
    dataset=pathlib.Path(args.dataset)
    output=pathlib.Path(args.output)
    (output/'videos').mkdir(parents=True,exist_ok=True)
    ffmpeg=shutil.which('ffmpeg')
    assert ffmpeg
    overview=Image.new('RGB',(1600,1320),'#0b1220')
    draw=ImageDraw.Draw(overview)
    text(draw,(25,18),'首批三维生成场景：6段 / 144帧 / 288个体观测',34)
    text(draw,(25,72),'完整遮挡、局部遮挡、相似个体交叉与出画；标签来自三维几何。',25,'#94a3b8')
    cards=[]
    all_counts=collections.Counter()
    for index,(name,title) in enumerate(NAMES.items()):
        folder=dataset/name
        data=json.loads((folder/'annotations.json').read_text())
        counts=collections.Counter(r['state'] for r in data['records'])
        all_counts.update(counts)
        targets=[r for r in data['records'] if r['track_id']==1]
        selected=max(targets,key=lambda r: (r['state']=='fully_occluded',r['occlusion_fraction'] or 0))['frame']
        if name=='out_of_view':
            selected=23
        raw=Image.open(folder/'rgb'/('%04d.png'%selected)).convert('RGB').resize((720,450))
        x=25+(index%2)*790
        y=140+(index//2)*385
        # Keep aspect ratio while leaving room for the descriptive state line.
        raw=raw.resize((560,350))
        overview.paste(raw,(x,y+35))
        text(draw,(x,y),title,25)
        state=next(r['state'] for r in targets if r['frame']==selected)
        text(draw,(x+575,y+68),'查询 ID1',21,PALETTE[1])
        text(draw,(x+575,y+105),STATES[state],24,STATE_COLORS[state])
        text(draw,(x+575,y+150),'帧 %02d'%selected,21,'#94a3b8')
        if not (args.reuse_media and (output/'videos'/(name+'.mp4')).exists()):
            writer=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r','6','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'videos'/(name+'.mp4'))],stdin=subprocess.PIPE)
            try:
                for frame in range(24):
                    panel=display_frame(folder,data,frame)
                    writer.stdin.write(panel.tobytes())
                    if name=='front_occlusion' and frame==12:
                        panel.save(output/'labels_example.png')
                writer.stdin.close()
                assert writer.wait()==0
            except Exception:
                writer.kill()
                raise
        shutil.copy2(folder/'annotations.json',output/(name+'_annotations.json'))
        cards.append({'name':name,'title':title,'state_counts':dict(counts)})
    overview.save(output/'scene_overview.png')
    make_curves(dataset,output)
    (output/'SHOWCASE_DATA.json').write_text(json.dumps({'scenes':cards,'state_counts':dict(all_counts),'training_steps':0},ensure_ascii=False,indent=2),encoding='utf-8')
    tiles=''.join('<article><h2>'+c['title']+'</h2><video controls preload="metadata" src="videos/'+c['name']+'.mp4"></video><p><a href="'+c['name']+'_annotations.json">逐帧标签 JSON</a></p></article>' for c in cards)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三维遮挡场景与标签验证</title><style>body{margin:0;background:#0b1220;color:#e2e8f0;font:17px/1.6 "Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:32px}h1{font-size:32px}h2{font-size:22px}a{color:#67e8f9}p{color:#cbd5e1}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:22px}article{background:#152033;padding:18px;border-radius:14px}video,img{width:100%;border-radius:8px}.note{padding:18px;background:#153b42;border-left:4px solid #2dd4bf}.metrics{display:flex;gap:30px;flex-wrap:wrap;font-size:23px;margin:24px 0}</style><main><h1>三维遮挡场景生成与自动标签</h1><p>Blender 4.2.1 / Cycles OptiX · RTX A6000 · Slurm 单卡有限任务</p><div class="metrics"><b>6 段场景</b><b>144 帧</b><b>288 个体观测</b><b>6,912 个几何抽查点</b></div><p class="note">本页展示生成数据与几何标签的工程验证。素材为程序化四足体，遮挡模型尚未训练，真实视频上的改善尚未验证。视频按逐原帧 6fps 慢放，生成帧率为 12fps。</p><img src="labels_example.png" alt="遮挡标签解释"><h2>完整序列与逐帧标注</h2><div class="grid">'''+tiles+'''</div><h2>遮挡与出视野分解</h2><img src="visibility_curves.png" alt="几何标签比例曲线"><p>完整投影来自目标独立几何查询；遮挡率在原视野内计算，出画比例单独记录。身份、完整掩码与真深度只作为监督数据。</p><p><a href="REPORT.md">实验报告</a> · <a href="TOOL_SELECTION.md">工具选型</a> · <a href="SHOWCASE_DATA.json">汇总数据</a> · <a href="annotations_and_mot.zip">标准标签下载</a> · <a href="使用说明.md">使用说明</a></p></main></html>'''
    (output/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps({'output':str(output),'state_counts':dict(all_counts),'videos':6},ensure_ascii=False))


if __name__=='__main__':
    main()
