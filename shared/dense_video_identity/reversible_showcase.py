"""Offline scientific exhibit, plots and paired videos from immutable real outputs."""
from pathlib import Path
import html,subprocess,shutil
import numpy as np
import cv2
from PIL import Image,ImageDraw
from .common import Run,read_json,write_json,load_gt,sha256,iou_matrix
from .render import font,color
from .learn import load_cache
from .reversible_adjudicate import event_matches

LABELS={'bytetrack':'ByteTrack','immediate':'立即补接 I','confirm':'时序确认 C','reversible':'确认 + 回退 M','botsort':'BoT-SORT','deepocsort':'Deep OC-SORT','fasttrack':'FastTracker'}
COLORS={'bytetrack':'#71839b','immediate':'#4077c9','confirm':'#139c91','reversible':'#d28b39','botsort':'#233954','deepocsort':'#9f80b6','fasttrack':'#9aafb9'}

def plots(root,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':13,'savefig.facecolor':'#f3f6fa'})
    dest=root/'showcase';dest.mkdir(exist_ok=True)
    arms=list(LABELS);fig,ax=plt.subplots(figsize=(12,7),facecolor='#f3f6fa');ax.set_facecolor('#f3f6fa');vals=[data['splits']['independent']['combined'][a]['IDF1'] for a in arms]
    ax.barh(range(7),vals,color=[COLORS[a] for a in arms],height=.62)
    ax.set_yticks(range(7),[LABELS[a] for a in arms]);ax.invert_yaxis();ax.set_xlim(0,80);ax.set_xlabel('完整视频 IDF1（越高越好，0–100）');ax.set_title('新确认视频：补接有收益，回退的额外收益未成立',loc='left',fontweight='bold',pad=24)
    for j,v in enumerate(vals):ax.text(v+.8,j,f'{v:.2f}',va='center',fontweight='bold')
    for s in ['top','right','left']:ax.spines[s].set_visible(False)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.text(.16,.02,'zebra_5 / duck_1 / duck_2 全序列 · 按 IDTP/IDFP/IDFN 合并 · 同前端输入',fontsize=11,color='#56677a');fig.tight_layout(rect=(0,.05,1,1));fig.savefig(dest/'new_videos_IDF1.png',dpi=160);fig.savefig(dest/'new_videos_IDF1.svg');plt.close(fig)
    rows=data['splits']['retrospective']['by_sequence']+data['splits']['independent']['by_sequence'];names=sorted({x['sequence'] for x in rows});idx={(x['sequence'],x['arm']):x for x in rows};a=['immediate','confirm','reversible']
    delta=np.array([[idx[n,k]['IDF1']-idx[n,'bytetrack']['IDF1'] for k in a] for n in names]);fig,ax=plt.subplots(figsize=(10,9),facecolor='#f3f6fa')
    ax.imshow(delta,cmap='RdYlGn',vmin=-6,vmax=6,aspect='auto');ax.set_xticks(range(3),[LABELS[k] for k in a]);ax.set_yticks(range(len(names)),[n+('  [新确认]' if idx[n,'bytetrack']['split']=='independent' else '') for n in names]);ax.tick_params(length=0)
    for j in range(len(names)):
        for k in range(3):ax.text(k,j,f'{delta[j,k]:+.2f}',ha='center',va='center',color='#172b43',fontsize=12)
    ax.set_title('逐视频完整结果：相对 ByteTrack 的 IDF1 变化',loc='left',pad=20,fontweight='bold');fig.tight_layout();fig.savefig(dest/'all_video_deltas.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,6),facecolor='#f3f6fa')
    for ax,key,title in zip(axs,['wrong','correct'],['错误补接输出减少','正确补接也被削减']):
        vals=[data['audit_totals']['independent'][a]['published_frames'][key] for a in ['confirm','reversible']];ax.set_facecolor('#f3f6fa');ax.bar(['仅确认 C','确认 + 回退 M'],vals,color=['#139c91','#d28b39'],width=.55);ax.set_title(title,loc='left',pad=22,fontweight='bold');ax.set_ylabel('已发布的补接观测帧数')
        for j,v in enumerate(vals):ax.text(j,v+12,str(v),ha='center',fontsize=21,fontweight='bold')
        ax.set_ylim(0,max(vals)*1.25)
        for s in ['top','right']:ax.spines[s].set_visible(False)
    fig.text(.07,.025,'同一固定检测与运动轨迹 · 以补接前旧轨迹的最后可匹配 GT 为参照 · 未知身份单独保留',fontsize=11,color='#56677a');fig.tight_layout(rect=(0,.08,1,1));fig.savefig(dest/'rollback_tradeoff.png',dpi=160);plt.close(fig)

def pane(img,record,frame,crop,width,height,target_box=None,focus_identity=None):
    x0,y0,x1,y1=crop;cut=cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_BGR2RGB);scale=min(width/cut.shape[1],height/cut.shape[0]);w,h=round(cut.shape[1]*scale),round(cut.shape[0]*scale)
    canvas=Image.new('RGB',(width,height),'#e7edf3');canvas.paste(Image.fromarray(cv2.resize(cut,(w,h))),((width-w)//2,(height-h)//2));d=ImageDraw.Draw(canvas);ox=(width-w)//2;oy=(height-h)//2
    target=[]
    for t in frame['tracks']:
        b=t['box']
        if b[2]<x0 or b[0]>x1 or b[3]<y0 or b[1]>y1:continue
        focused=focus_identity is not None and t['identity']==focus_identity
        xy=[ox+(b[0]-x0)*scale,oy+(b[1]-y0)*scale,ox+(b[2]-x0)*scale,oy+(b[3]-y0)*scale];c=color(t['identity']);lw=4 if focused else 1
        # The detector polygon is shown, but never an invented mask for hidden frames.
        p=np.asarray(record['polygons'][t['det_id']])
        if len(p)>=3:
            pts=[(ox+(xx-x0)*scale,oy+(yy-y0)*scale) for xx,yy in p];d.line(pts+[pts[0]],fill=c,width=lw)
        if focused:
            xx=max(3,min(width-120,xy[0]));yy=max(1,min(height-30,xy[1]-29));d.rounded_rectangle((xx,yy,xx+118,yy+29),radius=4,fill='#12263a');d.text((xx+7,yy+2),f"ID {t['identity']}",font=font(20),fill=c);target.append(t['identity'])
    if target_box is not None:
        b=target_box;xy=[ox+(b[0]-x0)*scale,oy+(b[1]-y0)*scale,ox+(b[2]-x0)*scale,oy+(b[3]-y0)*scale];d.rectangle(xy,outline='white',width=2)
    return canvas,target

def videos(root,data):
    seqs={s['name']:s for s in read_json(root/'input_manifest.json')['sequences']};out=root/'showcase/videos';out.mkdir(parents=True,exist_ok=True)
    cases=[
        dict(key='01_reconnect',name='duck_2',gt=5,focus_base=90,start=25,end=65,title='接回同一个体：身份补接的实际收益',note='仅确认与完整方案都接回原 ID；这一例不证明回退带来额外收益。',event=39),
        dict(key='02_stop_wrong_link',name='duck_2',gt=9,focus_base=98,start=72,end=106,title='及时止损：停止错用另一个体的旧 ID',note='G9 曾沿用 G10 的历史身份；回退停止错接，但不等于找回 G9 原 ID。',event=88),
        dict(key='03_false_rollback',name='zebra_5',gt=11,focus_base=30,start=78,end=124,title='代价：相似度下降，也可能还是同一个体',note='GT 仍为 G11；仅确认保留补接，完整方案撤销了这条正确连接。',event=107),
        dict(key='04_canopy_occlusion',name='pig_4',gt=8,focus_base=None,start=55,end=106,title='树冠遮挡：先重新看见，才能判断是不是原个体',note='真实物理遮挡案例；完全不可见区间不确定，未计入完整遮挡恢复。',event=86),
    ];catalog=[]
    for case in cases:
        seq=seqs[case['name']];split=seq['split'];ptr=read_json(root/('reversible_'+split+'_pointer.json'));folder=root/ptr['path'];arms=['bytetrack','confirm','reversible']
        tracks={a:read_json(folder/a/case['name']/'tracks.json') for a in arms};events=read_json(folder/'reversible'/case['name']/'events.json');gt=load_gt(root,seq);records,_=load_cache(root,seq)
        focus={a:event_matches(gt,tracks[a])[0] for a in arms}
        boxes=[r[2:6] for f in range(case['start'],case['end']+1) for r in gt[f] if int(r[1])==case['gt']]
        cap=cv2.VideoCapture(seq['video']);vw,vh=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));bb=np.asarray(boxes);lo=bb[:,:2].min(0);hi=bb[:,2:].max(0);pad=np.maximum((hi-lo)*.45,[80,65]);lo=np.maximum(lo-pad,[0,0]);hi=np.minimum(hi+pad,[vw,vh]);crop=list(map(int,[*lo,*hi]));cap.set(cv2.CAP_PROP_POS_FRAMES,case['start']-1)
        dest=out/(case['key']+'.mp4');fps=5;n=case['end']-case['start']+1
        proc=subprocess.Popen([str(root/'tools/ffmpeg.exe'),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
        for f in range(case['start'],case['end']+1):
            ok,img=cap.read();assert ok;canvas=Image.new('RGB',(1600,900),'#f2f5fa');d=ImageDraw.Draw(canvas)
            d.text((35,24),'TRACELAB  /  可撤销身份补接实验',font=font(22),fill='#2b69ad');d.text((35,66),case['title'],font=font(35),fill='#142b43');d.text((35,121),case['note'],font=font(22),fill='#5c7084')
            g=[r for r in gt[f] if int(r[1])==case['gt']];box=g[0][2:6] if g else None
            ids=[]
            for j,a in enumerate(arms):
                x=35+j*520;d.rounded_rectangle((x,177,x+498,660),radius=12,fill='white');d.text((x+15,190),LABELS[a],font=font(23),fill=COLORS[a]);panel,pids=pane(img,records[f-1],tracks[a][f-1],crop,498,375,box,focus[a].get((f,case['gt'])));canvas.paste(panel,(x,235));ids.append(pids)
                text='匹配 ID '+','.join(map(str,pids)) if pids else '当前无可靠匹配' if box is not None else '当前 GT 未标：不推定身份'
                d.text((x+15,623),text,font=font(19),fill='#263f58')
            published=min(f+4,seq['frames']);d.rounded_rectangle((35,689,1573,819),radius=14,fill='#17314c');d.text((56,705),f"原视频 {case['name']}  ·  参照个体 G{case['gt']}  ·  原帧 {f}  /  发布时刻 {published}",font=font(23),fill='white')
            ev=[e for e in events if e['frame']<=published and e['base_id']==case['focus_base']]
            relevant=[e for e in ev if e['type'] in {'confirmed','rollback'}]
            status=('已确认补接' if relevant[-1]['type']=='confirmed' else '已撤销补接')+f"：运动轨迹 {relevant[-1]['base_id']}" if relevant else '按冻结输出逐帧回放；已发布历史不回改'
            d.text((56,753),status,font=font(24),fill='#8bd6c8');d.text((35,847),f"逐原帧慢放 5 fps（原视频 {seq['fps']:g} fps）  |  白框：官方 GT 参照；彩色轮廓 / ID：算法输出",font=font(20),fill='#5c7084')
            proc.stdin.write(np.asarray(canvas).tobytes())
            if f==case['event']:canvas.save(out/(case['key']+'.jpg'),quality=94)
        proc.stdin.close();assert proc.wait()==0;cap.release();catalog.append(dict(**case,path='videos/'+dest.name,poster='videos/'+case['key']+'.jpg',frames=n,fps=fps,source_fps=seq['fps'],crop=crop,source_run=ptr['run_id'],sha256=sha256(dest),selection='posthoc illustrative case; all full-video scores separately reported'))
        print('EXHIBIT_VIDEO',case['key'],n,flush=True)
    write_json(root/'showcase/video_catalog.json',catalog);return catalog

def architecture(root):
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="700" viewBox="0 0 1500 700"><rect width="1500" height="700" fill="#f3f6fa"/><style>text{font-family:Microsoft YaHei,sans-serif;fill:#19344f}.h{font-size:30px;font-weight:700}.t{font-size:22px}.n{font-size:18px;fill:#597086}</style><defs><marker id="a" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8" fill="#5c7896"/></marker></defs><text x="55" y="65" class="h">固定运动轨迹 + 可撤销身份账本</text><text x="55" y="105" class="n">本轮待验证设计；补接有局部收益，时序确认与回退的独立增量尚未成立</text>'''
    boxes=[(55,180,250,125,'检测 / 分割前端','YOLO26m-seg · 权重冻结'),(370,180,250,125,'运动轨迹','ByteTrack · 框与观测固定'),(685,180,250,125,'输出缓冲','最多 4 帧 · 历史不可改'),(1000,180,440,125,'发布个体 ID + 轮廓','同一观测数 · 可追溯身份变更'),(370,400,250,130,'历史身份锚点','近 5 次特征 · 64D 身份头'),(685,400,250,130,'提议 → 多帧确认','3 次证据 · 不确定则保留原 ID'),(1000,400,440,130,'监测矛盾 → 撤销别名','回到原运动 ID；不污染运动状态')]
    boxes=[(x,y,w,h,title,'3次确认 · 超时用原ID' if title=='提议 → 多帧确认' else note) for x,y,w,h,title,note in boxes]
    for x,y,w,h,title,note in boxes:svg+=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="white" stroke="#cedae7" stroke-width="2"/><text x="{x+18}" y="{y+45}" class="t" font-weight="700">{title}</text><text x="{x+18}" y="{y+84}" class="n">{note}</text>'
    for path in ['M305 240 H365','M620 240 H680','M935 240 H995','M495 305 V395','M620 465 H680','M935 465 H995','M810 400 V310','M1220 400 V310']:svg+=f'<path d="{path}" fill="none" stroke="#5c7896" stroke-width="3" marker-end="url(#a)"/>'
    svg+='<text x="55" y="615" class="t">本轮可检验贡献：将身份接回做成有记录、可撤销的决策，并量化纠错与误撤销的代价。</text><text x="55" y="656" class="n">GT 仅用于事后评价与原图复核；不进入推理。零新增训练参数。一般确认 / 回退已有先例，不宣称首次。</text></svg>'
    (root/'showcase/architecture.svg').write_text(svg,encoding='utf-8')
    canvas=Image.new('RGB',(1500,700),'#f3f6fa');d=ImageDraw.Draw(canvas)
    d.text((55,29),'固定运动轨迹 + 可撤销身份账本',font=font(30),fill='#19344f')
    d.text((55,79),'本轮待验证设计；补接有局部收益，时序确认与回退的独立增量尚未成立',font=font(18),fill='#597086')
    for x,y,w,h,title,note in boxes:
        d.rounded_rectangle((x,y,x+w,y+h),radius=16,fill='white',outline='#cedae7',width=2);d.text((x+18,y+20),title,font=font(22),fill='#19344f');d.text((x+18,y+62),note,font=font(18),fill='#597086')
    for a,b in [((305,240),(365,240)),((620,240),(680,240)),((935,240),(995,240)),((495,305),(495,395)),((620,465),(680,465)),((935,465),(995,465)),((810,400),(810,310)),((1220,400),(1220,310))]:
        d.line((a,b),fill='#5c7896',width=3)
        if b[0]>a[0]:pts=[b,(b[0]-10,b[1]-5),(b[0]-10,b[1]+5)]
        else:sign=1 if b[1]>a[1] else -1;pts=[b,(b[0]-5,b[1]-10*sign),(b[0]+5,b[1]-10*sign)]
        d.polygon(pts,fill='#5c7896')
    d.text((55,589),'本轮可检验贡献：将身份接回做成有记录、可撤销的决策，并量化纠错与误撤销的代价。',font=font(22),fill='#19344f')
    d.text((55,634),'GT 仅用于事后评价与原图复核；不进入推理。零新增训练参数。一般确认 / 回退已有先例，不宣称首次。',font=font(18),fill='#597086');canvas.save(root/'showcase/architecture.png')

def webpage(root,data,catalog):
    rows=data['splits']['independent']['by_sequence'];allrows=sum([s['by_sequence'] for s in data['splits'].values()],[])
    cards=''.join(f'<article class="case"><span>{i+1:02d} / 真实输出回放</span><h3>{html.escape(v["title"])}</h3><p>{html.escape(v["note"])}</p><video controls preload="metadata" poster="{v["poster"]}" src="{v["path"]}"></video><p class="small">{v["name"]} · 原帧 {v["start"]}–{v["end"]} · 慢放5fps · 案例为事后选择，完整结果见下表</p></article>' for i,v in enumerate(catalog))
    table=''.join(f'<tr data-split="{r["split"]}" data-arm="{r["arm"]}"><td>{r["sequence"]}</td><td>{LABELS[r["arm"]]}</td><td>{r["IDF1"]:.2f}</td><td>{r["HOTA"]:.2f}</td><td>{r["IDSW"]}</td><td>{r["confirmed"]}</td><td>{r["rollbacks"]}</td></tr>' for r in allrows)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>TraceLab · 身份补接的收益与代价</title><style>*{box-sizing:border-box}body{margin:0;background:#eef3f8;color:#17314c;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1280px;margin:auto;padding:48px 30px 70px}nav{display:flex;gap:24px;align-items:center;color:#5d738b;font-size:14px}nav b{color:#146f9b;letter-spacing:3px;margin-right:auto}a{color:#2366a3}h1{font-size:50px;line-height:1.2;letter-spacing:-1px;margin:32px 0 20px;max-width:950px}h2{font-size:31px;margin:0 0 16px}h3{font-size:22px;margin:8px 0}p{color:#576d84;margin:10px 0}.lead{font-size:20px;max-width:960px}.badge{display:inline-block;border:1px solid #caa56a;color:#88581b;border-radius:20px;padding:5px 14px;margin-top:22px;background:#fff5df}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin:28px 0 42px}.stat{background:white;border-radius:16px;padding:23px}.stat strong{display:block;font-size:36px;color:#146e86}.stat span{font-size:14px;color:#6b8094}.section{padding:35px;background:white;border-radius:22px;margin:24px 0}.section>img{width:100%;border-radius:10px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.case{background:#f3f6fa;padding:22px;border-radius:16px}.case span{font-size:13px;color:#528b9e;letter-spacing:1px}.case video{width:100%;border-radius:10px;margin-top:15px;background:#182c43}.small{font-size:13px}.notice{border-left:4px solid #c58b48;padding:15px 20px;background:#fff5e6;color:#72562e}.filters{display:flex;gap:16px;margin:20px 0}select{padding:10px 14px;border:1px solid #cfdae7;border-radius:8px;background:white;color:#17314c;font-size:16px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:11px 14px;border-bottom:1px solid #e2e9f1;text-align:left}th{position:sticky;top:0;background:#f0f5fa}.scroll{max-height:480px;overflow:auto}.foot{font-size:13px;color:#6e8295;margin-top:24px}@media(max-width:800px){h1{font-size:34px}.grid,.stats{grid-template-columns:1fr}.section{padding:22px}main{padding:25px 16px}}</style><main><nav><b>TRACELAB</b><a href="#cases">案例回放</a><a href="#results">完整成绩</a><a href="#method">模型结构</a><a href="REPORT.md">实验报告</a></nav><span class="badge">第三轮实验 · 28358lan 笔记本实测</span><h1>接回个体身份，<br>也要看清撤销的代价。</h1><p class="lead">已完成19段视频、7组对照。身份补接存在实际收益；当前“确认 + 回退”方案还不能稳定超过基线，回退也会拆断正确连接。</p><div class="stats"><div class="stat"><strong>+1.13</strong><span>新视频 M 相对 ByteTrack 的 IDF1</span></div><div class="stat"><strong>−1.46</strong><span>新视频 M 相对 BoT-SORT 的 IDF1</span></div><div class="stat"><strong>64 / 16</strong><span>间断事件初筛 / 逐帧深入复核</span></div><div class="stat"><strong>0 参数</strong><span>新增训练参数 · 最多4帧发布延迟</span></div></div><section class="section"><h2>有效的部分，和仍未成立的主张</h2><p>新确认集：ByteTrack 65.51 → 仅确认 C 68.63 → 完整 M 66.64；BoT-SORT 为68.10。M 相对 ByteTrack 的视频簇95%区间为 [−0.59, +2.54]，未通过稳定改善门槛。</p><img src="new_videos_IDF1.png"><div class="notice">C / I 是预先指定的机制对照。其收益不能记作新增回退模块的贡献，也不能在看到结果后替换本轮主方法。</div></section><section class="section" id="cases"><h2>四段可核查的实验回放</h2><p>每幅画面都来自原始视频和已冻结输出，未手工修正 ID。颜色是预测身份，白框是官方标注参照。画面按原帧回放，并显示统一4帧缓冲后的发布时刻。</p><div class="grid">'''+cards+'''</div></section><section class="section"><h2>为什么回退没有带来额外改善？</h2><p>在可判定的已提交撤销中，9次仍属于同一GT个体，4次为不同GT，另7次无法判定。相似度变低并不充分说明身份接错。</p><img src="rollback_tradeoff.png"><p class="small">计数是观测帧，不是独立事件数。归因依据固定检测与GT的事后匹配，未知身份独立保留；不构成独立人工金标准。</p></section><section class="section"><h2>真实遮挡：已补证据，也保留缺口</h2><p>64个候选事件已初筛；16个事件复核42页连续帧，并补看43张原分辨率裁剪。11个物理遮挡事件保留165条前后可见标签（沿用并复核官方框），其中132个后窗口标签参与补充比较。</p><p>严格完整遮挡的可靠事件为0；没有把GT漏标、部分可见和出画面当作完全遮挡。11例部分遮挡的“三帧前后同ID”严格成功均为0/11。ByteTrack / C / M 的后窗口仅41/132帧有可靠匹配，前端可观测性仍是瓶颈。</p><div class="notice">这一轮尚未完成可用于验证“完全遮挡后恢复”的标注集。当前新增标签是AI复核事件层，几何框来源为官方GT；不能冒充独立人工逐帧标注。</div><p><a href="annotations/occlusion_labels.json">查看冻结标签与逐事件理由</a> · <a href="annotations/ANNOTATION_FREEZE.json">标注冻结记录</a></p></section><section class="section" id="method"><h2>被检验的模型设计</h2><img src="architecture.svg"><p>研究重点是让身份修改独立于运动轨迹，保留提议、确认、撤销的证据。一般确认和回退已有先例；本轮验证的复合机制尚未证明优于已有方法。</p></section><section class="section" id="results"><h2>全序列结果，不只挑成功画面</h2><img src="all_video_deltas.png"><div class="filters"><select id="split"><option value="independent">新确认 · 3视频</option><option value="retrospective">已见复核 · 12视频</option><option value="engineering">工程 / 原训练 · 4视频</option></select><select id="arm"><option value="all">全部7组对照</option>'''+''.join(f'<option value="{a}">{l}</option>' for a,l in LABELS.items())+'''</select></div><div class="scroll"><table><thead><tr><th>视频</th><th>方法</th><th>IDF1 ↑</th><th>HOTA ↑</th><th>IDSW ↓</th><th>确认次数</th><th>撤销次数*</th></tr></thead><tbody>'''+table+'''</tbody></table></div><p class="small">* 此列包含提议期撤销与已提交撤销，不能直接作为成功纠错数。详细类型在事后审计中分列。工程组不作为泛化证明。</p></section><section class="section"><h2>研究决策</h2><p>关闭当前固定回退配置，保留“固定运动轨迹上的身份补接”作为有实测收益的基础。优先研究如何区分错误身份与同一个体的姿态 / 视角变化：撤销需要身份冲突的反证，不能只依赖自相似下降。新的阈值搜索或训练尚未启动。</p><p><a href="evidence/SUMMARY.json">完整数值与冻结门槛</a> · <a href="evidence/link_audit.json">逐条补接 / 撤销审计</a> · <a href="evidence/partial_events.json">部分遮挡逐事件结果</a> · <a href="evidence/DECISION.json">实验裁决</a></p></section><p class="foot">Study: STUDY_DENSE_VIDEO_REVERSIBLE_IDENTITY_20261019 · 原始输出可追溯到3个完整比较Run。旧研究线与原实验结果保留。离线页面无网络依赖。</p></main><script>function filter(){document.querySelectorAll('tbody tr').forEach(r=>r.hidden=r.dataset.split!==document.querySelector('#split').value||(document.querySelector('#arm').value!=='all'&&r.dataset.arm!==document.querySelector('#arm').value))}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',filter));filter()</script></html>'''
    (root/'showcase/index.html').write_text(page,encoding='utf-8')

def main(root):
    ptr=read_json(root/'reversible_adjudication_pointer.json');data=read_json(root/ptr['path']/'SUMMARY.json')
    if (root/'reversible_visual_pointer.json').exists():
        old=read_json(root/'reversible_visual_pointer.json')['run_id'];dest=root/'showcase_versions'/old
        if not dest.exists():shutil.copytree(root/'showcase',dest)
    with Run(root,'reversible_visual_exhibit',dict(source_adjudication=ptr,manual_trajectory_edits=False,selection='positive and negative posthoc cases, not evaluation subset')) as run:
        plots(root,data);architecture(root);catalog=videos(root,data);webpage(root,data,catalog)
        for name in ['SUMMARY.json','DECISION.json','link_audit.json','partial_events.json','audit_by_sequence.json']:
            dest=root/'showcase/evidence'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(root/ptr['path']/name,dest)
        dest=root/'showcase/annotations';dest.mkdir(exist_ok=True)
        for name in ['occlusion_labels.json','ANNOTATION_FREEZE.json']:shutil.copy2(root/'annotations'/name,dest/name)
        write_json(run.path/'SUMMARY.json',dict(videos=catalog,figures=['new_videos_IDF1.png','all_video_deltas.png','rollback_tradeoff.png','architecture.svg'],offline_html='showcase/index.html'))
        write_json(root/'reversible_visual_pointer.json',dict(run_id=run.id,path=run.path.relative_to(root).as_posix()))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
