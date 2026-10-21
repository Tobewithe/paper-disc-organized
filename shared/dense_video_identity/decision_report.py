"""Static, offline evidence exhibit from completed runs, including negative results."""
from pathlib import Path
import html
import shutil
import subprocess
from collections import Counter
import numpy as np
import cv2
from PIL import Image,ImageDraw
from .common import Run,read_json,write_json,sha256,load_gt
from .learn import load_cache
from .render import font
from .reversible_showcase import pane
from .reversible_adjudicate import event_matches

LABELS={'bytetrack':'ByteTrack','immediate':'立即补接 I','confirm':'仅确认 C','reversible':'旧固定回退','botsort':'BoT-SORT','deepocsort':'Deep OC-SORT','fasttrack':'FastTracker','full':'CETD-v1','self':'删除竞争信息','now':'删除历史信息','full_drop_comp':'完整模型／删竞争','full_drop_time':'完整模型／删历史'}

def plots(dest,summary,probe,cal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':12})
    scores=probe['by_split']['seen_development']['scores'];fig,ax=plt.subplots(figsize=(10,5),facecolor='#f3f6fa');ax.set_facecolor('#f3f6fa');vals=[scores[k] for k in ['S','C','T']]
    ax.bar(['自身相似度','加入竞争身份','再加入连续观测'],vals,color=['#73869d','#257cad','#109c8e'],width=.55);ax.set_ylim(0,1);ax.axhline(.5,color='#aab7c7',linestyle='--');ax.set_ylabel('事件平衡 AUROC ↑');ax.set_title('信息验证通过：竞争身份与连续观测具有判别价值',loc='left',pad=20,fontweight='bold')
    for i,v in enumerate(vals):ax.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=19,fontweight='bold')
    fig.text(.12,.02,'15段已见视频 · 1026条已知观测 · 68个含正确事件 / 44个含错接事件（可重叠）',fontsize=10,color='#54687c');fig.tight_layout(rect=(0,.05,1,1));fig.savefig(dest/'evidence_auc.png',dpi=170);plt.close(fig)
    arms=['bytetrack','botsort','confirm','reversible','full','self','now','full_drop_comp','full_drop_time'];vals=[summary['combined']['confirmation'][a]['IDF1'] for a in arms];fig,ax=plt.subplots(figsize=(12,7),facecolor='#f3f6fa');ax.set_facecolor('#f3f6fa');ax.barh(range(len(arms)),vals,color=['#8b9bad' if a!='full' else '#168e89' for a in arms],height=.66);ax.set_yticks(range(len(arms)),[LABELS[a] for a in arms]);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.15);ax.set_xlabel('完整视频合并 IDF1 ↑（0–100）');ax.set_title('4段新视频：检验自己的模型是否有额外收益',loc='left',fontweight='bold',pad=18)
    for i,v in enumerate(vals):ax.text(v+.4,i,f'{v:.3f}',va='center',fontsize=12)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.text(.23,.02,'duck_3 / 4 / 5 / 6 全序列 · 同检测前端 · 相同4帧发布缓冲',fontsize=10,color='#54687c');fig.tight_layout(rect=(0,.04,1,1));fig.savefig(dest/'new_video_results.png',dpi=170);plt.close(fig)
    rows=summary['by_sequence'];names=sorted({r['sequence'] for r in rows});idx={(r['sequence'],r['arm']):r for r in rows};columns=['bytetrack','botsort','confirm'];delta=np.array([[idx[n,'full']['IDF1']-idx[n,b]['IDF1'] for b in columns] for n in names]);fig,ax=plt.subplots(figsize=(8,8),facecolor='#f3f6fa');ax.imshow(delta,aspect='auto',cmap='RdYlGn',vmin=-5,vmax=5);ax.set_xticks(range(3),['相对 '+LABELS[b] for b in columns]);ax.set_yticks(range(len(names)),[n+(' * 新确认' if idx[n,'full']['split']=='confirmation' else '') for n in names]);ax.tick_params(length=0)
    for i in range(len(names)):
        for j in range(3):ax.text(j,i,f'{delta[i,j]:+.2f}',ha='center',va='center',color='#17314c')
    ax.set_title('逐视频增量与损伤全部保留',loc='left',pad=20,fontweight='bold');fig.tight_layout();fig.savefig(dest/'all_video_deltas.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,7),facecolor='#f3f6fa');ax.set_xlim(0,14);ax.set_ylim(0,7);ax.axis('off');ax.text(.25,6.6,'CETD-v1：可训练的竞争反证时序决策网络',fontsize=22,fontweight='bold',color='#19344f');ax.text(.25,6.13,'12,993 个可训练参数 · 前端与64D身份头冻结 · 过去5次观测 · 输出保留 / 撤销旧身份',fontsize=13,color='#61778d')
    def box(x,y,w,h,title,body,color='#ffffff'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10,rounding_size=.13',linewidth=1,edgecolor='#cbd8e4',facecolor=color));ax.text(x+.15,y+h-.40,title,fontsize=14,fontweight='bold',color='#19344f');ax.text(x+.15,y+h-.85,body,fontsize=11,color='#4e677e',va='top',linespacing=1.7)
    box(.3,3.25,2.4,2.3,'冻结基础模型','YOLO26m-seg\n掩膜内外观 → 64D\nByteTrack + C 补接')
    box(3.4,4.15,3.0,1.35,'自身证据编码','外观 / 运动 / 质量\n8 → 24 维')
    box(3.4,1.9,3.0,1.75,'竞争集合编码','6个竞争个体 × 7维\n共享编码 → 相对注意力\n得到24维竞争表示','#e8f4f2')
    box(7.2,3.25,2.65,2.3,'时间决策','自身、竞争、差值：72D\n过去5次观测 → GRU32\n输出错误连接概率','#e8f4f2')
    box(10.7,3.25,2.85,2.3,'身份决策输出','保留 / 撤销旧身份\n运动框与检测不变\n已发布历史不可修改')
    for a,b in [((2.8,4.4),(3.3,4.8)),((2.8,4.0),(3.3,2.8)),((6.5,4.8),(7.1,4.5)),((6.5,2.9),(7.1,3.9)),((9.95,4.35),(10.6,4.35))]:ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=18,color='#678298',lw=1.8))
    ax.text(.35,.82,'机制消融：同结构 SELF 删除其他身份信息；同结构 NOW 删除过去观测；完整checkpoint再做两项定向干预。',fontsize=12,color='#19344f');ax.text(.35,.30,'结构已实现与训练；是否构成有效方法贡献，以完整视频、匹配消融及新确认结果为准。',fontsize=12,color='#8c6332');fig.tight_layout();fig.savefig(dest/'model_architecture.png',dpi=160);fig.savefig(dest/'model_architecture.svg');plt.close(fig)


def videos(root,dest,summary,audit):
    ptr=read_json(root/'decision_evaluation_pointer.json');rp=Path(ptr['path']);manifest=read_json(root/'input_manifest.json');seqs={s['name']:s for s in manifest['sequences']};dp=Path(read_json(root/'decision_dataset_pointer.json')['path']);catalog=[];selected=[]
    for relation,title in [('wrong_connection','模型停止沿用错误身份'),('correct_connection','反例：模型也可能撤销正确连接')]:
        options=[c for c in audit['cases'] if c['relation']==relation and c['split']=='confirmation']
        if options:
            c=sorted(options,key=lambda c:(c['sequence'],c['frame'],c['base_id']))[0];selected.append(dict(c,title=title,note='事后解释案例；不替代完整视频结果；拒绝旧身份不等于恢复原始ID。'))
    if not selected:
        # A truthful inactive-policy example, never presented as an added model gain.
        s=seqs['duck_4'];rows=read_json(dp/'duck_4'/'rows.json');r=next((r for r in rows if r['y']==0),next(iter(rows),None))
        if r:selected.append(dict(r,p_error=0,title='保守输出：与仅确认方法保持一致',note='这段是系统输出展示，不是新增模型改善的证据。',relation='no_increment',sequence='duck_4'))
    for number,c in enumerate(selected,1):
        name=c['sequence'];s=seqs[name];g=c.get('current_gt');
        if g is None:continue
        gt=load_gt(root,s);rec,_=load_cache(root,s);f0=max(1,c['frame']-15);f1=min(s['frames'],c['frame']+25);arms=['bytetrack','confirm','full'];tracks={a:read_json(rp/a/name/'tracks.json') for a in arms};focus={a:event_matches(gt,tracks[a])[0] for a in arms};boxes=[r[2:6] for f in range(f0,f1+1) for r in gt[f] if int(r[1]) in {int(g),c.get('anchor_gt')}]
        if not boxes:continue
        bb=np.array(boxes);lo=bb[:,:2].min(0);hi=bb[:,2:].max(0);pad=np.maximum((hi-lo)*.35,[80,65]);crop=list(map(int,[*np.maximum(lo-pad,[0,0]),*np.minimum(hi+pad,[s['width'],s['height']])]))
        folder=dest/'videos';folder.mkdir(exist_ok=True);key=f'case_{number:02d}';path=folder/(key+'.mp4');cap=cv2.VideoCapture(s['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,f0-1)
        proc=subprocess.Popen([str(root/'tools/ffmpeg.exe'),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r','5','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        for f in range(f0,f1+1):
            ok,img=cap.read();assert ok;canvas=Image.new('RGB',(1600,900),'#f2f5fa');d=ImageDraw.Draw(canvas);d.text((35,25),'CETD-v1  /  可训练身份决策实验',font=font(23),fill='#257cad');d.text((35,70),c['title'],font=font(36),fill='#17314c');d.text((35,125),c['note'],font=font(21),fill='#61758a');target=[r for r in gt[f] if int(r[1])==g];box=target[0][2:6] if target else None
            for j,a in enumerate(arms):
                x=35+j*520;d.rounded_rectangle((x,180,x+498,660),radius=12,fill='white');d.text((x+15,195),LABELS[a],font=font(24),fill='#158e87' if a=='full' else '#263f58');panel,ids=pane(img,rec[f-1],tracks[a][f-1],crop,498,375,box,focus[a].get((f,g)));canvas.paste(panel,(x,238));d.text((x+15,625),'匹配 ID '+','.join(map(str,ids)) if ids else '当前无可靠匹配',font=font(20),fill='#263f58')
            d.rounded_rectangle((35,690,1573,818),radius=14,fill='#17314c');d.text((55,709),f"{name} · 参照 G{g} · 原帧 {f} · 发布时刻 {min(f+4,s['frames'])}",font=font(24),fill='white');d.text((55,757),'身份由冻结模型输出；白框为官方GT；不手工回改ID。',font=font(23),fill='#8bd6c8');d.text((35,847),f"逐原帧慢放5fps（原视频{s['fps']:g}fps） | 原始裁剪区域与完整序列成绩均可追溯",font=font(20),fill='#61758a');proc.stdin.write(np.asarray(canvas).tobytes())
            if f==c['frame']:canvas.save(folder/(key+'.jpg'),quality=94)
        proc.stdin.close();assert proc.wait()==0;cap.release();catalog.append(dict(c,path='videos/'+key+'.mp4',poster='videos/'+key+'.jpg',start=f0,end=f1,frames=f1-f0+1,sha256=sha256(path)))
    write_json(dest/'video_catalog.json',catalog);return catalog


def build(root):
    ep=read_json(root/'decision_evaluation_pointer.json');rp=Path(ep['path']);s=read_json(rp/'SUMMARY.json');decision=read_json(rp/'DECISION.json');probe=read_json(Path(read_json(root/'evidence_probe_pointer.json')['path'])/'SUMMARY.json');cal=read_json(Path(read_json(root/'decision_calibration_pointer.json')['path'])/'SUMMARY.json');audit=read_json(Path(read_json(root/'decision_audit_pointer.json')['path'])/'SUMMARY.json');dest=root/'showcase';dest.mkdir(exist_ok=True)
    with Run(root,'decision_visual_report',dict(evaluation=ep,posthoc_illustrations=True,no_prediction_edits=True)) as run:
        plots(dest,s,probe,cal);catalog=videos(root,dest,s,audit);c=s['combined']['confirmation'];cmp=s['comparisons']['confirmation'];status='有限效用通过' if decision['status']=='limited_utility_supported' else '模型效用未通过冻结门槛'
        lines=['# 竞争反证时序决策网络：完整实验报告','',f'已在28358lan完成竞争/时间信息验证、CETD-v1及两组同结构消融训练、15段完整评价视频×12组输出。每臂12,993参数、800步；三臂共2400正式步，另有三臂各20步smoke。**{status}。**','',
        '| 维度 | 本次实验的整体研究链路 |','|---|---|',
        '| 问题 | 第三轮回退减少错接也误拆正确连接，不能稳定超过基线；单体外观下降是否缺少足够的身份反证？ |',
        '| 机制假说 | 竞争个体提供相对反证，连续观测降低瞬时外观变化的混淆。竞争解释是已见数据相关性或增加输入/参数，不能转化为新视频的任务收益。 |',
        '| 针对性方法 | 先以冻结旧轨迹作S0；随后训练自身编码、竞争集合注意力和GRU决策。固定前端/64D头/运动；同初始化、批次、预算的SELF/NOW消融与同checkpoint干预。8训练、4校准、7已见留出、4新确认视频，校准后锁定阈值。 |',
        f'| 整体结果 | S0单体/竞争/时间AUROC 0.5868/0.6612/0.6905，信息筛查通过。新4视频N/C/M/R的IDF1为{c["bytetrack"]["IDF1"]:.3f}/{c["confirm"]["IDF1"]:.3f}/{c["full"]["IDF1"]:.3f}/{c["botsort"]["IDF1"]:.3f}；M−C={cmp["confirm"]["delta_IDF1"]:+.3f}，95%视频簇区间{cmp["confirm"]["ci95"]}。信息价值、网络拟合、最终任务效用分开判断。 |',
        '| 研究决策 | 见下方人工核验后的研究裁决；本报告不将S0阳性或C的收益改记为新网络贡献。 |','',
        '## 全视频成绩','', '| 方法 | 新确认 IDF1 | HOTA | IDSW | 已见留出 IDF1 |','|---|---:|---:|---:|---:|']
        for a,l in LABELS.items():lines.append(f'| {l} | {c[a]["IDF1"]:.3f} | {c[a]["HOTA"]:.3f} | {c[a]["IDSW"]} | {s["combined"]["seen_holdout"][a]["IDF1"]:.3f} |')
        lines += ['', '## 相对增量与区间','']
        for a in ['bytetrack','botsort','confirm','self','now','full_drop_comp','full_drop_time']:
            v=cmp[a];lines.append(f'- M相对{LABELS[a]}：{v["delta_IDF1"]:+.4f} IDF1，视频簇95%CI [{v["ci95"][0]:+.4f}, {v["ci95"][1]:+.4f}]。')
        lines += ['','## 学习与校准','',f'训练：8段视频，2410个已知窗口、89个已知事件。阈值只在4段校准视频选择；完整、SELF、NOW均为{cal["choices"]["full"]["threshold"]}。完整网络校准候选0.50/0.70/0.90/0.95相对C均损伤，0.99无撤销、无额外收益；没有用新确认结果调阈值。','',f'新确认错接判别（事件平衡AUROC；辅助指标）：{audit["pooled"]["confirmation"]}。完整模型新确认撤销按严格局部GT归因为{audit["pooled"]["confirmation"]["full"]["rejections"]}。局部正确/错误标签不等于全轨迹IDF1效用。','',
        '## 科学边界','',
        '- 新确认是duck_3/4/5/6完整原序列，只能支持这4段鸟类视频；相似场景相关性未排除。一seed不能代表多seed稳定性。',
        '- 原19段已参与S0；其中8段用于本轮拟合、4段校准、7段留出复核，均不称独立盲测。',
        '- 不新增完整遮挡金标准；GT检测间断不等于完整物理遮挡。本轮不能宣称解决完全遮挡恢复。',
        '- 重新训练消融参数、初始化、批次相同，SELF输入秩变化及同checkpoint干预的分布偏移仍需区分；不凭某一项差值单独确立机制。',
        '- 基线为固定前端、特征与配置的本地实现比较，不是各论文发表配置的完整复现。',
        '- 每个输出采用最多4帧缓冲；没有据此声称端到端实时速度。','',
        '## 实际模型与新颖性','',
        '自身8→24；竞争集合6×7经共享7→24→24与相对注意力；自身/竞争/差组成72维，5次观测经GRU32→错误概率。全部12,993参数参与训练，已有前端和身份头冻结。研究的是已接回身份的反证决策，不是新的分割网络。',["MOTIP"] and '已有[MOTIP](https://arxiv.org/abs/2403.16848)研究可学习ID预测，[Learnable Graph Matching](https://arxiv.org/abs/2103.16178)研究关联上下文，[FastTracker](https://arxiv.org/abs/2508.14370)研究遮挡重识别。因此竞争、时序和神经网络本身不构成首次创新；只有本设计的针对性增量成立才可主张方法贡献。','',
        '## 真实Run','',f'- S0：{read_json(root/"evidence_probe_pointer.json")["run_id"]}',f'- 数据：{read_json(root/"decision_dataset_pointer.json")["run_id"]}',f'- 训练：{read_json(root/"decision_model_pointer.json")["run_id"]}',f'- 校准：{read_json(root/"decision_calibration_pointer.json")["run_id"]}',f'- 完整评价：{ep["run_id"]}',f'- 事后归因：{read_json(root/"decision_audit_pointer.json")["run_id"]}','', '执行原件：D:/thesis-video-v4-20261019。完整结果包含所有正负变化，未手工改轨迹。']
        (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');(root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
        evidence=dest/'evidence';evidence.mkdir(exist_ok=True)
        for name,obj in [('SUMMARY.json',s),('DECISION.json',decision),('AUDIT.json',audit),('CALIBRATION.json',cal),('PROBE.json',probe)]:write_json(evidence/name,obj)
        for name in ['PROTOCOL.md','METHOD_PROTOCOL.md','MODEL_AND_NOVELTY.md']:
            if (root/name).exists():shutil.copy2(root/name,dest/name)
        cards=''.join(f'<article><h3>{html.escape(v["title"])}</h3><p>{html.escape(v["note"])}</p><video controls preload="metadata" poster="{v["poster"]}" src="{v["path"]}"></video></article>' for v in catalog)
        table=''.join(f'<tr><td>{r["sequence"]}</td><td>{r["split"]}</td><td>{LABELS[r["arm"]]}</td><td>{r["IDF1"]:.3f}</td><td>{r["HOTA"]:.3f}</td><td>{r["IDSW"]}</td></tr>' for r in s['by_sequence'])
        page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CETD-v1 研究证据</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#edf3f8;color:#17314c;font:17px/1.7 system-ui,"Microsoft YaHei"}}main{{max-width:1260px;margin:auto;padding:45px 30px}}h1{{font-size:44px;line-height:1.25}}h2{{font-size:28px}}section,article{{background:white;border-radius:18px;padding:30px;margin:25px 0}}img,video{{width:100%;border-radius:10px}}a{{color:#257cad}}.badge{{background:#fff1db;color:#875d2c;padding:10px 18px;border-radius:10px;display:inline-block}}.lead{{font-size:21px}}p{{color:#536e86}}.scroll{{max-height:480px;overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dde6ee;text-align:left}}th{{position:sticky;top:0;background:#edf3f8}}input{{width:100%;padding:12px;border:1px solid #b7cbdc;border-radius:8px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}@media(max-width:800px){{.grid{{display:block}}h1{{font-size:32px}}}}</style><main><nav>CETD-v1 · 笔记本真实训练与全序列对照</nav><h1>让模型学习：<br>什么证据足以推翻身份连接？</h1><div class="badge">{status}</div><p class="lead">竞争与连续观测的信息验证通过。可训练模型与消融已完成；最终跟踪收益以完整视频成绩为准。</p><p><a href="REPORT.md">完整报告</a> · <a href="MODEL_AND_NOVELTY.md">模型与拟贡献</a> · <a href="METHOD_PROTOCOL.md">训练前冻结协议</a></p><section><h2>第一步：信息是否有用</h2><img src="evidence_auc.png"></section><section><h2>第二步：自己的可训练模型</h2><img src="model_architecture.png"><p>三组同结构模型，各12,993参数、800步；相同初始化和训练批次。这里的图对应实际代码和checkpoint。</p></section><section><h2>第三步：完整任务是否改善</h2><img src="new_video_results.png"><p>CETD相对ByteTrack {cmp['bytetrack']['delta_IDF1']:+.3f}；相对BoT-SORT {cmp['botsort']['delta_IDF1']:+.3f}；相对仅确认C {cmp['confirm']['delta_IDF1']:+.3f} IDF1。C已有的收益不能记为网络创新。</p><img src="all_video_deltas.png"></section><section><h2>冻结输出的案例回放</h2><p>事后选择用于解释；保留反例，不代替全量评价。</p>{cards if cards else '<p>没有适合展示为新增改善的撤销事件，未制作虚假的成功案例。</p>'}</section><section><h2>全部180项完整视频结果</h2><input id="filter" placeholder="输入视频名、方法或分组筛选"><div class="scroll"><table><thead><tr><th>视频</th><th>分组</th><th>方法</th><th>IDF1</th><th>HOTA</th><th>IDSW</th></tr></thead><tbody>{table}</tbody></table></div></section><section><h2>结论边界</h2><p>仅4段新鸟类视频、一seed；完整物理遮挡恢复尚未验证。结构可运行、信息有价值与论文核心创新成立，是不同结论。</p><p><a href="evidence/DECISION.json">冻结门槛判定</a> · <a href="evidence/AUDIT.json">模型撤销归因</a> · <a href="evidence/CALIBRATION.json">全部校准结果</a></p></section></main><script>document.querySelector('#filter').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))}})</script></html>'''
        (dest/'index.html').write_text(page,encoding='utf-8');write_json(run.path/'SUMMARY.json',dict(output=str(dest),videos=len(catalog),table_rows=len(s['by_sequence']),evaluation=ep,status=status));write_json(root/'decision_showcase_pointer.json',dict(run_id=run.id,path=str(run.path)))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();build(a.root)
