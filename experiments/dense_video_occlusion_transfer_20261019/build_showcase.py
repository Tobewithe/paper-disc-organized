"""Publication plots and an offline view of actual experiment artifacts."""
import pathlib,json,gzip,sys,html,collections,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image,ImageDraw,ImageFont
import cv2

REPO=pathlib.Path(__file__).resolve().parents[2]
S=REPO/'experiments/dense_video_occlusion_transfer_20261019'
V=S/'verified_delivery'
OUT=REPO/'paper/presentations/可见性学习与真实视频身份恢复_20261019'
STUDY=V/'experiments/dense_video_occlusion_transfer_20261019'
COLORS={'N':'#94a3b8','R':'#df8c43','C':'#546c84','GEOM':'#ad9c70','FULL':'#168c82','NOW':'#5578ba','NOCTX':'#9575b6','SHUFFLE':'#c85b69'}
LABELS={'N':'ByteTrack','R':'BoT-SORT','C':'三观测确认 C','GEOM':'框重叠权重','FULL':'完整模型','NOW':'无时序变化','NOCTX':'无周围图像','SHUFFLE':'打乱对象权重'}
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def pointer(kind):return STUDY/'runs'/read(V/(kind+'_pointer.json'))['run_id']
def finish(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
def style():
 plt.rcParams.update({'font.family':'Microsoft YaHei','axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#142a3b','axes.unicode_minus':False,'font.size':10,'figure.dpi':120})
def figures():
 OUT.mkdir(parents=True,exist_ok=True);style();tp=pointer('train');ep=pointer('adjudicate');train=read(tp/'SUMMARY.json');ev=read(ep/'SUMMARY.json');decision=read(ep/'DECISION.json')
 # Synthetic capability: separate from downstream identity utility.
 fig,axes=plt.subplots(1,3,figsize=(14,4.3));classes=['可见','部分遮挡','完全遮挡','出视野']
 for ax,arm in zip(axes,['FULL','NOW','NOCTX']):
  r=train[arm]['evaluation']['synthetic_test'];c=np.asarray(r['confusion']);norm=c/np.maximum(1,c.sum(1,keepdims=True));ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
  for i in range(4):
   for j in range(4):ax.text(j,i,str(c[i,j]),ha='center',va='center',color='white' if norm[i,j]>.5 else '#142a3b')
  ax.set(xticks=range(4),yticks=range(4),xticklabels=classes,yticklabels=classes,xlabel='预测',ylabel='几何真值',title=f"{LABELS[arm]} | F1={r['macro_F1']:.3f}\n可见比例 MAE={r['visibility_MAE']:.3f}");ax.tick_params(axis='x',rotation=25)
 fig.suptitle('未参与训练的动物资产：遮挡状态识别',fontsize=16,y=1.06);finish(fig,'synthetic_confusion')
 h=ev['combined']['heldout11'];arms=list(COLORS);fig,axes=plt.subplots(1,2,figsize=(14,5),gridspec_kw={'width_ratios':[1.2,1]})
 vals=[h[a]['IDF1'] for a in arms];bars=axes[0].bar(range(8),vals,color=[COLORS[a] for a in arms],width=.65);axes[0].bar_label(bars,fmt='%.2f',padding=4,fontsize=9);axes[0].set(xticks=range(8),xticklabels=[LABELS[a] for a in arms],ylabel='IDF1（点）',ylim=(0,max(vals)+8),title='11段真实视频：整段身份一致性');axes[0].tick_params(axis='x',rotation=38)
 refs=['N','R','C','GEOM','NOW','NOCTX','SHUFFLE'];d=[ev['comparisons'][r]['delta_IDF1'] for r in refs];ci=np.asarray([ev['comparisons'][r]['paired_video_bootstrap_95'] for r in refs]);yy=np.arange(len(refs));axes[1].axvline(0,color='#94a3b8',lw=1);axes[1].errorbar(d,yy,xerr=[np.asarray(d)-ci[:,0],ci[:,1]-np.asarray(d)],fmt='o',color=COLORS['FULL'],capsize=4);axes[1].set(yticks=yy,yticklabels=['完整 − '+LABELS[r] for r in refs],xlabel='IDF1差值（点），95%视频簇区间',title='新增收益与不确定性');axes[1].invert_yaxis();fig.tight_layout();finish(fig,'real_results')
 names=ev['groups']['heldout11'];rows=ev['by_sequence'];get=lambda n,a:next(r['IDF1'] for r in rows if r['sequence']==n and r['arm']==a)
 fig,ax=plt.subplots(figsize=(11,5));yy=np.arange(len(names));ax.barh(yy-.17,[get(n,'FULL')-get(n,'C') for n in names],height=.32,color=COLORS['FULL'],label='完整 − 确认 C');ax.barh(yy+.17,[get(n,'FULL')-get(n,'R') for n in names],height=.32,color=COLORS['R'],label='完整 − BoT-SORT');ax.set(yticks=yy,yticklabels=names,xlabel='IDF1差值（点）',title='逐视频收益与退化全部保留');ax.axvline(0,color='#334155',lw=.8);ax.legend(frameon=False);ax.invert_yaxis();fig.tight_layout();finish(fig,'per_video')
 fig,ax=plt.subplots(figsize=(10,4))
 for a in ['FULL','NOW','NOCTX']:
  c=read(tp/(a+'_curve.json'));ax.plot([v['step'] for v in c],[v['loss'] for v in c],label=LABELS[a],color=COLORS[a],lw=2)
 ax.set(xlabel='正式优化步数',ylabel='分类＋0.5×回归损失',title='同初始化、同批次、同预算：每臂2000步');ax.legend(frameon=False);fig.tight_layout();finish(fig,'training')
 selected=[0,1,8,15,22,29,48,55,60,65,66,71];fig,axes=plt.subplots(3,4,figsize=(14,8))
 for ax,i in zip(axes.flat,selected):
  p=V/'synthetic_v2'/('%03d'%i);a=read(p/'annotations.json');ax.imshow(Image.open(p/'rgb/010.png'));ax.axis('off');ax.set_title(f"{a['asset']} · {a['scenario']} · {a['split']}",fontsize=10)
 fig.suptitle('带骨骼动物、自然光和六类场景；低多边形资产，非照片级',fontsize=16);fig.tight_layout();finish(fig,'scene_gallery')
 # The full silhouette is a supervised geometric label; never a model input.
 folder=V/'synthetic_v2/000';a=read(folder/'annotations.json');f=10;arr=np.load(folder/'labels/010.npz');rgb=np.asarray(Image.open(folder/'rgb/010.png'));track=1
 candidates=[r for r in a['records'] if r['frame']==f and r['state']==1]
 if candidates:track=candidates[0]['track_id']
 row=next(r for r in a['records'] if r['frame']==f and r['track_id']==track);vis=arr['instance_id']==track;full=arr['amodal_'+str(track)];fig,axes=plt.subplots(1,3,figsize=(12,3.6))
 axes[0].imshow(rgb);axes[0].set_title('渲染图像（模型可读）');axes[1].imshow(vis,cmap='Greens');axes[1].set_title(f"实际可见 V = {row['visible_pixels']}像素");axes[2].imshow(full,cmap='Blues');axes[2].set_title(f"视野内完整轮廓 A = {row['in_frame_pixels']}像素")
 for ax in axes:ax.axis('off')
 fig.suptitle(f"可见比例 V/A = {row['visibility']:.3f}；出画比例另记 {row['truncation']:.3f}",fontsize=14);fig.tight_layout();finish(fig,'geometry_labels')
 fig,ax=plt.subplots(figsize=(14,4.6));ax.set(xlim=(0,14),ylim=(0,4.6));ax.axis('off')
 boxes=[(.2,2.7,2.1,1.1,'当前与过去 RGB\n观测框 / 外推框','#e6edf5'),(3,2.7,2.3,1.1,'冻结 ResNet18\n目标 ROI / 周围 ROI','#e6edf5'),(6,2.7,2.2,1.1,'392→128\nGRU 128→64','#c8ebe6'),(9,2.7,2.1,1.1,'4类遮挡状态\n连续可见比例','#c8ebe6'),(9,.5,2.1,1.1,'历史身份外观\n可见性加权记忆','#c8ebe6'),(12,.5,1.8,1.1,'固定候选与\n三观测确认','#e6edf5'),(3,.5,4.9,1.1,'仅合成几何监督；真实 GT 只用于评价\nNOW / NOCTX 同参数、同初始化、同预算','#f5eee2')]
 for x,y,w,hh,t,c in boxes:ax.add_patch(FancyBboxPatch((x,y),w,hh,boxstyle='round,pad=.04,rounding_size=.08',facecolor=c,edgecolor='none'));ax.text(x+w/2,y+hh/2,t,ha='center',va='center',fontsize=12)
 for x1,y1,x2,y2 in [(2.3,3.25,3,3.25),(5.3,3.25,6,3.25),(8.2,3.25,9,3.25),(10.05,2.7,10.05,1.6),(11.1,1.05,12,1.05)]:ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','color':'#567083','lw':1.8})
 ax.set_title('待检验的贡献：专门学习可见性，减少遮挡外观进入身份记忆',fontsize=16,pad=16);finish(fig,'model')
 return train,ev,decision

def main():
 train,ev,decision=figures();(OUT/'results.json').write_text(json.dumps({'training':train,'evaluation':ev,'decision':decision},ensure_ascii=False,indent=2),encoding='utf-8')
 model=read(pointer('train')/'MODEL.json');passed=all(decision['gates'].values());status='有限样本效用通过' if passed else '当前配置未建立稳定改善'
 cards=''.join(f'<article><span>{LABELS[a]}</span><strong>{ev["combined"]["heldout11"][a]["IDF1"]:.3f}</strong><small>真实11段 · IDF1</small></article>' for a in ['N','R','C','FULL'])
 sections=[('场景与监督','骨骼动画模型增加了动物形状、动作和环境变化。训练五类资产；Llama验证、Pug和Fox测试，按资产隔离。','scene_gallery'),('标签如何定义','可见性与出视野分开；完整轮廓只作合成监督，不进入真实推理。','geometry_labels'),('专门训练的模型',f'冻结图像骨干；新增可训练参数{model["trainable_parameters"]:,}。每臂2000步，真实GT训练0条。','model'),('训练完成','同样本顺序和初始化，固定最后一步权重，不按真实结果选模型。','training'),('遮挡能力','独立动物资产上的四类识别与可见比例误差。该成绩不等于真实身份恢复效果。','synthetic_confusion'),('真实任务效用','23段完整评价，其中11段历史留出为主要结果；并非首次盲测。误差线为2000次配对视频簇bootstrap。','real_results'),('逐视频检查','保留每段视频的改善与退化，完整模型相对C与BoT-SORT分别比较。','per_video')]
 sections.append(('一次GT证实的正确补接','duck_3的三段轨迹均对应官方GT74；模型接回同一身份，BoT-SORT也已保持。它支持一个局部真实改善，不能证明已解决完全遮挡。','correct_reconnection_case'))
 content=''.join(f'<section><div class="section-title"><h2>{i+1:02d}　{t}</h2><p>{p}</p></div><img src="{f}.png" alt="{html.escape(t)}"></section>' for i,(t,p,f) in enumerate(sections))
 text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>可见性学习与真实视频身份恢复</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f8;color:#142a3b;font-family:"Microsoft YaHei",sans-serif;line-height:1.7}}main{{max-width:1250px;margin:auto;padding:40px 24px}}header{{padding:42px;background:#132f42;color:white;border-radius:22px}}h1{{font-size:34px;margin:12px 0}}h2{{font-size:23px;margin:0}}p{{color:#536c7e}}header p{{color:#cbdae4;max-width:840px}}.tag{{color:#8fe0d4;letter-spacing:2px}}.status{{margin:24px 0 0;padding:15px 20px;background:#244357;border-radius:12px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}}article{{background:white;padding:24px;border-radius:16px;border-top:4px solid #168c82}}article span,article small{{display:block;color:#536c7e}}article strong{{font-size:36px;font-weight:600}}section{{background:white;border-radius:20px;padding:26px;margin:24px 0}}.section-title{{border-left:4px solid #168c82;padding-left:18px;margin-bottom:24px}}section img{{display:block;width:100%;height:auto}}video{{width:100%;border-radius:12px}}footer{{padding:30px;color:#536c7e;font-size:13px}}a{{color:#168c82}}.note{{background:#ecf5f3;border-radius:10px;padding:18px}}</style><main><header><div class="tag">毕业论文实验 · 2026.10.19 · A6000 / Slurm</div><h1>学习遮挡，能否更好地找回身份？</h1><p>从带骨骼的三维动物和自动几何标签出发，训练遮挡识别模型，再用固定真实检测结果验证身份恢复。能力、实际收益和机制贡献分别评价。</p><div class="status">{status}。不把系统跑通或训练损失下降当作实际改善。</div></header><div class="cards">{cards}</div>{content}<section id="videos"><div class="section-title"><h2>08　真实视频并排对照</h2><p>相同图像和时间轴；框与身份来自实际输出。案例是按已声明规则选择的可视化，整体结论以上方完整视频评价为准。</p></div><div id="video-list"></div></section><section><h2>结论边界</h2><p class="note">现有物理部分遮挡补充层11例；可靠完整物理遮挡金标准仍为0。GT间断仅是原因未标注的重现代理。一个seed、少量资产与历史可见数据不能支持普遍泛化或完全遮挡已解决的主张。</p><a href="results.json">查看完整数值与冻结裁决</a></section><footer>动物资产：Quaternius Farm Animals（CC0）；Fox模型：PixelMannen（CC0），动画tomkranis、glTF转换AsoboStudio/scurest（CC-BY4）。HDRI与地面纹理：<a href="https://polyhaven.com/">Poly Haven</a>（CC0）。真实视频：AnimalTrack公开基准。离线展示不上传视频或数据。</footer></main><script>fetch('video_catalog.json').then(r=>r.json()).then(v=>{{document.getElementById('video-list').innerHTML=v.map(c=>`<h3>${{c.title}}</h3><p>${{c.description}}</p><video controls preload="metadata" poster="${{c.poster}}"><source src="${{c.video}}" type="video/mp4"></video>`).join('')}}).catch(()=>{{}})</script></html>'''
 if (OUT/'video_catalog.json').exists():
  videos=read(OUT/'video_catalog.json');markup=''.join(f'<h3>{html.escape(c["title"])}</h3><p>{html.escape(c["description"])}</p><video controls preload="metadata" poster="{c["poster"]}"><source src="{c["video"]}" type="video/mp4"></video>' for c in videos);text=text.replace('<div id="video-list"></div>','<div id="video-list">'+markup+'</div>')
 text=text.replace('08　真实视频并排对照','09　真实视频并排对照')
 start=text.find('<script>fetch(')
 if start>=0:text=text[:start]+text[text.index('</script>',start)+9:]
 (OUT/'index.html').write_text(text,encoding='utf-8');print(OUT)
if __name__=='__main__':main()
