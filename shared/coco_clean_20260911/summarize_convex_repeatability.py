"""Archive S043 convex fits and S044 sampled-target repeatability, no retraining."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'experiments/coco_clean_20260911'


def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def check(p):
    rr=json.loads((p/'COMPLETE.json').read_text());num=0
    for name,h in rr['hashes'].items():
        assert hashlib.sha256((p/name.replace('\\','/')).read_bytes()).hexdigest()==h,name;num+=1
    return dict(status='PASS',hashes=num,scope='Artifact checks, not independent GPU reconstruction.')


def main():
    a=BASE/'diagnostics/convex_shared_readout_20260912';b=BASE/'diagnostics/teacher_repeatability_20260912'
    ma=json.loads((a/'metrics.json').read_text());sol=json.loads((a/'solvers.json').read_text());cond=json.loads((a/'conditioning.json').read_text())
    va=check(a);vb=check(b)
    for f in (b/'fits').glob('*.npz'):
        ar=np.load(f)
        assert not np.intersect1d(ar['A'],ar['B']).size
        assert not np.intersect1d(ar['E'],np.r_[ar['A'],ar['B']]).size
        assert len(ar['A'])==len(ar['B'])
    vb['disjoint_equal_budget_folds_verified']=64;dump(a/'VERIFICATION.json',va);dump(b/'VERIFICATION.json',vb)
    means=[]
    for mode in dict.fromkeys(r['mode'] for r in ma):
        rr=[r for r in ma if r['mode']==mode]
        means.append(dict(mode=mode,**{k:float(np.mean([r[k] for r in rr])) for k in ['teacher_mse','weighted_teacher_mse','teacher_kl','hard','sample_iou']}))
    reference={r['seed']:r for r in ma if r['mode']=='saved_head'}
    residualcapture=[]
    for mode in ['constant32','input73','saved_hidden128','adapted_hidden128']:
        values=[1-r['weighted_teacher_mse']/reference[r['seed']]['weighted_teacher_mse'] for r in ma if r['mode']==mode]
        residualcapture.append(dict(mode=mode,seed_values=values,mean=float(np.mean(values))))
    summarya=dict(means=means,seed_results=ma,weightedMSE_reduction_vs_saved=residualcapture,
        solvers=[{k:v for k,v in r.items() if k!='trace'} for r in sol],conditioning=cond,
        max_objective_gap_bound=max(r['objective_gap_upper_bound'] for r in sol),
        scope='7811fit instances only. Quadratic teacher-logit response target withridge1e-4 afterfitwhitening; '
            'all12true normal-equation residuals<1e-7; upper objectivegap bound based onridge. '
            'Not unregularized classcapacity lowerbound, not allnetwork information or taskAP claim.',verification=va)
    dump(a/'ANALYSIS.json',summarya)
    pixels=read(b/'pixels.csv');full=read(b/'full_masks.csv');witness=read(b/'witness.csv')
    images=sorted({int(r['image_id']) for r in witness});ii={v:k for k,v in enumerate(images)}
    draws=np.random.default_rng(20260912).multinomial(len(images),np.full(len(images),1/len(images)),size=2000)
    def interval(vv):
        vv=[(t,v) for t,v in vv if v is not None and np.isfinite(v)]
        if not vv:return dict(n=0,mean=None,ci95=None)
        im=np.array([ii[t[0]] for t,_ in vv]);vals=np.array([v for _,v in vv])
        count=np.bincount(im,minlength=len(images)).astype(float);total=np.bincount(im,weights=vals,minlength=len(images)).astype(float)
        den=np.einsum('bi,i->b',draws,count,optimize=False);num=np.einsum('bi,i->b',draws,total,optimize=False)
        v=np.divide(num,den,out=np.full(num.shape,np.nan),where=den>0)
        return dict(n=len(vv),images=len(set(im)),mean=float(vals.mean()),ci95=np.nanquantile(v,[.025,.975]).tolist())
    groups=[]
    for high in [True,False]:
        w=[r for r in witness if (r['high']=='True')==high]
        means_by=defaultdict(list)
        for r in pixels:
            if (r['high']=='True')!=high:continue
            tid=(int(r['image_id']),int(r['annotation_id']))
            for k in ['discrepancy','cosine','binary_disagreement','original_iou','A_iou','B_iou','original_bce','A_bce','B_bce']:
                if r[k]:means_by[tid,k].append(float(r[k]))
            means_by[tid,'A_iou_gain'].append(float(r['A_iou'])-float(r['original_iou']))
            means_by[tid,'B_iou_gain'].append(float(r['B_iou'])-float(r['original_iou']))
        tids=[(int(r['image_id']),int(r['annotation_id'])) for r in w]
        met={k:interval([(t,float(np.mean(v))) for (t,kk),v in means_by.items() if k==kk]) for k in dict.fromkeys(kk for t,kk in means_by)}
        fm={}
        for arm in ['A','B']:
            vals=defaultdict(list)
            for r in full:
                if (r['high']=='True')!=high or r['arm']!=arm:continue
                tid=(int(r['image_id']),int(r['annotation_id']))
                for k in ['iou_raw','coverage','neighbor','background']:
                    if r[k+'_after'] and r[k+'_before']:vals[tid,k].append(float(r[k+'_after'])-float(r[k+'_before']))
            fm[arm]={k:interval([(t,float(np.mean(v))) for (t,kk),v in vals.items() if kk==k]) for k in ['iou_raw','coverage','neighbor','background']}
        groups.append(dict(high=high,targets=len(w),skipped=sum(r['skipped']=='True' for r in w),
            full512_each=sum(int(r['fit_each'])==512 for r in w),less128_each=sum(int(r['fit_each'])<128 for r in w),
            fit_each_quantiles=np.quantile([int(r['fit_each']) for r in w],[0,.25,.5,.75,1]).tolist(),pixels=met,fullmask=fm))
    summaryb=dict(groups=groups,scope='64preselectedfit targets,3savedstates,A/B disjoint equalpixels,E commonunused; '
        'deduplicatedandimagecorrelated,samplebudgetvaries andreported. Each teacher uses itsownfitnormalizer. '
        'Seedmeans pertarget then2000imageclusterpointwiseCI; no multiplicitycontrol or equivalence/stabilitytest. '
        'Discrepancy is normalizedsquared difference not a noisevariance fraction. NoGT-freepredictions or taskAP.',verification=vb)
    dump(b/'ANALYSIS.json',summaryb)
    ra=ROOT/'refine-logs/coco-structure/CONVEX_SHARED_READOUT_RESULTS_20260912.md'
    lines=['# S043：固定表征的最后层已求解收敛，仍存在条件读出差距','',
        '2026-09-12。复用7,811个fit目标、512像素和三个保存教师；29.829秒，12个凸二次求解全部达到预先固定收敛门槛。没有读取transfer/val或计算新AP。','',
        '**结论：在当前固定中间表征、线性输出与正则目标下，剩余误差不能解释为最后层数值求解没收敛。这个限定结论不等于“全网没有信息”“全局系数表达不足”或不可改善的容量上限。**','',
        '## 求解与控制','',
        '四组分别为常量32维修正、原73维实例输入、S032保存头的128维中间特征、S042充分优化头的128维中间特征。后三组添加偏置。保持原模型系数为共同基准，只拟合共享线性修正矩阵；当前头有效输出也是中间特征的线性函数。','',
        '目标为0.5×实例factor归一加权的teacher logit均方误差 + 0.5×1e−4×||W||²。原型和输入用全fit二阶矩白化，特征值floor为最大值×1e−4；该可逆变换改善条件数，但正则几何依赖它。不是S041/S042概率KL目标，不能直接比较两种误差减少比例。','',
        'FP64矩阵自由预条件CG，最多800迭代；以重新计算的正规方程相对残差≤1e−7停止。实测常量1次、原输入201次、保存中间123–124次、适配中间101–104次。最大凸目标差上界||梯度||²/(2λ)=1.731e−10。仅适用于此正则目标，不是无正则最佳MSE证明。','',
        '## 训练集结果（三保存状态均值）','',
        '| 固定表征/输出 | teacher logit MSE | 概率KL | 采样IoU |','|---|---:|---:|---:|']
    for r in means:lines.append(f"| {r['mode']} | {r['teacher_mse']:.6f} | {r['teacher_kl']:.6f} | {100*r['sample_iou']:.3f}% |")
    lines+=['',
        '保存头的teacher加权MSE为2.278553；在其固定中间特征上收敛重拟合最后层为2.097559，仅减少约7.94%。适配后的中间特征进一步到2.003674，相对原保存头约减少12.06%。这些不是oracle IoU收益的兑现比例。','',
        '适配头本身概率KL0.079456优于重新拟合二次最后层的0.084885，而二次层MSE更好。这是不同目标下的取舍，不能择一宣布总体更强。全部IoU为训练512采样点，不是完整掩码或真实R75。','',
        '保存中间特征有45/47/45个特征值低于白化floor，适配中间均无；这是相关性的数值描述，不是特征坍塌或导致拥挤失败的证据。目标正则与表征共同变化，不能以差值做严格容量归因。','',
        '## 下一限定问题','',
        '若要归因实例条件信息不足，首先要检查教师目标对监督采样的重复性，避免要求共享头预测采样噪声。S044已完成，见TEACHER_REPEATABILITY_RESULTS_20260912.md。当前不扩大最后层优化迭代、不堆头容量，不把有限线性结果上升为整个架构缺陷。','',
        f'{va["hashes"]}项回执hash通过；有效最后层系数重放最大2.862e−6，低于事前1e−5容忍。无独立GPU重跑。',
        '源码convex_shared_readout.py；输出diagnostics/convex_shared_readout_20260912/ANALYSIS.json。']
    ra.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    rb=ROOT/'refine-logs/coco-structure/TEACHER_REPEATABILITY_RESULTS_20260912.md'
    hi=groups[0];lo=groups[1]
    lines=['# S044：教师具有共同改善方向，采样差异不能被忽略','',
        '2026-09-12。47张fit图、64个预选目标（高/非高各32）、三个保存状态、384次求解，4.781秒完成并全部收敛。没有共享网络训练、transfer/val评价或AP。','',
        '**结论：两批互不重叠监督点得到的教师并不完全一致，但都有正常原框下的掩码改善。现有结果不足以把全部共享预测困难归为采样噪声；也不足以宣称教师在所有目标上稳定。**','',
        '## 协议','',
        '从S039已固定128目标中只取64个fit目标，不按失败或教师效果重新挑选。每目标缓存2048坐标先去重、确定性打乱，再分互不相交A/B/E。A和B各k=min(512,floor(unique/3))点，E为其余共同评价点；三种子共享同一分割。每个教师只用自己的监督点计算原COCOY、RMS标准化和S040正则求解。不存在A/B/E坐标交叉，空间相关性仍在。','',
        f'高组k范围23–512，中位512，{hi["full512_each"]}/32达到512，{hi["less128_each"]}/32少于128；非高范围66–512，中位512，{lo["full512_each"]}/32达到512，{lo["less128_each"]}/32少于128。没有目标被跳过。不能称每个教师都有512独立像素。','',
        '## 共同未用像素与完整掩码','',
        '| 群组 | 修正响应余弦A/B | 二值分歧 | E原IoU | EA教师IoU | EB教师IoU |','|---|---:|---:|---:|---:|---:|']
    for g in groups:
        q=g['pixels'];lines.append('| '+('高' if g['high'] else '非高')+f" | {q['cosine']['mean']:.4f} | {100*q['binary_disagreement']['mean']:.3f}% | {100*q['original_iou']['mean']:.3f}% | {100*q['A_iou']['mean']:.3f}% | {100*q['B_iou']['mean']:.3f}% |")
    lines+=['',
        '高组归一平方差sum((ΔA−ΔB)²)/(sumΔA²+sumΔB²)均值0.1911，非高0.2140；这是响应差异指标，不能直接叫19.11%/21.40%的教师噪声或不可预测方差。','',
        '高组完整原COCO IoU：A较保存头+9.405点、B+8.784点；自身覆盖+3.894/+3.658，邻居错误−8.651/−8.513，背景−3.671/−2.963点。两教师都具有改善，不能只因教师彼此有分歧就否认读出机会；也不能把这些同图GT辅助收益当实际方法。','',
        'ANALYSIS.json提供三种子先按目标平均、2,000次图像簇bootstrap点态区间。未做等价检验，没有用“不显著”宣称两教师一样；高/非高未经组成控制，不作密集特有噪声结论。','',
        '## 研究取舍','',
        'S041的混合教师蒸馏停止；S042/S043限制了“纯粹最后层没优化好”的解释；S044表明教师存在共同有效方向，但有采样变化。这仍未给出已证实的密集机制方法。接下来若继续，应显式区分实例条件与图像原型基底适配：当前共享头只读h/level/box，其输出却要在逐图变化的原型坐标系中工作。检验同等容量下是否需要原型几何条件前，先用固定表征凸投影残差测其与图像/实例因素的关联，不能直接把这一结构观察当根因。旧S026逐像素P+h已做且NO_GO，不能重复换名；需明确新增的是全局读出的图像基底条件，而非再加一个普通MLP。该下一项尚未启动。','',
        f'{vb["hashes"]}项回执及64份A/B/E等预算、坐标互斥校验通过。没有独立GPU重跑。',
        '源码teacher_repeatability.py；输出diagnostics/teacher_repeatability_20260912/ANALYSIS.json。']
    rb.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    entry=('2026-09-12 S043/S044 COMPLETE / FIXED LINEAR GAP, REPEATABLE ORACLE COMPONENT：S0437811fit三状态四固定表征29.829秒，12/12凸二次PCG真残差<1e−7、max目标差界1.731e−10。保存头MSE2.278→固定hidden最后层2.098(约−7.94%)→adaptedhidden2.004；KL/IoU口径不同，不能断言全网输入缺失。S044预选64fit目标47图A/B/E坐标互斥、等预算k23–512/66–512，384fit全收敛4.781秒：高teacher修正cos.815、E二值分歧4.36%，完整IoU A/B+9.405/+8.784、均增自身降邻居/背景。采样变化存在但不是已证明的全部瓶颈。报告CONVEX_SHARED_READOUT_RESULTS及TEACHER_REPEATABILITY_RESULTS_20260912.md；所有进程完成，正式CCL暂停。下一有增量候选为图像原型基底条件与全局读出映射适配，必须区别S026P+h像素MLP，先残差/条件诊断，不直接当新方法；尚未启动。')
    for f in [ROOT/'refine-logs/coco-structure/EXPERIMENT_TRACKER.md',ROOT/'refine-logs/EXPERIMENT_TRACKER.md']:
        t=f.read_text(encoding='utf-8');i=t.find('\n');f.write_text(t[:i+1]+'\n'+entry+'\n'+t[i+1:],encoding='utf-8')
    f=ROOT/'refine-logs/coco-structure/TEACHER_FITTING_RESULTS_20260912.md';t=f.read_text(encoding='utf-8');i=t.find('\n')
    f.write_text(t[:i+1]+'\n后续S043固定表征凸二次拟合、S044教师采样重复性已完成；见CONVEX_SHARED_READOUT_RESULTS_20260912.md与TEACHER_REPEATABILITY_RESULTS_20260912.md。\n'+t[i+1:],encoding='utf-8')
    with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:
        for num,path in [('S043',ra),('S044',rb)]:f.write('\n- '+num+': '+str(path.relative_to(ROOT))+'; associated diagnostics ANALYSIS.json and source archived.\n')
    print(json.dumps(dict(S043=summarya['weightedMSE_reduction_vs_saved'],S044groups=groups,verification=[va,vb]),ensure_ascii=False))


if __name__=='__main__':main()
