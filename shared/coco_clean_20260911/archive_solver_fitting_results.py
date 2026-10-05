"""Archive S041/S042 observed findings; no training, no result selection."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/coco_clean_20260911'


def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def check(run):
    receipt=json.loads((run/'COMPLETE.json').read_text());passed=0;mutable=[]
    for name,digest in receipt['hashes'].items():
        f=run/name.replace('\\','/');actual=hashlib.sha256(f.read_bytes()).hexdigest()
        if actual!=digest:
            if name=='progress.json' and run.name=='solver_response_20260912':
                # Runner called progress('COMPLETE') after hashing receipt.
                progress=json.loads(f.read_text())
                if progress['stage']!='COMPLETE':raise RuntimeError('Unexpectedprogressmutation')
                mutable.append(dict(file=name,recorded=digest,current=actual,reason='Runner final progress update after receipt; results/source unchanged.'))
            else:raise RuntimeError(f'Changed artifact {f}')
        else:passed+=1
    return dict(status='PASS_WITH_DOCUMENTED_MUTABLE_PROGRESS' if mutable else 'PASS',immutable_hashes_passed=passed,
        mutable_progress=mutable,scope='Existing artifact verification only, not independent GPU reproduction.')


def main():
    a=BASE/'diagnostics/solver_response_20260912';b=BASE/'diagnostics/teacher_fitting_20260912'
    j=json.loads((a/'ANALYSIS.json').read_text());m=json.loads((b/'metrics.json').read_text());gr=read(b/'gradient.csv');lb=json.loads((b/'lbfgs.json').read_text())
    v=check(a)
    for f in a.glob('*_s*/COMPLETE.json'):
        r=json.loads(f.read_text());files=sorted((f.parent/'checkpoints').glob('epoch*.pt'))
        assert len(files)==15 and r['updates']==3675
        assert hashlib.sha256(files[-1].read_bytes()).hexdigest()==r['final_sha256']
    sets={}
    for r in read(a/'gt_recovery.csv'):sets.setdefault(r['arm'],set()).add(int(r['annotation_id']))
    assert len(sets)==13 and all(len(s)==2002 and s==next(iter(sets.values())) for s in sets.values())
    v.update(checkpoints=135,final_checkpoint_hashes=9,arms=13,ordinaryGT_each=2002,prediction_parity=j['parity'])
    dump(a/'VERIFICATION.json',v);vb=check(b);dump(b/'VERIFICATION.json',vb)
    fm=j['fit_metrics'];mean=lambda rows,key:float(np.mean([r[key] for r in rows]))
    capture=[]
    for s in range(3):
        start=next(r for r in fm if r['mode']=='saved' and r['seed']==s)['teacher_kl']
        end=next(r for r in fm if r['mode']=='solver_response' and r['seed']==s)['teacher_kl']
        capture.append(1-end/start)
    r1=ROOT/'refine-logs/coco-structure/SOLVER_RESPONSE_RESULTS_20260912.md'
    lines=['# S041：教师响应监督未提供额外密集召回收益','',
        '2026-09-12。三臂×三种子完成，每臂从保存S032同一权重、Adam状态、样本顺序继续15轮，135checkpoint齐全。1,200fit/7,811目标，300已探索transfer/2,002普通GT。训练+解码+官方评价391.953秒，另统计与P90。','',
        '**结论：当前solver-response配方未通过预定筛查门槛。不扩大训练，不调权重/轮数追分，不启动新确认集或重跑val。**','',
        '## 正常任务结果','',
        '三个保存种子的均值，AP/Recall按0–100显示；差距为非高减高R75，越小越好。','',
        '| 监督 | Mask AP | 高拥挤R75 | 非高R75 | 差距 |','|---|---:|---:|---:|---:|']
    for r in j['means']:lines.append('| '+r['mode']+' | '+' | '.join(f"{100*r[k]:.3f}" for k in ['mask_ap','r75_high','r75_low','gap'])+' |')
    lines+=['',
        'solver对direct：AP+0.069点（无AP区间），高R75−0.112[-0.678,+0.518]点；差距缩小量−0.053[-0.687,+0.620]。solver对self-response：高R75+0.112[-0.581,+0.884]，没有可靠额外收益。相对旧15轮保存头有所提高，同预算direct也提高，不能归为教师贡献。','',
        '真正低/中/高GT分别1,140/564/298。direct R75为64.825/68.085/55.369%，solver为64.678/68.203/55.257%。历史task_summary的r75_low表示nonhigh，ANALYSIS.groups区分了真正低/中/高。','',
        '统一全局Precision≥90%、MaskIoU=.75操作点下，saved/direct/self/solver高组Recall均17.450%；solver非高33.255%，direct32.805%，高/非高差距点估计扩大。仅评价曲线描述，不是部署阈值；没有按密度分别选阈值。','',
        '## 空间变化与任务指标的区别','',
        '266个固定框匹配高目标，solver对direct空间IoU+0.203[0.069,0.334]点，但自身覆盖−0.422[-0.648,-0.238]、邻居−0.807[-1.099,-0.531]、背景−0.581[-0.871,-0.342]点。仍有收缩取舍，未转成全部298高GT的严格恢复收益。','',
        '类别×大小共同支持24层、255高/800非高，solver对direct高减非高增益+0.317[-0.703,+1.554]点，也不支持密集专属效果。该控制粗糙且探索性，不作因果结论。','',
        '## 训练端线索','',
        '23,433个fit教师求解全达到梯度门槛，BCE约0.262→0.1065。系数范数比中位约2.56、95分位约6.76、最大31.23，保留所有目标，仅使用sigmoid概率KL，不直接回归系数。教师预计算约114.2秒、shared SGD约63.1秒；共享更新相同不等于总计算相同。','',
        f'混合损失训练使teacher KL相对起点下降平均{100*np.mean(capture):.3f}%；直接训练仅小幅下降。训练采样IoU原约81.545%，direct约81.695%，solver约81.685%。因此不能只说“拟合很好但跨图失败”，也不能直接断言输入缺失。solver绝对logit MSE增加，概率KL与logit拟合是不同口径。','',
        '## 下一判别已由S042完成','',
        '同头纯teacher-KL的同预算Adam，以及另行有限全批量LBFGS仅fit诊断，见TEACHER_FITTING_RESULTS_20260912.md。没有再评价transfer/AP，不把强优化器当新方法。','',
        '## 复核与范围','',
        f'{v["immutable_hashes_passed"]}项不可变回执hash、9最终checkpoint哈希、135checkpoint、13臂相同2,002GT身份通过；原模型和三个保存头预测JSON精确重放。P90另核对AP及逐GT hit75一致。',
        '唯一hash变化是progress.json：runner在COMPLETE回执后写入最终进度，已保存旧/新hash及原因，结果和源码未改。统计v1缩进错误、v2空低拥挤相邻对的整数NaN错误保留日志；v3仅修统计，无训练或解码重跑。',
        '三种子先逐GT均值再图像簇2,000bootstrap；点态区间未多重校正，没有AP区间。没有独立GPU训练复现、新方法或创新声明。','',
        '结果：experiments/coco_clean_20260911/diagnostics/solver_response_20260912/ANALYSIS.json。源码train_solver_response.py、summarize_solver_response.py；中文协议SOLVER_RESPONSE_PROTOCOL_20260912.md。']
    r1.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    means=[]
    for mode in ['initial','pure_adam','pure_lbfgs']:
        rows=[r for r in m if r['mode']==mode];means.append(dict(mode=mode,**{k:mean(rows,k) for k in ['teacher_kl','response_capture','hard','sample_iou','teacher_mse']}))
    result=dict(means=means,seed_metrics=m,gradient=gr,optimizer=[{k:v for k,v in r.items() if k!='loss_trace'} for r in lb],
        mixed_response_capture=capture,scope='Fit-only diagnostic on7811targets/3savedstates; no transferAP. LBFGS budgetdifferent,nonconvex and all3hititerationcap. '
        'No proof of missingfeatures/capacitylimit or intrinsicdensity-specificconflict.',verification=vb)
    dump(b/'ANALYSIS.json',result)
    r2=ROOT/'refine-logs/coco-structure/TEACHER_FITTING_RESULTS_20260912.md'
    lines=['# S042：混合损失不是充分解释，共享拟合仍有优化余地','',
        '2026-09-12。48.437秒完成。复用S041的7,811fit目标和三个教师，不读transfer/val，不生成新oracle，不报告AP。','',
        '**结论：不能把S041失败简单归为hard与teacher梯度互相抵消；同输入同头更充分优化能改善训练拟合，因此也不能断言输入不含可用信息。但所有强优化参考触及预算，仍不能区分容量、条件信息和更充分优化。**','',
        '## 受控比较','',
        'pure Adam：从同S032保存头及Adam历史、随机顺序出发，去掉hard loss，仅teacher KL，15轮×三种子45checkpoint。pure LBFGS：同一起点、同头同输入、纯teacherKL，100次全批量有限优化、history10、strong-wolfe、max_eval125；预算不同，属于拟合参考，不能用来证明方法公平提升。','',
        '| fit目标 | 教师KL | 相对初始KL下降 | 采样掩码IoU | 原hard loss |','|---|---:|---:|---:|---:|']
    for r in means:lines.append(f"| {r['mode']} | {r['teacher_kl']:.6f} | {100*r['response_capture']:.2f}% | {100*r['sample_iou']:.3f}% | {r['hard']:.6f} |")
    lines+=['',
        f'S041混合loss的KL下降约{100*np.mean(capture):.2f}%；去掉hard loss后同预算纯teacher Adam约7.03%，增幅有限。全批量参考约20.24%，有更多训练拟合空间，但不等于学到20%的全部oracle性能增益；它只表示相对KL减少比例。所有IoU来自fit的512采样点，不能拿来当完整GT mask或新图成绩。','',
        '## 梯度证据与限度','',
        '三个保存起点的全fit参数hard梯度与teacher梯度余弦分别−0.0265、0.8480、0.3360；总H+T下降方向在三种子都降低teacher目标的一阶值。没有一致强负相关或相互抵消证据。仅当前参数的局部导数，不能描述整段历史训练。','',
        'LBFGS三种子均100次迭代触上限，实际闭包评估108/110/111次，最终梯度无穷范数0.000891/0.000232/0.000701，尚未达到设定1e−7。不能把剩余误差当作共享映射的表达下界或输入信息缺失证明。也不继续增加迭代直到得到想要的结果。','',
        '概率KL下降同时绝对logit MSE上升。饱和概率的刻度/范数仍有歧义，因此不能用“teacher MSE没降”单独宣布拟合无效，也不能只看KL便宣布形状恢复。','',
        '## 对主线的取舍','',
        '当前保留的问题是“给定特征与原型，实例级求得的有效读出为何难以共享预测”；S041简单软目标方案停止，S042不自动升级为LBFGS方法。下一项若继续，应选有判别力的共享拟合基准，例如固定中间表征后的凸最后层teacher拟合，以较明确的优化收敛读数排除非凸优化影响；仅能限定该固定表征/最后层，不能证明全网信息不足。优先在fit限定问题，暂不再训练更大头/重跑val。尚未启动下一实验。','',
        f'{vb["immutable_hashes_passed"]}项回执hash通过。3种子数据顺序与教师annotation身份一致，45epoch checkpoint及三个LBFGS最终状态保存。无独立GPU重跑、无跨图方法结论。','',
        '输出experiments/coco_clean_20260911/diagnostics/teacher_fitting_20260912，源码diagnose_teacher_fitting.py。']
    r2.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    entry=('2026-09-12 S041/S042 COMPLETE / SIMPLE DISTILLATION NO_GO, FIT OPTIMIZATION UNRESOLVED：S041三臂三种子135checkpoint，solver高R7555.257vsdirect55.369差−.112[-.678,.518]点，AP+.069无CI，P90高同17.450；覆盖↓邻居↓的收缩取舍。停止混合KL版。S042仅fit7811/3种子48.437秒，混合KL减少约5.24%、pureAdam7.03%、finitefullLBFGS20.24%，采样IoU81.545→82.388；不是完整/AP。H/T起点梯度cos−.0265/.848/.336，无一致抵消，LBFGS全100迭代触预算且非平稳，不得判输入信息缺失/容量上限。报告SOLVER_RESPONSE_RESULTS及TEACHER_FITTING_RESULTS_20260912.md，diagnostics/solver_response_20260912与teacher_fitting_20260912。所有进程完成。下一可判别项为固定中间表征下凸共享最后层teacher拟合，不是调更大头/迭代追分，尚未启动；正式CCL/端到端暂停。')
    for f in [ROOT/'refine-logs/coco-structure/EXPERIMENT_TRACKER.md',ROOT/'refine-logs/EXPERIMENT_TRACKER.md']:
        t=f.read_text(encoding='utf-8');i=t.find('\n');f.write_text(t[:i+1]+'\n'+entry+'\n'+t[i+1:],encoding='utf-8')
    with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:
        f.write('\n- S041: '+str(r1.relative_to(ROOT))+'; diagnostics/solver_response_20260912/ANALYSIS.json; train_solver_response.py and summarize_solver_response.py.\n')
        f.write('- S042: '+str(r2.relative_to(ROOT))+'; diagnostics/teacher_fitting_20260912/ANALYSIS.json; diagnose_teacher_fitting.py.\n')
    print(json.dumps(dict(S041_verification=v,S042=result),ensure_ascii=False))


if __name__=='__main__':main()
