"""Independent desktop arithmetic/asset checks; never trains or changes outputs."""
import pathlib,json,hashlib,collections,datetime,shutil
import numpy as np
from build_showcase import REPO,S,V,OUT,pointer,read,LABELS

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,v):pathlib.Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def idf(rows):
 tp,fp,fn=[sum(r[k] for r in rows) for k in ['IDTP','IDFP','IDFN']];return 200*tp/max(1,2*tp+fp+fn)
def main():
 train=read(pointer('train')/'SUMMARY.json');ev=read(pointer('adjudicate')/'SUMMARY.json');original=read(pointer('evaluate')/'SUMMARY.json');decision=read(pointer('evaluate')/'DECISION.json');model=read(pointer('train')/'MODEL.json');checks={}
 for arm in ['FULL','NOW','NOCTX']:
  assert train[arm]['steps']==2000 and sha(pointer('train')/(arm+'.pt'))==train[arm]['sha256']
  d=np.load(pointer('train')/(arm+'_synthetic_test_predictions.npz'));conf=np.bincount(d['y']*4+d['prob'].argmax(1),minlength=16).reshape(4,4);f1=(2*np.diag(conf)/np.maximum(1,conf.sum(0)+conf.sum(1))).mean();r=train[arm]['evaluation']['synthetic_test'];assert conf.tolist()==r['confusion'] and abs(f1-r['macro_F1'])<1e-12;assert abs(float(np.abs(d['visibility']-d['truth_visibility']).mean())-r['visibility_MAE'])<1e-6
  checks[arm+'_synthetic_recomputed']=True
 assert read(pointer('train')/'run.json')['training_steps']==6000 and read(pointer('train')/'run.json')['smoke_training_steps']==60 and model['trainable_parameters']==87877
 checks['train_steps_and_parameters']=True
 # Same bootstrapped videos; independently sum sufficient statistics.
 names=ev['groups']['heldout11'];idx={(r['sequence'],r['arm']):r for r in ev['by_sequence']};rng=np.random.default_rng(20261019);resamples=[rng.choice(names,len(names),replace=True) for _ in range(2000)];recomputed={}
 for split,nn in ev['groups'].items():
  for arm in LABELS:assert abs(idf([idx[n,arm] for n in nn])-ev['combined'][split][arm]['IDF1'])<1e-10
 for ref,v in ev['comparisons'].items():
  values=[idf([idx[n,'FULL'] for n in sample])-idf([idx[n,ref] for n in sample]) for sample in resamples];ci=np.percentile(values,[2.5,97.5]);assert np.max(np.abs(ci-v['paired_video_bootstrap_95']))<1e-10;recomputed[ref]={'delta_IDF1':idf([idx[n,'FULL'] for n in names])-idf([idx[n,ref] for n in names]),'CI95':ci.tolist()}
 checks['all_IDF1_and_2000_bootstrap_independently_recomputed']=True
 # Old frozen comparator parity: the prior complete evaluation contains 15 sequences.
 old=read(pathlib.Path('C:/Users/Public/thesis-video-v4-20261019/verified_delivery/runs/RUN_DECISION_FULL_EVALUATION_20261019T134938_db35851b/SUMMARY.json'));mapping={'bytetrack':'N','botsort':'R','confirm':'C'};parity=[]
 for row in old['by_sequence']:
  if row['arm'] not in mapping:continue
  r=idx[row['sequence'],mapping[row['arm']]];fields=['IDF1','HOTA','AssA','DetA','IDSW','FP','FN','IDTP','IDFP','IDFN'];delta=max(abs(r[k]-row[k]) for k in fields);assert delta<1e-8,(row['sequence'],row['arm'],delta);parity.append({'sequence':row['sequence'],'arm':mapping[row['arm']],'max_difference':delta})
 assert len(parity)==45;checks['45_historical_baseline_rows_parity']=True
 # Immutable primary outputs survive the supplementary matching correction.
 for key in ['by_sequence','combined','comparisons','per_video_FULL_minus_C_IDF1','engineering_checks']:assert ev[key]==original[key]
 assert len(ev['engineering_checks'])==23 and all(all(v for k,v in c.items() if k!='sequence') for c in ev['engineering_checks']);checks['23_video_causality_geometry_identity_and_C_replay']=True
 fptr=read(V/'features_pointer.json');new=p=V/'experiments/dense_video_occlusion_transfer_20261019/runs'/fptr['run_id'];repair=read(new/'SUMMARY.json');assert repair['synthetic_data_byte_identical'] and repair['trained_checkpoints_reused'];checks['real_input_repair_did_not_change_synthetic_or_models']=True
 counts=collections.Counter();scene_count=0;pixel_checks=0
 for folder in sorted((V/'synthetic_v2').iterdir()):
  a=read(folder/'annotations.json');assert len(a['records'])==128 and len({(r['frame'],r['track_id']) for r in a['records']})==128;pixel_checks+=a['independent_checked_pixels'];scene_count+=1
  for r in a['records']:
   v,ai,af=r['visible_pixels'],r['in_frame_pixels'],r['full_pixels'];assert 0<=v<=ai<=af;state=3 if ai==0 else 2 if v==0 else 1 if v/ai<.85 else 0;assert state==r['state'];counts[state]+=1
  for p in (folder/'labels').glob('*.npz'):
   arrays=np.load(p);frame=int(p.stem)
   for k in range(1,5):
    vv=arrays['instance_id']==k;aa=arrays['amodal_'+str(k)];assert not np.any(vv&(aa==0));row=next(r for r in a['records'] if r['frame']==frame and r['track_id']==k);assert vv.sum()==row['visible_pixels'] and aa.sum()==row['in_frame_pixels']
 assert scene_count==72 and pixel_checks==36864 and sum(counts.values())==9216;checks['72_scene_label_arithmetic_and_exported_masks']=True
 result={'checks':checks,'baseline_parity':parity,'independent_comparisons':recomputed,'scene_state_counts':dict(counts),'label_pixel_checks':pixel_checks,'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formal_training_steps':6000,'smoke_steps':60,'verified_files':len(read(V/'RESULT_FILES.json')),'primary_metrics_changed_by_supplementary_fix':False}
 save(S/'CONFIGURATION_REVIEW.json',result)
 h=ev['combined']['heldout11'];comp=ev['comparisons'];rows=ev['by_sequence'];winloss={}
 for ref in ['N','R','C']:
  ds=[idx[n,'FULL']['IDF1']-idx[n,ref]['IDF1'] for n in names];winloss[ref]={'improved':sum(d>1e-9 for d in ds),'unchanged':sum(abs(d)<=1e-9 for d in ds),'worse':sum(d< -1e-9 for d in ds),'worst_delta':min(ds),'best_delta':max(ds)}
 realtable='\n'.join(f'| {LABELS[a]} | {h[a]["IDF1"]:.6f} | {h[a]["HOTA"]:.6f} | {h[a]["IDSW"]} |' for a in LABELS)
 comparetable='\n'.join(f'| 完整−{LABELS[a]} | {c["delta_IDF1"]:+.6f} | [{c["paired_video_bootstrap_95"][0]:+.6f}, {c["paired_video_bootstrap_95"][1]:+.6f}] |' for a,c in comp.items())
 pertable='\n'.join(f'| {n} | {idx[n,"FULL"]["IDF1"]:.3f} | {idx[n,"FULL"]["IDF1"]-idx[n,"N"]["IDF1"]:+.3f} | {idx[n,"FULL"]["IDF1"]-idx[n,"R"]["IDF1"]:+.3f} | {idx[n,"FULL"]["IDF1"]-idx[n,"C"]["IDF1"]:+.3f} |' for n in names)
 synthtable='\n'.join(f'| {LABELS[a]} | {train[a]["evaluation"]["synthetic_test"]["macro_F1"]:.6f} | {train[a]["evaluation"]["synthetic_test"]["visibility_MAE"]:.6f} | 2000 | 87877 |' for a in ['FULL','NOW','NOCTX'])
 sequences=read(V/'data/sequences.json')['sequences'];frames=sum(s['frames'] for s in sequences);physical=ev['physical_partial']['FULL'];trainingrun=pointer('train').name;evalrun=pointer('evaluate').name;adjudicaterun=pointer('adjudicate').name
 report=f'''# 可见性学习与真实视频身份恢复：完整实验报告

已在算力服务器完成8种骨骼动物、72段三维场景、三组专门模型训练，以及23段真实视频×8组完整对照。**实际出现小幅增量，但没有达到稳定超过基线的预冻结要求。** 新增效果集中于duck_3：相对C +0.647179 IDF1，其余10段留出不变；整体+0.262926点，区间下界0，低于BoT-SORT。不能以相对ByteTrack的全部收益冒充新模型贡献。

| 维度 | 本次实验的整体研究链路 |
|---|---|
| **问题** | 既有C有部分真实收益，但第四轮决策模型没有额外贡献；未做专门遮挡监督。检验“专门学习遮挡可见性，是否能减少身份记忆污染并改善恢复”。 |
| **机制假说** | H：遮挡外观污染最近身份记忆，准确可见性权重应比不加权、框重叠、打乱对象权重改善恢复；时序和周围场景应各有额外信息。竞争解释是检测覆盖/身份可分性/域差异限制，合成识别虽可学却不能改善真实身份。 |
| **针对性方法** | 五种训练资产、Llama验证、Pug/Fox测试；冻结图像骨干，查询锚点＋目标/周围ROI＋8步GRU，四类状态与比例监督。FULL/NOW/NOCTX同87,877参数、初始化/批次，各2000步；真实GT训练0条。固定前端、64维身份特征及C策略，只改最近5条记忆权重；N/R/C/GEOM/FULL/NOW/NOCTX/SHUFFLE共8组，23视频{frames:,}帧，主评价11段历史留出。 |
| **整体结果** | 合成FULL F1 0.90454，高于NOW 0.86104，低于NOCTX 0.91498。真实FULL−N +2.220702点CI[+0.468404,+3.437182]；FULL−R −4.171077点CI[−8.232949,−0.718009]；新增FULL−C +0.262926点CI[0,+0.499987]，幅度0.5及下界>0均未过。FULL−SHUFFLE −0.107516，CI跨0；对象权重的独立作用未成立。相对C无实测逐视频退化、HOTA +0.126315；11例物理部分遮挡严格成功均0，检测41/132，没有物理恢复新增收益。 |
| **研究决策** | **我主张接下来聚焦“遮挡前后同一身份与相似个体的可学习区分”，停止本轮可见性加权配置的追加训练。** 依据：专门遮挡能力已获得，新增任务收益仅1/11段，打乱对象权重不更差，强身份参照R仍明显领先；继续加大遮挡分类/调权重没有本轮证据支持。唯一下一步是冻结跨间断身份正负对与拒接监督的新数据合同，明确同个体前后正对、同场景近似个体负对和真实固定候选评价，随后才立新训练协议；本轮未启动新模型。若在独立物理事件上确认本权重的对象对应优势和≥0.5点增量，再考虑恢复此路线。 |

## 模型与专门训练

输入392维：目标ROI128＋周围ROI128＋首次目标锚点128＋几何8。Linear392→128/ReLU、GRU128→64、四类状态头及可见比例头。训练分类CE＋0.5SmoothL1，AdamW lr0.001/wd0.0001，batch128，seed20261019，last固定，不按真实结果挑checkpoint。正式6000步、工程smoke60步；训练作业Slurm145真实完成，无后续加训。详见[模型与创新边界](MODEL_AND_NOVELTY.md)。

| 模型 | 合成独立资产macro-F1 | 比例MAE | 正式步数 | 可训练参数 |
|---|---:|---:|---:|---:|
{synthtable}

几何标签9,216条：可见3,587、部分遮挡5,273、完全遮挡297、出视野59。训练查询有效6,124条，验证/测试各1,536条；20条训练初始观测因约定10%丢失未建立查询。测试完全遮挡仅54、出视野仅10；单seed、相同渲染器及场景模板，不能把0.905当真实视频准确率。

## 真实完整视频

23段完整公开GT视频，8历史fit仅背景、4校准仅描述、11历史留出为主评价；均非首次盲测。共同11视频有105,127个GT实例观测。N及所有C策略组均99,493条轨迹观测；R保留其100,895条原生输出，因此R比较是同前端/同GT的完整方法对照，不是同轨迹集合的单模块归因。GT用于评价；图像观察和身份决策只读当前/过去数据，允许C原有最多4帧确认缓冲。

| 方法 | IDF1（点） | HOTA（点） | IDSW |
|---|---:|---:|---:|
{realtable}

| 对比 | IDF1差值（点） | 2000次配对视频簇95%区间 |
|---|---:|---:|
{comparetable}

相对N：改善{winloss['N']['improved']}、不变{winloss['N']['unchanged']}、退化{winloss['N']['worse']}段，最差{winloss['N']['worst_delta']:+.3f}点。相对R：改善{winloss['R']['improved']}、不变{winloss['R']['unchanged']}、退化{winloss['R']['worse']}段，最差{winloss['R']['worst_delta']:+.3f}点。相对C：改善1、不变10、退化0段。有限样本的无退化不等于总体风险已证明为零。

| 视频 | FULL IDF1 | FULL−N | FULL−R | FULL−C |
|---|---:|---:|---:|---:|
{pertable}

## 物理遮挡与代理严格分开

复用11例AI逐帧复核的物理部分遮挡、165条边界观测，不是独立人工金标准；可靠完整遮挡事件仍0。原合同用全GT与全输出一对一Hungarian、IoU≥0.5：FULL与C遮挡后检测均41/132，保持先前身份的观测均8，错用既有其它身份观测均22，严格3帧前/3帧后同身份成功均0/11。没有实际物理遮挡恢复新增证据。

Run149补充层初版错误地逐标签独立取最大IoU，产生54/132；该数字明确作废。Run150仅重算该补充层，恢复旧一对一口径；原轨迹、全部主指标、bootstrap及科学裁决完全不变。GT时间间断代理另存各组gap_proxy.json，不能解释为已验证完整遮挡。

## 核验与产物

- 本机独立从IDTP/IDFP/IDFN复算全部IDF1和2000次区间；45行历史N/R/C指标一致；三组混淆矩阵/F1/MAE从预测数组复算。
- 23段全1权重精确重放C、几何不变、身份唯一和前缀因果性全部通过；真实长间断输入修复不改合成数组或已训权重。
- 72段全部标签重建后，36,864个独立像素检查通过；导出掩码可见⊆完整轮廓及面积/状态本机复核通过。
- [冻结协议](PROTOCOL.md) · [实现合同](IMPLEMENTATION_CONTRACT.md) · [配置复核JSON](CONFIGURATION_REVIEW.json) · [工程恢复](ENGINEERING_CORRECTIONS.md)；失败日志、源快照及负结果保留。
- 训练Run：`{trainingrun}`；真实评价：`{evalrun}`；补充裁决：`{adjudicaterun}`。服务器项目`/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2`，共享环境未修改。
- [离线展示与实验数据图](../../paper/presentations/可见性学习与真实视频身份恢复_20261019/index.html)。最佳增量和零增量案例均展示，不能仅凭选片下总体结论。
'''
 (S/'REPORT.md').write_text(report,encoding='utf-8');save(S/'DECISION.json',{'source_decision':decision,'direction':'cross-gap identity discrimination and rejection supervision','next_action':'freeze the cross-gap identity positive/negative data contract before a new training study','next_training_started':False,'win_loss_vs_comparators':winloss,'physical_metrics_source':adjudicaterun})
 save(S/'study.json',{**read(S/'study.json'),'execution_status':'completed','scientific_status':'utility_not_established','artifact_status':'checkpoints_and_results_completed_showcase_in_preparation','transfer_status':'archive_and_all_files_verified','training_steps_completed':6000,'smoke_training_steps':60,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':read(V/'experiments/dense_video_occlusion_transfer_20261019/study.json')['jobs'],'final_runs':{k:read(V/(k+'_pointer.json')) for k in ['train','evaluate','adjudicate']},'primary_result':comp})
 print('INDEPENDENT_REVIEW_PASSED',checks,winloss)
if __name__=='__main__':main()
