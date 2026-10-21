"""One-shot post-run evidence audit/report; no model, training, retries or polling."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess
import traceback

ROOT = Path('/srv/gpu-lab/users/a25023104/projects/paper-disc-organized')
STUDY = ROOT/'experiments/mask_branch_joint_replay_20261018'
ARMS = ('G_budget', 'C_native', 'PC_native')

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def write(p, value):
    assert not p.exists(), f'No overwrite: {p}'
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    out=STUDY/'closure/results'
    out.mkdir(parents=True,exist_ok=False)
    result={'started_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'scope':'existing artifacts only; no model or metric rerun; not an independent statistical recomputation',
            'training_steps_added':0,'model_forwards':0,'source_sha256':sha(Path(__file__)),
            'status':'running','checks':{},'runs':{}}
    report=None
    try:
        cfg=read(STUDY/'CONFIG.json'); frozen=read(STUDY/'FROZEN_MANIFEST.json')
        for f in frozen['files']: assert sha(ROOT/f['path'])==f['sha256'], 'Frozen file changed: '+f['path']
        result['checks']['frozen_files']=len(frozen['files'])
        pointers=read(STUDY/'POINTERS.json')
        keys=('PREPARE','NATIVE','SMOKE',*[f'TRAIN_{a}' for a in ARMS],*[f'EVAL_{a}' for a in ARMS],'STATISTICS')
        paths={}
        for key in keys:
            assert key in pointers, 'Required completed stage missing: '+key
            p=STUDY/'runs'/pointers[key]['run_id']; paths[key]=p
            meta=read(p/'run.json')
            assert meta['execution_status']=='completed', 'Incomplete stage: '+key
            assert meta['config_sha256']==sha(STUDY/'CONFIG.json'), 'Config mismatch: '+key
            assert sha(p/'SUMMARY.json')==pointers[key]['summary_sha256'], 'Summary mismatch: '+key
            manifest=read(p/'ARTIFACT_MANIFEST.json')
            for name,digest in manifest.items(): assert sha(p/name)==digest, 'Artifact changed: '+key+'/'+name
            result['runs'][key]={'run_id':p.name,'artifact_hashes_verified':len(manifest)}
        assert read(paths['NATIVE']/'SUMMARY.json')['migration_status']=='passed'
        assert read(paths['SMOKE']/'SUMMARY.json')['status']=='passed'
        for arm in ARMS:
            p=paths['TRAIN_'+arm]; summary=read(p/'SUMMARY.json'); audit=read(p/'CONFIG_AUDIT.json')
            assert summary['optimizer_steps']==15000 and summary['images_seen']==60000 and summary['epochs']==3
            assert audit['status']=='passed' and audit['same_cross_arm_native_inputs_and_GT_join'] is True
            assert sha(p/'last.pt')==summary['checkpoint_sha256']==audit['final_checkpoint_sha256']
            assert read(paths['EVAL_'+arm]/'SUMMARY.json')['checkpoint_sha256']==summary['checkpoint_sha256']
        result['checks']['formal_steps']=45000
        stat=read(paths['STATISTICS']/'SUMMARY.json')
        assert stat['images']==5000 and stat['bootstrap']['replicates']==2000
        assert stat['bootstrap']['seed']==cfg['bootstrap']['seed']
        native=read(paths['NATIVE']/'SUMMARY.json')
        damage=read(paths['STATISTICS']/'DAMAGE_CENSUS.json')
        result.update(status='completed_evidence_audit_passed',scientific_gate_passed=stat['primary_gate_passed'],
                      scientific_decision=stat['decision'],unique_next_action=stat['unique_next_action'])
        if stat['primary_gate_passed']:
            decision='优先采用联合掩码分支作为方法基座；唯一下一步是设计开发外确认及容量对照，本次不自动新增训练。'
        else:
            eligible=['Ns']+[a for a in ('G_old_s',*ARMS) if damage[a]['damage_fraction']<=.01 and (a!='PC_native' or stat['guards']['inference_cost'])]
            best=max(eligible,key=lambda a:stat['Mask_AP_points'][a])
            decision=f'本配置未通过冻结主门槛，停止追加训练。优先保留本次低损伤候选中实测AP最高的{best}，将其收益与可修机会、损伤代价纳入论文证据；唯一下一步是人工复核结案主张及适用边界，不自动开新实验。'
        rows=[]
        for name in ('Ns','G_old_s',*ARMS):
            ap=stat['Mask_AP_points'][name]
            rows.append(f"| {name} | {ap:.6f} | {ap-stat['Mask_AP_points']['Ns']:+.6f} | {ap-stat['Mask_AP_points']['G_old_s']:+.6f} | {damage[name]['damaged']}/{stat['native_success_denominator']} |")
        comparisons=[]
        for name,test in stat['primary_comparisons'].items():
            comparisons.append(f"| {name} | {test['AP_points']:+.6f} | [{test['ci95'][0]:+.6f}, {test['ci95'][1]:+.6f}] | {test['minimum_AP_points']:.2f} | {'通过' if test['passed'] else '未过'} |")
        report='\n'.join([
            '# 原生掩码分支联合适配：服务器自动结案记录','',
            '三臂各完成20k图×3轮、15000次更新及完整5000图评价。本记录依据已完成Run自动整理；配置和产物哈希已复核，尚未进行本机独立统计复算。','',
            '| 维度 | 本次实验的整体研究链路 |','|---|---|',
            '| 问题 | 旧G已有收益，但固定P的系数修复尚未充分兑现已知机会，需要检验原生分支适配是否提供额外实用收益。 |',
            '| 机制假说 | 原型和系数头共同适配可能改善修复；竞争解释是训练预算或仅原生系数头已足够。参数容量未匹配，不能单独归因为纯协同机制。 |',
            '| 针对性方法 | G_budget、C_native、PC_native共享20k、三轮、监督与优化配方；固定检测及9216面积门控，比较同服务器原生/旧G及两个预算参照，检查全成功GT损伤与成本。 |',
            f"| 整体结果 | 主门槛{'通过' if stat['primary_gate_passed'] else '未通过'}；各方法相对原生与旧G增量、四项PC区间和护栏见下表。 |",
            '| 研究决策 | '+decision+' |','',
            '| 方法 | Mask AP | 相对原生AP点 | 相对旧G AP点 | 原成功GT损伤 |','|---|---:|---:|---:|---:|',*rows,'',
            '| PC必要比较 | AP差值 | 配对95%区间 | 最小增量 | 门槛 |','|---|---:|---|---:|---|',*comparisons,'',
            '护栏：'+json.dumps(stat['guards'],ensure_ascii=False)+'。',
            '冻结裁决：'+stat['decision']+'。',
            '冻结唯一后继：'+stat['unique_next_action']+'。',
            '旧G参照为同服务器重放，历史笔记本分数不混作本次分母。全部验证图曾参与开发，本轮不是独立外部确认。','',
            '证据Run：'+', '.join(paths[k].name for k in keys)+'.','',
            '配置/文件核验详见 CONFIGURATION_AUDIT.json；所有模型、预测和统计文件保留在各Run目录。'])
    except BaseException as error:
        result.update(status='incomplete_or_evidence_audit_failed',error=str(error))
        (out/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        report='# 服务器执行未完成或证据核验未通过\n\n'+str(error)+'\n\n没有追加训练、放宽门槛或伪造缺失结果。保留原Run，需按真实错误作工程核对。'
    result['finished_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    write(out/'CONFIGURATION_AUDIT.json',result)
    (out/'REPORT.md').write_text(report+'\n',encoding='utf-8')
    state=read(STUDY/'study.json');state.update(execution_status=result['status'],closure='closure/results/REPORT.md',
        scientific_status=result.get('scientific_decision','not_adjudicated'),last_verified_utc=result['finished_at_utc'])
    if result['status']=='completed_evidence_audit_passed':state['actual_formal_training_steps_verified']=45000
    (STUDY/'study.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
