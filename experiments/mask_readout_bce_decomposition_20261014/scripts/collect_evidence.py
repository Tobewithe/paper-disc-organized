"""Independent arithmetic verification of returned eight-cell measurements."""
import argparse
from collections import Counter
import math
from pathlib import Path
import sys

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[1]
PROTO=ROOT/'experiments/mask_readout_proto_consistency_20261014/runs/RUN_PROTO_AUDIT_FULL_2a0b0e0afeb64cd7a1300a10aac1a30f'
sys.path.insert(0,str(ROOT/'experiments/mask_readout_fit_diagnostic_20261014/scripts'))
import collect_diagnostic_evidence as transfer
transfer.STUDY=STUDY
read,rows,digest,save=transfer.read,transfer.rows,transfer.digest,transfer.save
NAMES=('TN_to_TN','TN_to_FP','FP_to_TN','FP_to_FP','FN_to_FN','FN_to_TP','TP_to_FN','TP_to_TP')
STABLE=('TN_to_TN','FP_to_FP','FN_to_FN','TP_to_TP')
FIELDS=('zero_loss','B_loss','delta_loss','decrease_credit','increase_debit')


def near(a,b,tol=1e-12):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,(a,b,tol)


def audit(run):
    meta,contract,summary=read(run/'run.json'),read(run/'EVAL_CONTRACT.json'),read(run/'SUMMARY.json')
    assert meta['execution_status']=='completed' and read(run/'CONFIG_AUDIT.json')['status']=='passed'
    assert read(run/'LOCAL_TRANSFER_VERIFICATION.json')['status']=='passed'
    for i,(relative,sha) in enumerate(read(run/'SOURCE_MANIFEST.json')['files'].items()):
        assert digest(run/'source_snapshot'/(f'{i:02d}_'+Path(relative).name))==sha
    assert digest(run/'PROTOCOL.md')==meta['protocol_sha256']
    assert digest(run/'EVAL_CONTRACT.json')==meta['contract_sha256']
    assert contract['new_training_steps']==0
    state=read(run/'FROZEN_MODEL_STATE.json')
    assert state['before']==state['after'] and state['new_optimizer_steps']==0 and state['model_eval']
    assert not state['parameters_require_grad'] and not state['parameter_gradients_present']
    key=lambda r:(r['image_id'],r['annotation_id'],r['training_output_id'])
    ids=set(contract['image_ids'])
    previous={key(r):r for r in rows(PROTO/'PAIR_PROTO_AUDIT.jsonl') if r['image_id'] in ids}
    expected={k for k,r in previous.items() if r['direction']=='loss_down_iou_down' and r['proto_direction']=='down'}
    actual=rows(run/'PAIR_BCE_DECOMPOSITION.jsonl')
    assert len(actual)==len(expected)==len({key(r) for r in actual})
    assert {key(r) for r in actual}==expected
    credits,shares,stable_credits=[],[],[]
    main_count=0
    dominance=Counter()
    empty_count=0
    max_closure=0.
    for r in actual:
        old=previous[key(r)]
        assert all(r[k]==v for k,v in old.items())
        d=r['decomposition']; g=d['groups']
        assert set(g)==set(NAMES)
        assert sum(c['pixels'] for c in g.values())==r['zero_proto']['roi_pixels']
        for c in g.values():
            assert type(c['pixels']) is int and c['pixels']>=0
            assert all(math.isfinite(c[f]) for f in FIELDS)
            assert c['decrease_credit']>=0 and c['increase_debit']>=0
            near(c['B_loss']-c['zero_loss'],c['delta_loss'])
            near(c['increase_debit']-c['decrease_credit'],c['delta_loss'])
        for f in FIELDS:
            total=math.fsum(g[n][f] for n in NAMES)
            near(total,d['totals'][f])
            near(math.fsum(g[n][f] for n in NAMES[4:]),d['foreground'][f])
            near(math.fsum(g[n][f] for n in NAMES[:4]),d['background'][f])
        err=abs(math.fsum(g[n]['delta_loss'] for n in NAMES)-(old['B_gt_bce']-old['zero_gt_bce']))
        assert err<=1e-7
        near(err,r['BCE_delta_closure_abs_error'],1e-14)
        max_closure=max(max_closure,err)
        counts={n:g[n]['pixels'] for n in NAMES}
        before={'tp':counts['TP_to_TP']+counts['TP_to_FN'],'fp':counts['FP_to_FP']+counts['FP_to_TN'],
                'fn':counts['FN_to_FN']+counts['FN_to_TP'],'tn':counts['TN_to_TN']+counts['TN_to_FP']}
        after={'tp':counts['TP_to_TP']+counts['FN_to_TP'],'fp':counts['FP_to_FP']+counts['TN_to_FP'],
               'fn':counts['FN_to_FN']+counts['TP_to_FN'],'tn':counts['TN_to_TN']+counts['FP_to_TN']}
        assert before==d['counts_zero'] and after==d['counts_B']
        assert all(before[k]==old['zero_proto'][k] and after[k]==old['B_proto'][k] for k in before)
        gt=before['tp']+before['fn']; u0,u1=gt+before['fp'],gt+after['fp']
        if u0 and u1:
            fg=(after['tp']-before['tp'])*(1/u0+1/u1)/2
            bg=(before['tp']+after['tp'])*(1/u1-1/u0)/2
        else:
            assert gt==0
            fg,bg=0.,old['delta_proto_iou']
            empty_count+=1
        parts=d['iou_decomposition']
        near(fg,parts['foreground_TP_term']); near(bg,parts['background_FP_term'])
        near(fg+bg,old['delta_proto_iou'])
        harm_fg,harm_bg=max(-fg,0),max(-bg,0)
        dom='foreground' if harm_fg-harm_bg>1e-12 else 'background' if harm_bg-harm_fg>1e-12 else 'tie'
        assert dom==parts['dominant_negative_term']; dominance[dom]+=1
        assert parts['empty_union_boundary']==(not(u0 and u1))
        credit=math.fsum(g[n]['decrease_credit'] for n in NAMES)
        stable=math.fsum(g[n]['decrease_credit'] for n in STABLE)
        assert credit>0
        share=stable/credit
        near(share,d['stable_credit_share'])
        assert d['stable_credit_majority']==(share>.5)
        near(stable,d['stable_decrease_credit'])
        near(g['FP_to_TN']['decrease_credit']+g['FN_to_TP']['decrease_credit'],d['corrected_decrease_credit'])
        shares.append(share);credits.append(credit);stable_credits.append(stable);main_count+=share>.5
    n=len(actual)
    assert n==summary['target_pairs'] and main_count==summary['stable_credit_majority_pairs']
    near(main_count/n,summary['stable_credit_majority_fraction'])
    near(math.fsum(shares)/n,summary['equal_pair_mean_stable_credit_share'])
    near(math.fsum(stable_credits)/math.fsum(credits),summary['ratio_of_summed_pair_normalized_stable_credit'])
    assert dict(dominance)==summary['IoU_negative_term_dominance']
    assert empty_count==summary['empty_union_boundary_pairs']
    near(max_closure,summary['max_BCE_delta_closure_abs_error'],1e-14)
    for name in ('totals','foreground','background'):
        for f in FIELDS:
            near(math.fsum(r['decomposition'][name][f] for r in actual)/n,summary['equal_pair_mean_contributions'][name][f])
    for name in NAMES:
        assert sum(r['decomposition']['groups'][name]['pixels'] for r in actual)==summary['groups'][name]['pixels']
        for f in FIELDS:
            near(math.fsum(r['decomposition']['groups'][name][f] for r in actual)/n,summary['groups'][name][f])
    for f in ('foreground_TP_term','background_FP_term','delta_iou'):
        near(math.fsum(r['decomposition']['iou_decomposition'][f] for r in actual)/n,summary['IoU_terms_mean'][f])
    checks=rows(run/'IMAGE_CHECKS.jsonl')
    old_checks={r['image_id']:r for r in rows(PROTO/'IMAGE_CHECKS.jsonl')}
    assert len(checks)==len(ids) and {r['image_id'] for r in checks}==ids
    assert all(r==old_checks[r['image_id']] for r in checks)
    assert sum(r['valid_supervision_pairs'] for r in checks)==summary['measurement']['all_supervised_pairs_verified']
    if summary['scope']=='full':
        assert n==2159 and len(checks)==2000 and summary['measurement']['all_supervised_pairs_verified']==14268
        assert summary['diagnostic_decision']['hypothesis_supported']==(main_count/n>.5)
    result={'status':'passed','scope':summary['scope'],'target_pairs':n,'stable_credit_majority_pairs':main_count,
            'stable_credit_majority_fraction':main_count/n,'IoU_negative_term_dominance':dict(dominance),
            'max_BCE_delta_closure_abs_error':max_closure,'integer_counts_and_all_group_accounting_recomputed':True,
            'old_pair_fields_and_image_losses_unchanged':True,'new_training_steps':0,
            'summary_sha256':digest(run/'SUMMARY.json'),'configuration_audit_sha256':digest(run/'CONFIG_AUDIT.json'),
            'audit_script_sha256':digest(Path(__file__)),'claim_limit':'independent arithmetic and identity check, not local model re-inference'}
    save(run/'LOCAL_RESULT_RECHECK.json',result)
    print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('package','verify','audit'));p.add_argument('runs',nargs='+',type=Path)
    args=p.parse_args()
    for run in args.runs:
        {'package':transfer.package,'verify':transfer.unpack_verify,'audit':audit}[args.action](run)
