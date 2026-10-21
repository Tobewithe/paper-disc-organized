"""Frozen RFC-v1 paired estimands and decisions, also used by independent verification."""
from rfc_paths import ROOT
from posttrain_run import require
from run_s2_evaluation import image_effects, bootstrap

GROUPS = ('N', 'G', 'A', 'Q', 'J', 'AeqQ', 'QeqA', 'AeqJ', 'JeqA')
CONTRASTS = (('G','N'), ('A','N'), ('A','G'), ('Q','N'), ('Q','G'),
             ('A','Q'), ('A','J'), ('AeqQ','QeqA'), ('AeqJ','JeqA'))


def summarize(effects, damage):
    require(len(effects)==2925, 'all fixed-pair denominator changed')
    require(len({(r['image_id'],r['annotation_id']) for r in effects})==2925, 'duplicate pair')
    require(sum(r['population']=='T' for r in effects)==655, 'target denominator changed')
    images={g:{scope:image_effects(effects,g+'_delta',scope=='target')
               for scope in ('target','all')} for g in GROUPS}
    require(len(images['A']['target'])==257 and len(images['A']['all'])==508,'effective images changed')
    comparisons={}
    for a,b in CONTRASTS:
        comparisons[a+'-'+b]={scope:bootstrap({i:images[a][scope][i]-images[b][scope][i]
                                               for i in images[a][scope]}) for scope in ('target','all')}
    require(abs(comparisons['G-N']['target']['mean']-.010024491543988425)<1e-12,'G target replay drift')
    require(abs(comparisons['G-N']['all']['mean']-.0008909711434980322)<1e-12,'G overall replay drift')
    ag=comparisons['A-G']['target']; an=comparisons['A-N']['target']
    gates={'extra_utility_A_G':ag['mean']>=.005 and ag['ci95'][0]>0,
           'S2_repair_A_N':an['mean']>=.005 and an['ci95'][0]>0,
           'S2_all_pairs':comparisons['A-N']['all']['mean']>=-.001,
           'S2_damage':damage['A']['gate']=='damage_guardrail_passed',
           'identity_and_configuration':True}
    mechanism={k:comparisons[k]['target']['ci95'][0]>0 for k in
               ('A-Q','A-J','AeqQ-QeqA','AeqJ-JeqA')}
    passed=all(gates.values())
    precision_only=(all(v for k,v in gates.items() if k!='S2_damage') and
                    damage['A']['gate']=='blocked_insufficient_precision')
    return {'contrasts':comparisons,'gates':gates,'mechanism_gates':mechanism,
            'utility_status':'passed' if passed else 'blocked' if precision_only else 'failed',
            'mechanism_supported':all(mechanism.values()),'damage':damage,
            'target_pairs':655,'all_pairs':2925,'dev_images':512,
            'scope':'reused COCO development; one training seed; not independent confirmation',
            'next_action':('recommend same A last checkpoint full S3; needs separate authorization' if passed else
                           'close RFC-v1 investment, retain G, integrate evidence into manuscript; no further training or evaluation')}
