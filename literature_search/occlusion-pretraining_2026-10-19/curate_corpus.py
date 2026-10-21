"""Apply evidence-backed metadata corrections and implementation-first reading order.

Raw and automated-merge records remain intact. No model experiment is run.
"""
import json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent
rows = json.loads((ROOT/'corpus.merged.json').read_text(encoding='utf-8'))
corrections = []

def fix(row, changes, source, reason):
    before = {k:row.get(k) for k in changes}
    row.update(changes)
    corrections.append({'title':row['title'], 'before':before, 'after':changes, 'source':source, 'reason':reason})

for r in rows:
    aid = r.get('arxiv_id')
    if aid == '2604.20748':
        fix(r, {'venue':'arXiv preprint; journal publication not verified'},
            'https://arxiv.org/abs/2604.20748',
            'Remove unverified third-party review/withdrawal status. A journal-style PDF header alone does not verify acceptance.')
    if aid == '2211.03726':
        fix(r, {'year':2022, 'venue':'NeurIPS 2022 Datasets and Benchmarks', 'doi':'10.52202/068431-0989'},
            'https://proceedings.neurips.cc/paper_files/paper/2022/hash/58168e8a92994655d6da3939e7cc0918-Abstract-Datasets_and_Benchmarks.html',
            'Official proceedings corrects a CVPR2023 venue in snowball metadata.')
    if aid == '2410.11831':
        fix(r, {'year':2025, 'preprint_year':2024, 'venue':'ICCV 2025'},
            'https://openaccess.thecvf.com/content/ICCV2025/papers/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.pdf',
            'Distinguish initial arXiv year from verified conference year.')
    if aid == '2305.03052':
        fix(r, {'venue':'CVPR 2023'}, 'https://tcow.cs.columbia.edu/', 'Official project identifies venue.')
    if aid == '2503.06261':
        fix(r, {'venue':'CVPR 2025'}, 'https://weient.github.io/sameo.github.io/', 'Official project identifies venue.')
    if aid == '2103.14258':
        fix(r, {'venue':'ICCV 2021'}, 'https://openaccess.thecvf.com/content/ICCV2021/html/Tokmakov_Learning_To_Track_With_Object_Permanence_ICCV_2021_paper.html', 'Conference source in discovery record.')
    if aid == '1509.01329':
        fix(r, {'year':2017, 'preprint_year':2015}, 'https://doi.org/10.1109/CVPR.2017.320', 'Do not mix first preprint year and published venue year.')
    if 'SyDog-Video:' in r['title']:
        fix(r, {'year':2024, 'online_year':2023, 'venue':'International Journal of Computer Vision 132:1986–2002',
                'authors':['Moira Shooter','Charles Malleson','Adrian Hilton'],
                'pdf_url':'https://link.springer.com/content/pdf/10.1007/s11263-023-01946-z.pdf'},
            'https://link.springer.com/article/10.1007/s11263-023-01946-z',
            'Publisher verifies 2023 online release, 2024 issue, and author list.')
    if r.get('doi','').lower() == '10.21105/joss.04901' if r.get('doi') else False:
        fix(r, {'authors':['Maximilian Denninger','Dominik Winkelbauer','Martin Sundermeyer','Wout Boerdijk','Markus Knauer','Klaus H. Strobl','Matthias Humt','Rudolph Triebel']},
            'https://joss.theoj.org/papers/10.21105/joss.04901', 'Complete authors checked against official JOSS BibTeX.')
        r.pop('author_note', None)
    if r.get('venue') and '\ufffd' in r['venue']:
        r['venue'] = r['venue'].replace('\ufffdC','–').replace('\ufffd','–')
    r['record_kind'] = 'engineering_documentation' if r['title'].startswith('Cosmos Transfer') else 'research_work'
    r['abstract_note'] = 'Discovery description; may be a researcher summary rather than the original abstract. Use the linked original for quotations.'
    r['automatic_rank'] = r['rank']

priority = [
 '2503.06261','10.1109/cvpr52734.2025.01886','2604.20748','2305.03052',
 '2203.03570','10.21105/joss.04901','2408.00714','2511.16719','2409.01573',
 '2410.11831','2412.04623','2411.19210','10.1109/cvpr52688.2022.00914',
 '2103.14258','2210.12733','2309.13248','2409.14095','2307.15055',
 '10.1109/cvpr.2019.00322','2312.12433','2311.07761','2306.08637',
 '2211.03726','2302.01872','2102.01558','2401.14398'
]
def identity_keys(r):
    return [str(r.get(k) or '').lower() for k in ('arxiv_id','doi')]
def sort_key(r):
    idx = min([priority.index(k) for k in identity_keys(r) if k in priority] or [len(priority)])
    return idx, r['automatic_rank']
rows.sort(key=sort_key)

peripheral = {
 'Monocular Quasi-Dense 3D Object Tracking':'3D detection/tracking background; not evidence for proposed 2D occlusion pretraining.',
 'Amodal Segmentation for Laparoscopic Surgery Video Instruments':'Domain-specific background; not used to select the general animal/object model.',
 'Temporally Consistent Unpaired Multi-domain Video Translation by Contrastive Learning':'Metadata discovery only; not used to support a method claim.',
 'Occlusion-Supervised Amodal Instance Segmentation':'SSRN discovery metadata; independent primary method/venue verification incomplete.',
 'A2vis: Amodal-Aware Approach to Video Instance Segmentation':'SSRN discovery metadata; independent primary method/venue verification incomplete.',
 'A segmentation-aware object detection model with occlusion handling':'Historical background only.',
 'Occlusion-Aware Object Localization, Segmentation and Pose Estimation':'Historical background only.'
}
screening = []
for i,r in enumerate(rows,1):
    r['rank'] = i
    r['screening'] = 'background_not_used_for_core_claims' if r['title'] in peripheral else 'included_reference'
    if r['title'] in peripheral:
        r['screening_reason'] = peripheral[r['title']]
    r['reading_priority_note'] = 'Manually ordered for this implementation question after automated deduplication; source counts do not represent independent replications.'
    screening.append({'rank':i,'title':r['title'],'kind':r['record_kind'],'decision':r['screening'],'reason':r.get('screening_reason')})

(ROOT/'corpus.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'metadata_corrections.json').write_text(json.dumps(corrections,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'screening.json').write_text(json.dumps(screening,ensure_ascii=False,indent=2),encoding='utf-8')
lines = [f'# 遮挡预训练文献与资源库：{len(rows)} 条去重记录\n',
         '编号为实施问题优先级，不是引用量排名。`research_work` 包含会议/期刊与预印本；工程文档另标。来源重复发现不等于独立实验证据。字段修正与筛选决定均保留。\n']
for r in rows:
    link = r.get('url') or r.get('pdf_url') or ('https://doi.org/'+r['doi'])
    lines.append(f"## #{r['rank']:02d} {r['title']}\n")
    lines.append(f"{r.get('year','')} · {r.get('venue') or 'venue not verified'} · {r['record_kind']} · {r['screening']}\n")
    authors = ', '.join((r.get('authors') or [])[:3])
    if len(r.get('authors') or []) > 3 or r.get('author_note'): authors += ' et al.（详见原文）'
    lines.append(authors+'\n')
    lines.append(f"[原始来源]({link})"+(f" · [PDF]({r['pdf_url']})" if r.get('pdf_url') else '')+'\n')
    lines.append('来源记录：'+', '.join(r.get('sources',[]))+'\n')
    if r.get('screening_reason'): lines.append('未用于核心方法判断：'+r['screening_reason']+'\n')
(ROOT/'corpus.md').write_text('\n'.join(lines),encoding='utf-8')
mapping={k:r['rank'] for r in rows for k in identity_keys(r) if k}
summary_path=ROOT/'summary.md'
summary=summary_path.read_text(encoding='utf-8')
summary=re.sub(r'\{\{([^}]+)\}\}',lambda m:f"[#{mapping[m.group(1).lower()]:02d}]",summary)
summary_path.write_text(summary,encoding='utf-8')
print(json.dumps({'records':len(rows),'core_or_secondary':sum(r['screening']=='included_reference' for r in rows),'corrections':len(corrections)}))
