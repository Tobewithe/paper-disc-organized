"""Summarize actual retrieval outcomes and verify local research-artifact integrity."""
import collections, hashlib, json, pathlib, re

ROOT=pathlib.Path(__file__).resolve().parent
corpus=json.loads((ROOT/'corpus.json').read_text(encoding='utf-8'))
manifest=json.loads((ROOT/'pdfs/manifest.json').read_text(encoding='utf-8'))
manifest.sort(key=lambda r:r['rank'])
(ROOT/'pdfs/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
byrank={r['rank']:r for r in corpus}
pdfbyrank={r['rank']:r for r in manifest}
errors=[]
for r in manifest:
    if r['title'] != byrank[r['rank']]['title']: errors.append('rank/title mismatch '+str(r['rank']))
    if r['status']=='acquired':
        p=ROOT/r['file']; data=p.read_bytes()
        if not data.startswith(b'%PDF-'): errors.append('invalid PDF '+str(r['rank']))
        if hashlib.sha256(data).hexdigest()!=r['sha256']: errors.append('hash mismatch '+str(r['rank']))
        if not (ROOT/r['text_file']).exists(): errors.append('missing text '+str(r['rank']))

facts=[
 (1,'freeze','SAMEO trains an amodal decoder with frozen EfficientSAM encoders; detector-box prompting is existing prior art.'),
 (2,'separation-to-fusion','SAMBA already combines modal/amodal separation-to-fusion and occlusion-rate-associated experts.'),
 (3,'Target-Aware','Amodal SAM already combines an adapter, synthetic target occlusion and consistency objectives; a separate AmodalSAM2 paper was not verified.'),
 (4,'causal','TCOW uses synthetic occlusion/containment supervision with causal tracking; it is not an offline-only comparator.'),
 (5,'PyBullet','Kubric supports physically structured rendered supervision; our existing Blender assets need not be recreated.'),
 (7,'occlusion prediction head','SAM2 already includes an occlusion prediction head and occlusion memory embedding.'),
 (7,'mosaic','SAM2 uses video augmentation including same-clip mosaic for similar-looking distractors.'),
 (9,'Distillation','OED is prior art for occluded/unoccluded teacher-student feature learning for detection; its demonstrated setting is not universal YOLO video transfer.'),
 (10,'freeze it','CoTracker3 freezes the confidence/visibility prediction layer during pseudo-real trajectory training.'),
 (11,'diffusion','DiffVAS uses diffusion priors for video amodal segmentation; generated hidden content is not an observed identity label.'),
 (12,'composit','TABE uses layered video composition for amodal evaluation; distinguish its evaluation task from causal instance identity restoration.'),
 (13,'object-occluder-occluded','WALT already learns object/occluder/occluded layer representations from automatically generated training supervision.'),
 (18,'visibility','PointOdyssey supplies synthetic trajectories and visibility; point visibility is not equivalent to instance-level physical occlusion.'),
 (19,'amodal','SAIL-VOS supplies amodal and modal video segmentation labels from a rendered environment.'),
 (20,'bounding','TAO-Amodal provides amodal box supervision; do not reinterpret boxes as hidden-region pixel ground truth.')
]
evidence=[]
for rank,term,claim in facts:
    m=pdfbyrank.get(rank,{})
    item={'corpus_rank':rank,'title':byrank[rank]['title'],'claim_summary':claim,
          'evidence_type':'primary paper; selected method/data sections read',
          'url':byrank[rank].get('url'),'literal_search_term':term}
    if m.get('status')=='acquired':
        txt=(ROOT/m['text_file']).read_text(encoding='utf-8').replace('\x00','')
        pos=txt.casefold().find(term.casefold())
        pages=re.findall(r'PAGE (\d+)',txt[:pos]) if pos>=0 else []
        item.update(text_file=m['text_file'],pdf_file=m['file'],search_term_found=pos>=0,
                    first_matching_pdf_page=int(pages[-1]) if pages else None)
    evidence.append(item)
evidence.extend([
 {'evidence_type':'official implementation/release','claim_summary':'SAM2.1 Hiera-small has about46M parameters and public custom image/video fine-tuning support; suitability here is our recommendation, not a measured result.','url':'https://github.com/facebookresearch/sam2'},
 {'evidence_type':'official implementation/release','claim_summary':'SAM3.1 released March27 2026 with Object Multiplex shared-memory multi-object tracking; SAM3 repository lists848M parameters and newer dependencies.','url':'https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md'},
 {'evidence_type':'official implementation','claim_summary':'DiffVAS default amodal training reports about120h on8A6000 with online depth estimation; this is not our benchmark.','url':'https://github.com/Kaihua-Chen/diffusion-vas'},
 {'evidence_type':'existing local source, read-only','claim_summary':'Generator already writes instance IDs, per-instance amodal masks, visibility/truncation and occluder pixel counts. Persistent per-pixel flow/depth and clean paired RGB were not seen in its output contract.','file':'../../shared/dense_video_occlusion/generate.py','lines':[132,175]},
 {'evidence_type':'completed prior experiment, not a new run','claim_summary':'Existing visibility classification did not yield stable incremental real identity recovery; supports changing the supervised capability target, not claiming all occlusion representations fail.','file':'../../experiments/dense_video_occlusion_transfer_20261019/REPORT.md'},
 {'evidence_type':'proposed, not experimentally supported','claim_summary':'Controlled occluder substitution plus competing-instance supervision may teach transferable target evidence separation. Must compare with same-data/same-budget ordinary augmentation and distillation.'}
])
(ROOT/'evidence_matrix.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')

raw_counts={}
for p in (ROOT/'raw').glob('*.json'):
    if p.name.endswith('.status.json'): continue
    rows=json.loads(p.read_text(encoding='utf-8'))
    if isinstance(rows,dict): rows=rows.get('results',rows.get('papers',[]))
    raw_counts[p.name]=len(rows)
status={
 'date':'2026-10-19','status':'completed_research_synthesis','depth':3,
 'retrieval_scope':'Focused deep multi-source search with primary-source reference snowballing; not an exhaustive systematic review.',
 'raw_record_counts':raw_counts,'raw_total':sum(raw_counts.values()),'unique_records':len(corpus),
 'record_kinds':dict(collections.Counter(r['record_kind'] for r in corpus)),
 'included_core_or_secondary':sum(r['screening']=='included_reference' for r in corpus),
 'background_not_used_for_core_claims':sum(r['screening']!='included_reference' for r in corpus),
 'pdf_attempted':len(manifest),'pdf_acquired':sum(r['status']=='acquired' for r in manifest),
 'pdf_not_acquired':[{'rank':r['rank'],'title':r['title'],'attempts':r['attempts']} for r in manifest if r['status']!='acquired'],
 'source_limitations':[
   'Semantic Scholar API: four theme queries and seed citation retrieval returnedHTTP429; records are transparently primary-source fallbacks, not successfulS2 query results.',
   'OpenAlex/Crossref: some direct metadata results succeeded, some searches were rate-limited; exact requests/errors in source status.',
   'DBLP: six JSONAPI attempts returned nonJSON; one verified record recovered byDBLP web search.',
   'Open repository coverage used actual accessible official project/dataset pages; not all repositories or open-access aggregators were searched successfully.',
   'Forward citation graph was not recovered; snowballing primarily followed backward references and directly linked official related work.',
   '41 downloaded PDFs does not mean all41 were read in full; core method/label/implementation claims were checked selectively, with deeper reading of the closest prior art.'
 ],
 'novelty_status':'Candidate contribution, not established originality or empirical efficacy.',
 'new_training_started':False,'rules_or_old_protocols_changed':False,
 'user_clarification':'Existing self-modelled3D scenes/data are the source. Synthetic training supervision is automatic; manual annotation is not a prerequisite.',
 'verification_errors':errors
}
(ROOT/'SEARCH_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
plan=json.loads((ROOT/'SEARCH_PLAN.json').read_text(encoding='utf-8'))
plan['status']='completed_research_synthesis'
plan['terminology_correction']='Initial query AmodalSAM2 resolved to Amodal SAM arXiv2604.20748; no separate paper verified.'
(ROOT/'SEARCH_PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
text=(ROOT/'summary.md').read_text(encoding='utf-8')
if '{{' in text: errors.append('unresolved reference placeholder')
for match in re.finditer(r'\]\(([^)]+)\)',text):
    target=match.group(1)
    if target.startswith(('http://','https://','#')):continue
    if not (ROOT/target.split('#')[0]).exists():errors.append('missing report target '+target)
if errors: raise SystemExit(json.dumps(errors,ensure_ascii=False))
print(json.dumps({'raw':sum(raw_counts.values()),'unique':len(corpus),'pdf_acquired':status['pdf_acquired'],'report_links':'passed','pdf_hashes':'passed'},ensure_ascii=False))
