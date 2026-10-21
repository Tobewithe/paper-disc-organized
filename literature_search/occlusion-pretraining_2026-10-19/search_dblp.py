import json, pathlib, urllib.request, urllib.parse, time
ROOT=pathlib.Path(__file__).resolve().parent
queries=['Kubric','amodal segmentation','CoTracker','synthetic video','occlusion distillation','SAM 2 Segment Anything']
rows=[]; status=[]
for query in queries:
    url='https://dblp.org/search/publ/api?'+urllib.parse.urlencode({'q':query,'format':'json','h':15})
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ResearchLiteratureReader/1.0'}),timeout=15) as r:
            data=json.load(r)
        hits=data.get('result',{}).get('hits',{}).get('hit',[])
        if isinstance(hits,dict): hits=[hits]
        for hit in hits:
            item=hit.get('info',{})
            authors=item.get('authors',{}).get('author',[])
            if not isinstance(authors,list): authors=[authors]
            authors=[a.get('text','') if isinstance(a,dict) else a for a in authors]
            year=int(item['year']) if str(item.get('year','')).isdigit() else None
            if year and year>2026:continue
            ee=item.get('ee')
            if isinstance(ee,list):ee=ee[0] if ee else None
            doi=item.get('doi')
            if not doi and isinstance(ee,str) and 'doi.org/' in ee: doi=ee.split('doi.org/',1)[1]
            rows.append({'title':item.get('title','').rstrip('.'),'authors':authors,'year':year,'venue':item.get('venue'),'doi':doi,'arxiv_id':None,'pdf_url':None,'url':ee or item.get('url'),'abstract':'','source':'dblp_api','query':query,'metadata_url':item.get('url')})
        status.append({'query':query,'url':url,'status':'ok','hits':len(hits)})
        print('ok',query,len(hits),flush=True)
    except Exception as exc:
        status.append({'query':query,'url':url,'status':'failed','error':str(exc)})
        print('failed',query,str(exc),flush=True)
    time.sleep(1)
(ROOT/'raw').mkdir(exist_ok=True)
(ROOT/'raw/F_dblp.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'raw/F_dblp.status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
