"""Acquire public original PDFs without requiring or inventing a contact email."""
import concurrent.futures, hashlib, json, pathlib, re, shutil, time, urllib.request
from pypdf import PdfReader

ROOT = pathlib.Path(__file__).resolve().parent

def acquire(record, folder):
    title = record['title']
    rank = record.get('rank', record.get('id', 0))
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:85]
    name = f'{rank:02d}_{slug}' if isinstance(rank, int) else f'{rank}_{slug}'
    result = {'rank': rank, 'title': title, 'attempts': []}
    urls = []
    if record.get('pdf_url'): urls.append(record['pdf_url'])
    if record.get('arxiv_id'): urls.append('https://arxiv.org/pdf/' + record['arxiv_id'])
    target = folder / (name + '.pdf')
    # Reuse already acquired originals when ranks change after corpus merging.
    if not target.exists():
        for manifest_path in ROOT.glob('*/manifest.json'):
            try:
                prior = json.loads(manifest_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            for item in prior:
                if item.get('title', '').casefold() == title.casefold() and item.get('status') == 'acquired':
                    source = ROOT / item['file']
                    if source.exists():
                        shutil.copy2(source, target)
                        result['reused_from'] = item['file']
                        if not urls and item.get('url'): urls.append(item['url'])
                        break
            if target.exists(): break
    for url in dict.fromkeys(urls):
        try:
            if not target.exists():
                req = urllib.request.Request(url, headers={'User-Agent': 'ResearchLiteratureReader/1.0'})
                with urllib.request.urlopen(req, timeout=25) as response:
                    data = response.read(60_000_001)
                if len(data) > 60_000_000: raise ValueError('PDF over 60 MB limit')
                if not data.startswith(b'%PDF-'): raise ValueError('not a PDF')
                target.write_bytes(data)
            data = target.read_bytes()
            reader = PdfReader(target)
            texts = [f'\n\nPAGE {i+1}\n' + (page.extract_text() or '') for i, page in enumerate(reader.pages)]
            text_path = folder / (name + '.txt')
            text_path.write_text(''.join(texts), encoding='utf-8')
            result.update(status='acquired', url=url, file=str(target.relative_to(ROOT)), text_file=str(text_path.relative_to(ROOT)), bytes=len(data), pages=len(reader.pages), sha256=hashlib.sha256(data).hexdigest())
            break
        except Exception as exc:
            result['attempts'].append({'url': url, 'error': str(exc)})
    else: result['status'] = 'not_acquired'
    print(result['status'], rank, title, flush=True)
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('--out', default='pdfs')
    parser.add_argument('--top', type=int, default=50)
    args = parser.parse_args()
    records = json.loads(pathlib.Path(args.corpus).read_text(encoding='utf-8'))[:args.top]
    folder = ROOT / args.out
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(acquire, record, folder) for record in records]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            (folder / 'manifest.json').write_text(json.dumps(sorted(results, key=lambda r: str(r['rank'])), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'attempted':len(results),'acquired':sum(r['status']=='acquired' for r in results)}), flush=True)
