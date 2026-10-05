"""Render every finished page and check basic manuscript build properties."""
from pathlib import Path
import sys, json, re
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'build/python_deps'))
import fitz
from PIL import Image, ImageOps, ImageDraw
out=P/'build/preview';out.mkdir(exist_ok=True)
info={}
for stem in ('main','supplement','title_page'):
    doc=fitz.open(P/f'{stem}.pdf')
    alltext='\n'.join(page.get_text() for page in doc)
    (P/'build'/f'{stem}_extracted.txt').write_text(alltext,encoding='utf8')
    thumbs=[]
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False)
        pix.save(out/f'{stem}_{i+1}.png')
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        im.thumbnail((370,540))
        thumb=Image.new('RGB',(390,570),'#dce1e6');thumb.paste(im,((390-im.width)//2,25))
        ImageDraw.Draw(thumb).text((12,7),f'{stem} / {i+1}',fill='black')
        thumbs.append(thumb)
    cols=3;rows=(len(thumbs)+cols-1)//cols
    canvas=Image.new('RGB',(390*cols,570*rows),'#c5ccd2')
    for i,im in enumerate(thumbs):canvas.paste(im,((i%cols)*390,(i//cols)*570))
    canvas.save(out/f'{stem}_contact.png')
    log=(P/'build'/f'{stem}.log').read_text(errors='replace')
    problems=[line for line in log.splitlines() if 'Overfull' in line or 'undefined' in line or 'Error:' in line]
    info[stem]={'pages':len(doc),'words_extracted':len(alltext.split()),'layout_errors':problems,
                'page_dimensions_pt':list(doc[0].rect)}
    if stem=='main':assert len(doc)<=7
    assert not problems,(stem,problems)
tex=(P/'main.tex').read_text()
abstract=tex.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
info['abstract_words']=len(abstract.split())
assert info['abstract_words']<=200
highlights=(P/'highlights.txt').read_text().splitlines()
info['highlight_lengths']=[len(x) for x in highlights]
assert 3<=len(highlights)<=5 and all(len(x)<=85 for x in highlights)
(P/'build/quality_check.json').write_text(json.dumps(info,indent=2),encoding='utf8')
print(json.dumps(info,indent=2))
