// Read-only rendering of the exported PowerPoint package for QA.
import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, PresentationFile } from 'file:///C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const base = 'C:/Dpan/codexproject/paper-disc-organized/paper/presentations/projects/dense_instance_thesis_20261014';
const input = path.join(base,'exports', (await fs.readdir(path.join(base,'exports'))).filter(n=>n.endsWith('.pptx')).sort().at(-1));
const p = await PresentationFile.importPptx(await FileBlob.load(input));
const out = path.join(base,'qa',path.basename(input,'.pptx'));
await fs.mkdir(out,{recursive:true});
for(let i=0;i<p.slides.items.length;i++) {
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 console.log(`Rendered ${i+1}/${p.slides.items.length}`);
}
console.log(out);
