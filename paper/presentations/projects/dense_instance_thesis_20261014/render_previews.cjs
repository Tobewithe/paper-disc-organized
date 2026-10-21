// Renders authored slides for visual QA; does not author or edit slide content.
const fs = require('node:fs');
const path = require('node:path');
const sharp = require('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const project = __dirname;
async function main() {
 const out = path.join(project, 'qa', 'svg');
 fs.mkdirSync(out, {recursive:true});
 const names = fs.readdirSync(path.join(project, 'svg_output')).filter(n=>n.endsWith('.svg')).sort();
 for(const n of names) await sharp(path.join(project,'svg_output',n),{density:144}).resize(1280,720).png().toFile(path.join(out,n.replace('.svg','.png')));
 for(let start=0;start<names.length;start+=4){
   const tiles=[];
   for(let j=0;j<4 && start+j<names.length;j++){
     tiles.push({input:await sharp(path.join(out,names[start+j].replace('.svg','.png'))).resize(640,360).toBuffer(),left:(j%2)*640,top:Math.floor(j/2)*360});
   }
   await sharp({create:{width:1280,height:720,channels:3,background:'#D4DEE7'}}).composite(tiles).png().toFile(path.join(project,'qa',`montage_${start+1}-${Math.min(start+4,names.length)}.png`));
 }
 console.log(JSON.stringify({rendered:names.length,output:out}));
}
main().catch(e=>{console.error(e);process.exit(1)});
