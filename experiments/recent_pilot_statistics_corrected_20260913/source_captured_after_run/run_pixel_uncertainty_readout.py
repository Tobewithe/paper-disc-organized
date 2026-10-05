"""S066 runner generated from the audited S032 pipeline."""
from pathlib import Path
base=Path(__file__).with_name('train_readout_label_controls.py').read_text(encoding='utf-8')
base=base.replace("MODES=['native_overlap','native_independent','raw_coco']", "MODES=['raw_coco','pixel_uncertainty']")
base=base.replace("from rich_pixel_readout import GlobalHead,instance_features,loss_value", "from rich_pixel_readout import GlobalHead,instance_features,loss_value\nfrom train_pixel_uncertainty_readout import pixel_uncertainty_loss")
base=base.replace("data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]}", "data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]}\n    data['pixel_uncertainty']=data['raw_coco']")
base=base.replace("prior_norm=torch.load(a.prior/'normalizer.pt',map_location='cuda',weights_only=True)['x']", "prior_norm=torch.load(a.prior/'normalizer.pt',map_location='cuda',weights_only=True)")
base=base.replace("method='Existing global73D(h+level+box geometry)->128->128->128(mean4x32) coefficient residual. Same S026 architecture.',", "method='S066 same frozen global73D coefficient head; raw COCO control versus pixelwise frozen-logit uncertainty weighting.',")
base=base.replace("loss=loss_value(z,data[mode][ix],data['factor'][ix])", "z0=(data['p'][ix]*data['c'][ix][:,None]).sum(-1)\n                    loss=(pixel_uncertainty_loss(z,data['pixel_uncertainty'][ix],z0,data['factor'][ix]) if mode=='pixel_uncertainty' else loss_value(z,data[mode][ix],data['factor'][ix]))")
start=base.find("            if mode=='native_overlap':\n                old=")
if start>=0:
 end=base.find("            trained.append",start); base=base[:start]+base[end:]
generated=Path(__file__).with_name('_generated_pixel_uncertainty_runner.py'); generated.write_text(base,encoding='utf-8'); exec(compile(base,str(generated),'exec'),globals(),globals())
