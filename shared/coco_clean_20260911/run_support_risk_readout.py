"""Generate and execute an S061 runner from the audited S032 pipeline."""
from pathlib import Path

base = Path(__file__).with_name('train_readout_label_controls.py').read_text(encoding='utf-8')
base = base.replace("MODES=['native_overlap','native_independent','raw_coco']", "MODES=['raw_coco','support_risk']")
base = base.replace(
    "from rich_pixel_readout import GlobalHead,instance_features,loss_value",
    "from rich_pixel_readout import GlobalHead,instance_features,loss_value\nfrom train_support_risk_readout import support_risk_loss")
base = base.replace(
    "method='Existing global73D(h+level+box geometry)->128->128->128(mean4x32) coefficient residual. Same S026 architecture.',",
    "method='S061 same frozen global73D residual head; raw COCO control versus support-preserving background-risk loss.',")
base = base.replace(
    "data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]}",
    "data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]}\n    data['support_risk'] = data['raw_coco']")
base = base.replace(
    "prior_norm=torch.load(a.prior/'normalizer.pt',map_location='cuda',weights_only=True)['x']",
    "prior_norm=torch.load(a.prior/'normalizer.pt',map_location='cuda',weights_only=True)")
base = base.replace(
    "loss=loss_value(z,data[mode][ix],data['factor'][ix])",
    "z0=(data['p'][ix]*data['c'][ix][:,None]).sum(-1)\n                    loss=(support_risk_loss(z,data['support_risk'][ix],z0,data['factor'][ix]) if mode=='support_risk' else loss_value(z,data[mode][ix],data['factor'][ix]))")
base = base.replace("if mode=='native_overlap':", "if mode=='raw_coco':")
start = base.find("            if mode=='raw_coco':\n                old=")
if start >= 0:
    end = base.find("            trained.append", start)
    base = base[:start] + base[end:]
generated = Path(__file__).with_name('_generated_support_risk_runner.py')
generated.write_text(base, encoding='utf-8')
exec(compile(base, str(generated), 'exec'), globals(), globals())
