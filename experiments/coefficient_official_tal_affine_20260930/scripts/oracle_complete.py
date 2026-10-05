"""Complete the per-candidate C reference; never re-solve shared parameters."""
import argparse, gc, json
from pathlib import Path
import torch
from official_tal_solve_fast import load_rows, Bank, solve_independent

def main():
    ap=argparse.ArgumentParser()
    for k in ('cache','shared','out'): ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    shared=torch.load(a.shared,map_location='cpu',weights_only=False)
    summaries={}
    for split in ('fit','dev','val'):
        rows=load_rows(a.cache,split)
        bank=Bank(rows,shared['stats'],torch.device('cuda'),9.83241)
        summary,delta,identities=solve_independent(bank,120,16)
        norms=[r['gradient_norm'] for r in identities]
        summary.update(converged_gradient_le_1e_4=sum(x<=1e-4 for x in norms),iteration_limit=sum(r['exit_reason']=='iteration_limit' for r in identities),all_identities_retained=True)
        summaries[split]=summary
        torch.save({'delta':delta,'identities':identities},a.out/f'ORACLE_{split}.pt')
        (a.out/'ORACLE_SUMMARY.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
        print(json.dumps({'split':split,**summary}),flush=True)
        del bank,rows; gc.collect(); torch.cuda.empty_cache()

if __name__=='__main__': main()
