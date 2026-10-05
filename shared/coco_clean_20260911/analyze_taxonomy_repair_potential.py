"""Cross the layered failure taxonomy with available S056 oracle repairs.

Only the 5,575 fixed box-good/mask-bad slots have S056 repair columns. The
script reports rescue rates in that conditional population and never fills
unobserved failure classes with zeros.
"""
from pathlib import Path
import argparse, json
import pandas as pd


ORACLES = {
    "same_neighbor": "oracle_same_remove_iou",
    "background": "oracle_background_remove_iou",
    "own_complete": "oracle_own_complete_iou",
    "all_fp": "oracle_all_fp_remove_iou",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--repairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    tax = pd.read_csv(a.taxonomy)
    rep = pd.read_csv(a.repairs)
    if tax.annotation_id.nunique() != len(tax): raise RuntimeError("taxonomy IDs not unique")
    keep = ["annotation_id", *ORACLES.values(), "original_iou"]
    d = tax.merge(rep[keep], on="annotation_id", how="inner", validate="one_to_one")
    if len(d) != len(rep): raise RuntimeError("repair join is not one-to-one")
    rows=[]
    for scope,g in d.groupby("failure_scope", dropna=False):
        for name,col in ORACLES.items():
            rows.append(dict(scope=scope,n=len(g),oracle=name,
                             rescued75=int((g[col]>=.75).sum()),
                             rescue_rate=float((g[col]>=.75).mean()),
                             mean_delta_iou=float((g[col]-g.original_iou).mean())))
    by_density=[]
    g=d[d.failure_scope=="box_good_support_sufficient_mask_bad"]
    for density,z in g.groupby("density_q", dropna=False):
        row=dict(scope="box_good_support_sufficient_mask_bad",density=density,n=len(z))
        for name,col in ORACLES.items():
            row[name+"_rescued75"]=int((z[col]>=.75).sum())
            row[name+"_rescue_rate"]=float((z[col]>=.75).mean())
        by_density.append(row)
    result={
        "protocol": {
            "taxonomy": str(a.taxonomy), "repairs": str(a.repairs),
            "join": "annotation_id one-to-one",
            "rescue": "oracle IoU >= 0.75; conditional on S056 fixed box-good/mask-bad slots",
            "unobserved": "no_final_slot and box-limited classes have no S056 repair estimate",
            "interpretation": "oracle rescue is opportunity sizing, not deployable AP or additive decomposition",
        },
        "joined_slots": int(len(d)), "taxonomy_gt": int(len(tax)),
        "scope": rows, "sufficient_mask_failure_by_density": by_density,
    }
    (out/"SUMMARY.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    pd.DataFrame(rows).to_csv(out/"scope_oracle_rescue.csv",index=False)
    pd.DataFrame(by_density).to_csv(out/"sufficient_by_density.csv",index=False)
    print(json.dumps({"status":"COMPLETE","taxonomy_gt":len(tax),"joined_slots":len(d),"out":str(out)},ensure_ascii=False))


if __name__ == "__main__": main()
