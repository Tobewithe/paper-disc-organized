import argparse, csv, json, os
from pathlib import Path
import joblib
import numpy as np
from fit_risk_calibration import features
from portable_risk import PortableRisk


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((Path(a.bank)/'candidate_records.csv').open(encoding='utf-8',newline='')))
    summaries={}
    fields={'value':'value','feature':'feature_idx','threshold':'num_threshold','missing_left':'missing_go_to_left','left':'left','right':'right','leaf':'is_leaf'}
    for mode in ('area','shape','response'):
        model=joblib.load(Path(a.models)/(mode+'.joblib'))
        trees=[stage[0].nodes for stage in model._predictors]
        assert all(len(stage)==1 for stage in model._predictors)
        assert all(not t['is_categorical'].any() for t in trees)
        size=max(len(t) for t in trees)
        nodes={key:[np.pad(t[field],(0,size-len(t))).tolist() for t in trees] for key,field in fields.items()}
        spec=dict(format='numeric_hgb_v1',features=int(model.n_features_in_),baseline=float(model._baseline_prediction[0,0]),
                  max_depth=int(max(t['depth'].max() for t in trees)),nodes=nodes)
        path=out/(mode+'.json');path.write_text(json.dumps(spec,allow_nan=False),encoding='utf-8')
        portable=PortableRisk(path);x=features(rows,mode);reference=model.predict(x)
        actual=np.concatenate([portable.predict(x[i:i+4096]) for i in range(0,len(x),4096)])
        error=float(np.max(np.abs(actual-reference)));different=int(np.sum((actual>0)!=(reference>0)))
        assert error<1e-12 and different==0,(mode,error,different)
        summaries[mode]=dict(candidates=len(x),max_abs_error=error,decision_mismatches=different,trees=len(trees))
    temp=out/'SUMMARY.json.tmp';temp.write_text(json.dumps(summaries,indent=2),encoding='utf-8');os.replace(temp,out/'SUMMARY.json')
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
