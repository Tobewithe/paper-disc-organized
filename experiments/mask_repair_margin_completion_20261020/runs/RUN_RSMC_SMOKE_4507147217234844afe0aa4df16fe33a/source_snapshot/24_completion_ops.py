"""Frozen V3 normalization; retain the final result and precision failure."""
import numpy as np

def matched_coefficients(decoder,center,vectors,amplitude,tolerance):
    norm=decoder.rms(vectors)
    if not np.isfinite(norm).all() or np.any(norm<=1e-10):raise ValueError('zero/nonfinite pressure direction')
    deltas=np.asarray(vectors,dtype=np.float64)*(amplitude/norm)[:,None]
    scales=np.ones(len(deltas));history=[]
    for iteration in range(9):
        result=(np.asarray(center,dtype=np.float64)[None]+deltas*scales[:,None]).astype(np.float32)
        actual=decoder.rms(result.astype(np.float64)-np.asarray(center,dtype=np.float64))
        if not np.isfinite(actual).all() or np.any(actual<=1e-10):raise ValueError('rounded pressure vanished')
        error=np.abs(actual/amplitude-1)
        history.append({'corrections':iteration,'max_relative_error':float(error.max()),'bad_directions':int((error>tolerance).sum())})
        if np.all(error<=tolerance):break
        if iteration<8:
            bad=error>tolerance;scales[bad]*=amplitude/actual[bad]
    # Preserve final V3 coefficients; failure remains recorded, never relabelled valid.
    precision_valid = bool(np.all(np.abs(actual/amplitude-1)<=tolerance))
    return result,actual,norm,{'scales':scales.tolist(),'history':history,'precision_valid':precision_valid}
