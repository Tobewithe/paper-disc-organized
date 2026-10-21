"""Original full-population statistics with explicit numerical-failure status."""
from margin_stats import analyze as original_analyze


def analyze(rows, cfg, scope, output_indices=None):
    result = original_analyze(rows, cfg, scope, output_indices)
    tol = cfg['numerics']['rms_relative_tolerance']
    records = [p for row in rows for p in row['pressure']]
    diag = {}
    for center in ['C', 'W']:
        linear = [abs(p[center+'_actual_rms']/r['pressure_amplitude']-1) for r in rows for p in r['pressure']]
        physical = [abs(p[center+'_decoded_difference_rms']/r['pressure_amplitude']-1) for r in rows for p in r['pressure']]
        diag[center] = {'directions':len(records), 'original_RMS_failed':sum(x>tol for x in linear),
            'decoded_difference_failed':sum(x>tol for x in physical),
            'original_max_relative_error':max(linear), 'decoded_max_relative_error':max(physical)}
    valid = all(row['pressure_precision_valid'] for row in rows)
    result.update(pressure_diagnostics=diag,
        pressure_invalid_pairs=sum(not r['pressure_precision_valid'] for r in rows),
        original_pressure_invalid_pairs=sum(not r['original_pressure_precision_valid'] for r in rows),
        pressure_precision_valid=valid, all_pairs_retained=True,
        analysis_permission='descriptive_actual_implementation_when_numerically_invalid')
    result['numerical_valid'] &= valid
    if scope == 'full' and not result['numerical_valid']:
        result['decision'] = 'INCONCLUSIVE_NUMERICAL'
        result['training_readiness'] = 'not_recommended_at_scope'
    return result
