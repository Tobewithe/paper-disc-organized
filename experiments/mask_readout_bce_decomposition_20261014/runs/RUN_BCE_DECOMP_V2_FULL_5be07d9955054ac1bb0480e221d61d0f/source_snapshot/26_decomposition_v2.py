"""Separate pixel attribution from the original FP32 reduction roundoff."""
import math
import decomposition_statistics as v1
from decomposition_statistics import adjudicate


def partition(zero_logits, B_logits, targets, roi, zero_pixel_bce, B_pixel_bce):
    result=v1.partition(zero_logits,B_logits,targets,roi,zero_pixel_bce,B_pixel_bce)
    area=roi.sum((1,2))
    direct0=(zero_pixel_bce.double()*roi).sum((1,2))/area
    direct1=(B_pixel_bce.double()*roi).sum((1,2))/area
    for row,a,b in zip(result,direct0.cpu().tolist(),direct1.cpu().tolist()):
        error=max(abs(row['totals']['zero_loss']-a),abs(row['totals']['B_loss']-b),
                  abs(row['totals']['delta_loss']-(b-a)))
        if error>1e-12:
            raise ValueError('direct FP64 sum differs from partition contributions')
        row.update(direct_zero_BCE_FP64=a,direct_B_BCE_FP64=b,FP64_partition_abs_error=error)
    return result


def closure_receipt(decomposition, original_zero_FP32, original_B_FP32):
    a,b=decomposition['direct_zero_BCE_FP64'],decomposition['direct_B_BCE_FP64']
    r0,r1=original_zero_FP32-a,original_B_FP32-b
    rounding=r1-r0
    direct_gap=decomposition['totals']['delta_loss']-(original_B_FP32-original_zero_FP32)
    error=abs(decomposition['totals']['delta_loss']+rounding-(original_B_FP32-original_zero_FP32))
    if error>1e-12:
        raise ValueError('pixel contribution plus explicit original reduction roundoff not closed')
    return {'zero_FP32_reduction_roundoff':r0,'B_FP32_reduction_roundoff':r1,'delta_FP32_reduction_roundoff':rounding,
            'raw_pixel_delta_minus_FP32_delta':direct_gap,'BCE_delta_closure_abs_error':error}


def summarize(rows):
    result=v1.summarize(rows)
    result.update(accounting_revision=2,
        max_raw_FP32_FP64_delta_gap=max(abs(r['raw_pixel_delta_minus_FP32_delta']) for r in rows),
        max_FP64_partition_abs_error=max(r['decomposition']['FP64_partition_abs_error'] for r in rows),
        mean_delta_FP32_reduction_roundoff=math.fsum(r['delta_FP32_reduction_roundoff'] for r in rows)/len(rows),
        mean_original_FP32_delta_BCE=math.fsum(r['delta_gt_bce'] for r in rows)/len(rows),
        closure_definition='FP64 pixel contributions plus separately measured original FP32 reduction roundoff')
    return result
