# S083 frozen decoder grid and threshold probe

The official candidate, box, coefficient, and prototype tensors are frozen. Only post-logit sampling grid and threshold change.

- Device: `cuda`; images: 1483; decoded targets: 9626; skipped: 0
- Official `input640_t0` replay max absolute IoU difference vs S078: `0.000e+00`
- `native160` and post-interpolation 320/1280 arms are decoder diagnostics; they do not create new feature information.

See `SUMMARY.csv` for image-bootstrap means and 95% intervals, and `per_target_arm.csv` for target-level values.

Interpretation guardrail: a recoverable gap under a decoder arm localizes loss to sampling/threshold/cropping, but does not establish a trainable method or explain standard AP by itself.
