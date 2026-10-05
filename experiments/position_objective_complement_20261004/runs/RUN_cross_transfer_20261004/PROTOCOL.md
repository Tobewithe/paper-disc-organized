# Cross-position transfer diagnostic

Apply the already-frozen G-BIAS learned on geometry candidates to the original official TAL raw positions. Keep each official candidate's own coefficient, prototype, predicted box and original-image decode; add only the per-level 32-vector bias. This is a fixed cross-transfer replay, not a new training method. Use the same 196 val images and 5000 image bootstrap as Stage 1. A positive result would show the bias transfers without donor replacement; a non-positive result would show the geometry benefit is tied to the geometry-position distribution.
