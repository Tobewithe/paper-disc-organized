# Position-conditioned shared affine screen

Use fixed geometry candidates and the existing native GT-box objective. Augment each normalized 64-d feature with six deterministic raw-grid features `[1,x,y,x²,y²,xy]` per pyramid level; solve one shared affine 70→32 map per level on fit, then decode geometry val with the original boxes and prototypes. Compare against the already frozen G-BIAS result. This is one fixed design, no term/weight sweep. Continue only if image-macro IoU exceeds G-BIAS by a predeclared practical margin of 0.2 pp; otherwise stop simple coordinate readout.
