# Cover letter draft（PRL 投稿附函草稿，待作者信息）

Dear Editors,

We submit our manuscript **"Recoverable but unrealized: the mask-quality gap of prototype-based
instance segmentation"** for consideration by *Pattern Recognition Letters*.

**Problem and claim.** Prototype-based instance segmenters decode masks from a shared prototype
bank and per-instance coefficients. A candidate can localize its object and still fail a strict
mask criterion, and aggregate accuracy does not reveal where the quality is lost. We ask two
questions: are such mask failures recoverable inside the frozen representation, and if so, why
does the native model not recover them?

**What we show.** On full COCO val2017 with a frozen official checkpoint we (i) build a three-layer
accounting of where mask quality is lost (raw candidate geometry, class-argmax conditioning,
output retention); (ii) show that **75.0% of strict argmax-conditioned mask failures admit a
coefficient correction that reaches Mask75 with the prototype bank frozen**, under a documented
finite-oracle definition re-derived and independently verified in this study (66.0% under the
original panel's partially recorded decode conventions — both reported); and (iii) exclude seven
candidate explanations with pre-registered controlled comparisons: candidate-position selection,
supervision quantity, ownership readout, ownership through a fixed solver, shared final-layer
refitting, post-hoc global correction, oracle-target distillation into the native branch, and
constrained functional projection. The evidence localizes the bottleneck at how coefficients are
realized rather than at whether the information exists.

**Why PRL.** The contribution is a focused negative-space result with a quantitative accounting
method: it is short, mechanism-level, and gives the next method attempt a specific target and a
set of controlled comparisons it must beat. All readouts are diagnostics on one frozen checkpoint;
no accuracy claim is made, and single-model scope is stated in the limitations.

**Reproducibility.** Every number traces to a retained run record; per experiment we retain the
protocol frozen before execution, per-candidate and per-GT records, independent verification runs,
source snapshots with hashes, and transfer manifests.

We believe the manuscript fits the readership of PRL and is not under consideration elsewhere.

Sincerely,
[Author names, affiliations, and corresponding author — to be supplied]
