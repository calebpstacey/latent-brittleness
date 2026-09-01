# Preregistration — Cohort E (clean two-regime design)

**Written before any Cohort E data were collected. Supersedes `PREREG_C.md`.**

Timestamped by deposit to Zenodo prior to running. Concept DOI 10.5281/zenodo.22120112.

---

## Why this supersedes PREREG_C

`PREREG_C.md` fixed a baseline band from a single-core pilot run at a different
configuration. When executed on the intended configuration, that band caught **9 of 72
models (12.5%)** — far below the ≥4 seeds/cell across ≥3 cells that PREREG_C itself
requires for Q1. Q1 and Q2 were therefore **not evaluable**, not failed: with most cells
holding a single model, within-cell centring is degenerate by construction.

**The Cohort C run is reported in full, not discarded.** Its output, log and data are
deposited alongside this document. It establishes the CIFAR operating range, which is
what a pilot is for.

**Full disclosure of what we have seen.** We have seen the Cohort C P₀ distribution
(n=72, ceiling 0.486, median 0.310) and its degradation curves. Because PREREG_C
specified that its band was *"not tuned after seeing degradation data"*, we do not
re-tune that band and re-run under PREREG_C. This is a new preregistration with the
design changed for stated reasons, and we note that anyone assessing Cohort E should
weigh it accordingly: the *design* is informed by seen data, the *hypotheses* are not.

## What changes, and why

| | Cohort B / C | Cohort E |
|---|---|---|
| Training budget | epoch scan for best match, epochs 3–30 | **fixed**, identical for every model |
| Baseline matching | select models inside a band | **none — keep all models**, residualise P₀ |
| Data split | one eval set for matching *and* stress | **three-way**: train / match / stress |
| MNIST training set | 5,000 | 30,000 |
| CIFAR training set | 8,000 | 30,000 |
| Expected yield | 12–50% | **100%** |

Removing the band fixes four problems at once:

1. **Yield.** Every trained model enters the analysis.
2. **Test-set reuse.** Nothing selects on the stress set, because nothing selects at all.
3. **Epoch confound.** Fixed budget means training duration is no longer a variable.
4. **Regime artificiality.** More data and a full budget give converged, competent models
   rather than deliberately early snapshots.

Baseline equivalence is now enforced **statistically** (residualise P₀, block on
architecture) rather than by selection. This is the stronger form of the claim: same
architecture, literally identical recipe, P₀ controlled as a covariate.

## Design

Two arms, one protocol.

- **Arm 1 — MNIST + MLP.** Cells: width {64,128,256} × depth {1,2} × dropout {0.1,0.3}.
- **Arm 2 — CIFAR-10 + CNN.** Cells: channels {16,32,64} × blocks {2,3} × dropout {0.1,0.3}.

12 cells × 6 seeds = 72 models per arm, **all retained**.

Splits: 30,000 train / 2,000 match / 2,000 stress, disjoint. The match set is used only
to report P₀. The stress set is touched only during the stress sweep.

Thirteen stressors in four families, identical to Cohort C. Perturbation RNG seeded
independently of the training seed. Ten ε points. Capacity damage in both fractional and
fixed-absolute (per-layer cap) parameterisation.

## Pre-committed questions

All tests residualise P₀ and are computed within architecture cell.

**Q1 — Seed-level heterogeneity.** Permutation test of H₀ = *no persistent cross-stressor
model effect within architecture cell*, permuting model labels independently per stressor
within cell. *Pass:* p < 0.01 in **both** arms.

**Q2 — Within-family coherence.** *Pass:* mean within-family ρ exceeds between-family by
≥ 0.15 after P₀ residualisation, in **both** arms.

**Q3 — Parameterisation reversal.** *Pass:* fractional and fixed-absolute capacity damage
give opposite-signed ρ with width/channels, both |ρ| > 0.3, in **at least one** arm, with
the direction consistent across arms where both are significant.

**Q4 — Curve dimensionality.** *Pass:* first two functional-PCA modes ≥ 90%, mode 1 a
monotone ramp, in both arms. **Reported against a monotone null** — we do not claim
two-dimensionality itself, only the size of the second mode relative to null.

**Q5 — Curvature coordinate-sensitivity.** *Pass:* curvature has the lowest rank-stability
under ε → ε² in both arms, **and** its margin over the next-lowest exceeds 0.02.
(Cohort C separated curvature from area by 0.007 at n=9, which is not a result.)

## Decision rule

- **4–5/5** — the theory generalises across regimes
- **2–3/5** — partial; failing items become stated boundary conditions
- **0–1/5** — the effects are specific to the exploratory regime

Additionally, **any question is reported as NOT EVALUABLE rather than FAIL** if the arm
delivers fewer than 4 models per cell across fewer than 3 cells. A power failure is not
a negative result and will not be reported as one.

## Confirmatory status

Q1 is confirmatory in Cohort E. In the deposited manuscript it is a corrected post-hoc
analysis, because the original null was invalid and its replacement was built afterwards.
Preregistering it here, before data collection, converts it — provided the analysis runs
exactly as specified above.

Q3 was a pre-committed prediction in PREREG.md and remains so.

Q4 and Q5 are confirmatory replications of exploratory Cohort B findings.

## Pre-committed contingency

If Cohort E gives a **weaker** persistence effect than Cohort B, that is the result and
will be reported as such. Cohort B's effect was measured under a design with test-set
reuse and an epoch confound; a smaller clean effect is the more trustworthy number, and
the difference between them is itself informative. We commit to this now, before looking.

## What would falsify

Q1 failing in both arms under a clean design would mean the seed-level heterogeneity
reported in the deposited manuscript is an artefact of checkpoint selection on the stress
set. That is a live possibility and we would report it.
