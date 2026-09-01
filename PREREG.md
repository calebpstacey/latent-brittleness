# Preregistration — Cohort B validation

**Written before Cohort B was trained. Cohort A frozen at `FROZEN_A.json`, sha256[:16] = `0d2a2f146757e398`.**

---

## Hypothesis under test

Baseline-equivalent systems have structured, multidimensional failure behaviour, and that
structure is (a) reproducible in an independent cohort and (b) not reducible to architecture.

## What is frozen and may not be re-fit

- Q95 anchors per (stressor, curve-feature), from Cohort A, 4-point ε grid
- Factor loadings PC1/PC2/PC3 on the 11 original stressors, sign-fixed to
  PC1+ on `noise`, PC2+ on `weight_drop`, PC3+ on `rotate`
- B-score column means and SDs used for z-scoring

Cohort B factor scores are computed as `Z_B @ loadings_A`. No re-estimation.

## Design changes from Cohort A

| Item | Cohort A | Cohort B |
|---|---|---|
| Baseline band | 0.821–0.896 (7.5pp) | 0.855–0.885 (3.0pp), snapshot at best epoch |
| Architecture | randomised, then filtered | **blocked**: 12 cells, ≥4 seeds/cell |
| ε points | 4 | 10 (k/9, k=0..9) — contains the A subgrid exactly |
| Capacity damage | fractional only | fractional **and** fixed-absolute |
| ε reparameterisation | not tested | linear and quadratic, computed from same curves |
| Curve storage | 4 features | full 10-point curve retained |

Architecture cells: width ∈ {64,128,256} × depth ∈ {1,2} × dropout ∈ {0.1,0.3}.

## Decision model

$$B_{i,s} = \mu + A_{\text{arch}} + G_{\text{model|arch}} + T_s + (A\times T) + (G\times T) + \varepsilon$$

Blocking lets `A` (architecture) and `G` (model within architecture) be estimated
separately rather than partialled out post hoc.

## Success criteria — all five must hold

1. **Loading coherence.** Applying frozen Cohort-A loadings to Cohort B, the mean
   loading of input-family stressors on PC1 and of capacity-family stressors on PC2
   each exceed the mean cross-family loading, with 90% bootstrap CI excluding zero.
2. **Stability.** Bootstrap Tucker congruence between Cohort-B-refit PC1/PC2 and the
   frozen Cohort-A PC1/PC2 ≥ 0.85 (component-matched, not rank-matched).
3. **Family separation.** Within-family mean ρ exceeds between-family mean ρ by ≥ 0.15.
4. **Baseline independence.** Criteria 1 and 3 hold after residualising every B-score
   against P₀.
5. **Parameterisation robustness.** Criteria 1 and 3 hold under ε → ε² recomputation
   of the curve features.

## Kill condition

If criterion 1 or 2 fails, the factor result is cohort-specific and goes in the drawer.
Criteria 3–5 failing individually downgrade rather than kill: the result is reported as
parameterisation-dependent or baseline-entangled.

## Separately reported, not part of the pass/fail

- **Within-cell variance.** Do same-architecture, same-P₀ models still differ in
  degradation profile? This is the core latent-brittleness claim stripped of architecture.
  Reported as var(G|arch) / [var(G|arch) + var(residual)] with a permutation test.
- **Functional PCA on the retained 10-point curves**, asking whether the empirical
  modes of degradation resemble (S,C,A,L) or want a different basis.
- **Per-feature coordinate sensitivity** under ε → ε², reported feature by feature.
  Prediction from Cohort A: area stable, curvature unstable.

## Pre-committed predictions

- PC1 (input/perceptual) and PC2 (capacity) reproduce; PC3 does not.
- Curvature shows the largest rank instability under ε → ε²; area the least.
- Fractional and absolute capacity damage load with **opposite** sign on width.
- Within-cell model variance is non-zero but smaller than between-cell architecture variance.
