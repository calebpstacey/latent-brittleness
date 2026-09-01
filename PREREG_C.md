# Preregistration — Cohort C (generalisation)

**Written before Cohort C was trained.** Regime change: MNIST + MLP → **CIFAR-10 + CNN**.

Cohorts A and B remain frozen. Nothing from them is re-fit here.

## Purpose

Cohort B established the surviving claims within one regime. A reviewer can reasonably
say those are properties of small MLPs on MNIST. Cohort C tests whether the **qualitative
theory** transfers to a different dataset, a different architecture family, and a
different notion of "unit" (channels rather than neurons).

Numerical values are **not** expected to replicate. The five questions below are
qualitative and were fixed in advance.

## Design

- **Data**: CIFAR-10, 8000 train / 2000 test, 3×32×32.
- **Architecture**: small CNN. Blocked cells over
  `channels ∈ {16, 32, 64} × conv_blocks ∈ {2, 3} × dropout ∈ {0.1, 0.3}` = 12 cells,
  ≥4 seeds per cell.
- **Baseline matching**: snapshot at the epoch closest to target P₀; keep only models
  inside a narrow band. Band set from a pilot, not tuned after seeing degradation data.
- **Stressors**: same 4 families. Capacity damage redefined for CNNs — **channel**
  ablation replaces neuron ablation — in both fractional and fixed-absolute form.
- **ε grid**: 10 points, k/9.
- Full curves retained.

## Pre-committed questions (qualitative pass/fail)

**Q1 — Seed-level heterogeneity.** Do models sharing architecture, recipe and baseline
still differ in degradation? *Pass:* within-cell share of (architecture + model)
variance ≥ 15%, permutation p < 0.01.

**Q2 — Within-family coherence.** *Pass:* mean within-family ρ exceeds mean
between-family ρ by ≥ 0.15, and this survives residualisation on P₀.

**Q3 — Parameterisation reversal.** Do fractional and fixed-absolute capacity damage
give **opposite-signed** relationships with model width/channels? *Pass:* signs oppose
and both |ρ| > 0.3.

**Q4 — Curve dimensionality.** *Pass:* first two functional-PCA modes explain ≥ 90% of
curve variance, mode 1 ≈ magnitude, mode 2 ≈ early/late tilt.

**Q5 — Curvature is weak and coordinate-sensitive.** *Pass:* curvature explains the
least curve variance among the four features **and** shows the lowest rank stability
under ε → ε².

## Interpretation rule

- **5/5 or 4/5**: the theory generalises beyond MNIST/MLPs.
- **2–3/5**: partial generalisation; the failing items become stated boundary conditions.
- **0–1/5**: the Cohort B result is regime-specific.

Any failure is reported. No question may be dropped or restated after seeing results.

## Explicit non-goals

- Not testing whether Cohort A/B factor loadings reproduce. That hypothesis was already
  killed under preregistration in Cohort B and is not revived here.
- Not testing numerical equality of any effect size across regimes.
