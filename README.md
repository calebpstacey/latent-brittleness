# Latent Brittleness

Preregistered study of failure heterogeneity and measurement dependence in neural network
robustness.

**Data, code and all preregistrations:** [10.5281/zenodo.22120112](https://doi.org/10.5281/zenodo.22120112)
**Author:** Caleb Stacey, independent researcher — [ORCID 0009-0004-8175-1400](https://orcid.org/0009-0004-8175-1400)

---

## What this is

Two networks reach the same test accuracy. Are they equally reliable?

This work measures full **degradation curves** across thirteen stressors in four perturbation
families, for networks matched on architecture, training recipe and baseline accuracy, in two
regimes (MNIST+MLP and CIFAR-10+CNN, 72 models each).

### Findings

**Failure heterogeneity persists under matched specification, in both regimes.** Under a
preregistered design with a fixed training budget, no model selection, and a three-way disjoint
train/match/stress split, a permutation test rejects at p = 6.0e−4 (MNIST) and p = 2.0e−4
(CIFAR). Networks identical in architecture and training specification converge to parameter
states with measurably different failure behaviour.

**Robustness conclusions depend on the perturbation measure.** On a single global
magnitude-pruning trajectory — the same weights removed in the same order — the identical
physical states analysed under two coordinates give ρ(width, degradation) = **+0.63** aligned by
surviving weight count and **−0.30** aligned by surviving fraction. A pure coordinate change
reverses an architectural conclusion.

**Four self-corrections are reported in full.** An exploratory three-factor decomposition failed
preregistered replication; a 22-predictor pre-stress screen collapsed under architecture control;
a magnitude-plus-tilt curve representation was withdrawn against a monotone null; and a family
organisation of stressors reported at +0.464 under a design with test-set reuse falls to +0.173
under the clean design.

---

## Repository contents

```
code/
  cohortE.py         primary two-regime experiment (MNIST+MLP, CIFAR-10+CNN)
  cohortC_all.py     CIFAR-10 generalisation cohort, self-contained
  stress2.py         stressor suite and magnitude calibration
  cohortB.py         blocked validation cohort
  freeze.py          locks calibration anchors and factor loadings
  validateB.py       preregistered pass/fail criteria
  analyze.py         variance decomposition and factor analysis
  screen.py          pre-stress predictor screen, grouped cross-validation
  within.py          within-cell tests with multiple-comparison correction
  confound.py        fractional vs absolute damage parameterisation
  factors.py         parallel analysis and bootstrap factor stability
  tensor.py          per-metric family structure
  recheck.py         permutation tests, parallel analysis
  recheck2.py        component-matched bootstrap, partial correlations

preregistration/
  PREREG.md          Cohort B — written before Cohort B was trained
  PREREG_C.md        Cohort C — written before execution
  PREREG_E.md        Cohort E — deposited to Zenodo before the experiment ran

manuscript/
  LATENT_BRITTLENESS_Stacey.pdf   full manuscript
  PRUNING_SURVEY.md               code-level survey of eight pruning implementations
  fig1.png                        the coordinate reversal
```

## Running it

```bash
pip install -r requirements.txt
python code/cohortE.py mnist     # ~20 min on a GPU
python code/cohortE.py cifar     # ~35 min
python code/cohortE.py analyse   # preregistered verdict
```

Resumable — re-run the same command after any interruption. CUDA is auto-detected.

## Preregistration

All three preregistrations were deposited to Zenodo **before** their corresponding experiments
ran; the version DOIs establish the ordering. `PREREG_E.md` discloses openly which data had
been seen at the time it was written.

A failed pilot (Cohort C, 9 of 72 models retained) is deposited in full rather than discarded,
along with its diagnosis.

## Licence

Code: MIT. Text, figures and data: CC BY 4.0.

## Citing

> Stacey, C. (2026). *Latent Brittleness: Measurement Dependence and Failure Heterogeneity in
> Neural Network Robustness.* Zenodo. https://doi.org/10.5281/zenodo.22120112

## Corrections welcome

Several initially compelling findings in this project are withdrawn rather than published,
because adversarial review caught them. If something here looks too good, it may well be —
please open an issue.
