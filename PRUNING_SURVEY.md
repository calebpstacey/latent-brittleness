# Pruning parameterisation survey

**Purpose.** §5 identifies a confound that is active only where damage is operationalised as
a fraction of weights. This survey establishes what parameterisations the literature
actually uses, by reading implementations rather than papers.

**Method.** For each entry we located the line of code computing the pruning threshold and
recorded (a) whether the count removed derives from a *size* or from a *fixed target*, and
(b) whether it is applied per layer or across the whole network. Every entry below was read
directly from source. Implementations are cited by repository and file so a reader can
verify.

**n = 8 implementations.** A sample, not a census, and reported as such.

---

## Results

| # | implementation | file | threshold computation | class | scope |
|---|---|---|---|---|---|
| 1 | PyTorch `torch.nn.utils.prune` | `torch/nn/utils/prune.py` | `amount` is float *or* int | **both** | either |
| 2 | Frankle & Carbin, LTH release | `foundations/pruning.py` | `round(percent * sorted_weights.size)` | fractional | per-layer |
| 3 | Frankle et al., `open_lth` | `pruning/sparse_global.py` | `ceil(pruning_fraction * n_remaining)` | fractional | **global** |
| 4 | Lee et al., SNIP | `snip/model.py` | `int(round(num_params * (1 - target_sparsity)))` | fractional | global |
| 5 | Wang et al., GraSP | `pruner/GraSP.py` | `int(len(all_scores) * (1 - keep_ratio))` | fractional | global |
| 6 | Tanaka et al., SynFlow | `Pruners/pruners.py` | `int((1.0 - sparsity) * numel())` | fractional | configurable |
| 7 | Han et al. (reimpl.), `prune_by_std` | `net/prune.py` | `np.std(layer_weights) * sensitivity` | **distributional** | per-layer |
| 8 | Han et al. (reimpl.), `prune_by_percentile` | `net/prune.py` | `np.percentile(abs(all_alives), q)` | fractional | global |

Repositories: `pytorch/pytorch`; `google-research/lottery-ticket-hypothesis`;
`facebookresearch/open_lth`; `namhoonlee/snip-public`; `alecwangcq/GraSP`;
`ganguli-lab/Synaptic-Flow`; and a widely used PyTorch reimplementation of Han et al. (2015).

Entry 6 additionally implements SNIP, GraSP, magnitude and random pruning under the same
threshold arithmetic, so the fractional pattern covers five methods in that file alone.

---

## Findings

**1. Fractional dominates; scope does not.** Seven of eight compute the count removed as a
fraction of a size. Only entry 7 does not. But *scope* splits nearly evenly — three
per-layer, four global, one configurable at runtime. Since surviving capacity relates to
architecture differently under the two scopes, this matters as much as the fractional
question.

**2. At least four non-equivalent parameterisations are in use.**

- **fractional per-layer** — count scales with each layer's size
- **fractional global** — count scales with total network size; layer allocation emergent
- **absolute** — count independent of size (available, but not the documented default)
- **distributional** — threshold from the weight distribution; count emerges from training

The fourth is the most consequential. Under `prune_by_std`, two layers with different weight
distributions retain different fractions at the same `sensitivity` setting. The relationship
between architecture and surviving capacity is then mediated by whatever shaped those
distributions during training, and cannot be read off the architecture at all.

**3. "Fraction" denotes different quantities — but not all differences are of one kind.**
GraSP's `ratio` is a prune fraction (`keep_ratio = 1-ratio`); SynFlow's `sparsity` is a keep
fraction. These are **algebraically invertible**: one-shot, and producing identical retained
sets under the appropriate substitution. They are notational variants, not distinct
parameterisations — though SynFlow's naming is inverted relative to its meaning, since
`sparsity` denotes the fraction *retained*, as its own consistency check
(`remaining_params - total_params*sparsity`) confirms.

`open_lth`'s `pruning_fraction` is different in kind: a fraction *of currently remaining
weights*, applied iteratively. The retained set after $n$ rounds depends on the schedule and
not only on the terminal density, so it does **not** reduce to a one-shot fraction. Two
notational variants and one functionally distinct scheme. An earlier version of this survey
described these as "three meanings", conflating the two kinds of difference.

**4. The choice is a type annotation in the standard API.** PyTorch's pruning functions take
`amount` as int or float: float is a fraction, int an absolute count. The two
parameterisations whose disagreement §5 documents are therefore the same function call with a
different Python type, and nothing in the signature indicates the choice can invert an
architectural conclusion. The official tutorial demonstrates the fractional form applied per
module.

**5. The clearest evidence that the choice is treated as an implementation detail.** The same
research group implemented the same method under two different scopes. The LTH release code
(entry 2) applies a percent to each layer independently; `open_lth` (entry 3), the authors'
later framework, concatenates all layers and thresholds once. Neither file mentions the
difference, and neither is wrong. The choice simply is not treated as methodological.

---

## What this licenses in §5

Not "the field uses the wrong parameterisation" — none of these is wrong. The defensible
claim is:

> The parameterisation of internal damage is not standardised and is not treated as a
> methodological choice. Across eight implementations we find at least four non-equivalent
> schemes, a functionally distinct iterative scheme alongside two
> notational variants of "fraction", and one case of the same authors switching scope between two releases of the same method without comment. In the
> most widely used pruning API, the choice between the two parameterisations whose
> disagreement we document is a Python type.
>
> The confound identified in §5 requires **both** conditions to be met: it is active
> wherever a conclusion **comparing architectures of different sizes** is drawn **under a
> fractional parameterisation**. Neither alone suffices. A study reporting compression
> ratios for a single architecture is unaffected however it parameterises damage; a study
> comparing architectures under an absolute or distributional scheme is unaffected too.
> Seven of eight implementations in this sample satisfy the second condition; we did not
> systematically record the first, which is the obvious extension.

## Limitations

- **n = 8.** Proportions are indicative, not estimates.
- Entries were selected for influence and code availability, not at random. Highly cited
  work with public code is over-represented.
- Two entries are third-party reimplementations (Han et al.; SynFlow's versions of SNIP and
  GraSP), though for GraSP we verified the original separately and it matches.
- We did not systematically record whether each *paper* draws an architectural conclusion.
  That column determines where the confound is live rather than merely present, and is the
  obvious extension.
