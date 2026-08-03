---
doc_type: reproduction
doc_status: phenomenon
template_id: repro-rank-collapse-figure5-linear-ce
---

:::{div} cr-eyebrow
Representation Compression and Information Flow · Phenomenon reproduction
:::

# Rank Collapse in a Linear Bottleneck

:::{div} cr-article-lead
Skewed class frequency plus a representation bottleneck can produce
primacy bias, entropy expansion, and late-window compression - the three
phases of the Rank Collapse trajectory.
:::

::::::{div} cr-article-meta
:::::{div} cr-meta-person
::::{div} cr-avatar
GS
::::
::::{div} cr-meta-copy
:::{div} cr-meta-label
Author
:::
**郭绍阳 (Guo Shaoyang)**
::::
:::::
:::::{div} cr-meta-scope
::::{div} cr-meta-copy
:::{div} cr-meta-label
Scope
:::
**Phenomenon reproduction (calibrated)**
::::
:::::
::::::

:::{admonition} Abstract
:class: cr-abstract

This template reconstructs Figure 5 of the Rank Collapse paper on a
six-sample orthogonal toy. The full softmax cross-entropy trajectory is
run on a bias-free linear factorization `θ ∈ R^{6×2}, F = Sθ, W ∈ R^{2×4}`
with `d = 2 < |V| = 4`. The canvas directly renders the Figure 5D RankMe
curve and captures `F`, `W`, and the two eigenvalue histories for the
B/C/D report. This is a post-hoc deterministic figure reconstruction,
not exact numerical recovery and not a seed-robustness claim.
:::

**Paper:** [Tracing the Representation Geometry of Language Models from Pretraining to Post-training](https://arxiv.org/abs/2509.23024)

**Template:** `repro: Rank Collapse Figure 5 linear CE (calibrated)`

**Template ID:** `repro-rank-collapse-figure5-linear-ce`

## Reproduction Goal

The paper claims that skewed class frequency together with a
representation bottleneck can yield three intertwined phenomena:

1. **Primacy bias** - frequent classes are learned first.
2. **Selection bias** - the per-sample representation drift magnitude
   `|dσ_i/dt|` is proportional to the current singular value `σ_i`.
3. **Late-window compression** - after an initial entropy expansion,
   the effective rank (RankMe) compresses toward the bottleneck
   dimension `d`.

Figure 5 illustrates this on six inputs with labels `[0, 0, 1, 1, 2, 3]`,
giving class counts `[2, 2, 1, 1]`. Appendix B assumes orthogonal rows
`SSᵀ = I`; this template chooses the canonical coordinates `S = I₆`,
with no test split.

On this toy, the target observable is a three-stage RankMe trajectory
rather than loss decrease alone: an initial **warmup** in which the
two-dimensional representation stays close to its starting rank, an
**entropy expansion** in which frequent classes are learned first and
the representation briefly becomes more isotropic, and a **late-window
compression** in which the leading singular values separate and RankMe
decreases within the published 0-300 step window. This calibrated
Figure-5D-style trajectory is the phenomenon claimed by this template;
it does not claim exact numerical recovery, arbitrary-seed robustness,
or persistent/asymptotic collapse.

## Paper Experiment and Reproduction Boundary

The paper does not disclose its learning rate, numerical initialization,
seed, or toy source code. The template therefore locks a *calibrated*
configuration that reproduces the published 0-300 step window.

| Item | Paper (Figure 5) | Template |
| --- | --- | --- |
| Dataset | six orthogonal samples, labels `[0,0,1,1,2,3]` | `S = I₆`, same labels |
| Class counts | `[2,2,1,1]` (skewed) | same |
| Model | `θ ∈ R^{6×2}`, `F = Sθ`, `W ∈ R^{2×4}` | bias-free linear factorization, both Linear biases frozen at zero |
| Bottleneck | `d = 2 < |V| = 4` | same |
| Objective | full softmax cross-entropy | same |
| Optimizer | not disclosed | SGD, `lr = 0.06`, `momentum = 0`, `weight decay = 0` |
| Initialization | Theorem B.2 assumes `FᵀF = WWᵀ` | `rank_aligned_initialization`, balanced singular values `[1.56, 1.20]` |
| Seed | not disclosed | `25` |
| Training length | 0-300 plotted updates | `300` steps, full-batch (`batchSize = -1`) |
| Device | not disclosed | `cpu` |

This reproduction checks whether the calibrated template reconstructs
the published 0-300 step RankMe trajectory. It does not provide
arbitrary-seed robustness, exact numeric recovery of the paper's curve,
or a persistent/asymptotic collapse claim, so its conclusions remain a
post-hoc deterministic figure reconstruction.

## Experiment Configuration

### Node graph

```text
paper_classification_dataset  ──►  trainer
mlp_model                      ──►  trainer
rank_aligned_initialization   ──►  mlp_model
sgd_optimizer                  ──►  trainer
cross_entropy_loss             ──►  trainer
observable_representation_rankme ──►  trainer
observable_accuracy            ──►  trainer
trainer  ──► training_visualization
trainer  ──► observable_viz  (RankMe trajectory, Figure 5D)
trainer  ──► observable_viz  (accuracy)
```

### Key parameters

The table above already fixes every item the paper discloses (dataset,
class counts, model shape, bottleneck, objective); this template's own
choices only fill in what the paper leaves undisclosed - `mlp_model`
realizes the `d = 2` bottleneck as `depth = 1`, `width = 2`,
`activation = "identity"`; `rank_aligned_initialization` balances the
singular values via `scale = 1.2`, `singularRatio = 1.3`; and
`observable_representation_rankme` reads `representationId = "0::output"`
with `captureTrajectories = true` so the full RankMe path can be
plotted. The exact per-node values are locked by the Template's
Baseline Test.

## Run in Comfy Research

1. Open **Templates** and load `repro: Rank Collapse Figure 5 linear CE (calibrated)`.
2. Click **Train** on the `Trainer` node.
3. Inspect the `Figure 5D · RankMe trajectory` panel for the three phases:
   warmup → entropy expansion → late-window compression.
4. The `Full-training accuracy` panel confirms that the model still fits
   the skewed training set despite the rank collapse.

The data is generated on the fly by `paper_classification_dataset` - no
offline dataset file is needed. Changing `experimentMode`, `inputDim`,
or `trainSize` immediately regenerates the synthetic matrix.

## Results

The screenshot below comes from the real ComfyResearch UI: the calibrated
Figure 5 template after the Trainer reached `complete` at 300 steps, with
loss, accuracy, and the Figure 5D RankMe trajectory visible.

:::{figure} ../../_images/app/rank-collapse-figure5-results-screenshot.png
:alt: ComfyResearch Rank Collapse Figure 5 Template after training completed
:class: cr-reproduction-screenshot

Live ComfyResearch UI after the 300-step full-batch run. The
`Representation RankMe Viz` panel shows the warmup → expansion →
compression signature within the published 0-300 step window.
:::

The calibrated full run exhibits the paper's three-phase signature within
the 0-300 step window:

- **Warmup (approximately steps 0-40):** RankMe moves from `1.9096807` to
  a trough of `1.8919100` at step 33.
- **Entropy expansion (approximately steps 40-214):** RankMe rises to
  `1.9931071` at step 214 as the representation differentiates.
- **Late-window compression (steps 214-300):** RankMe falls to
  `1.9839966`; the terminal slope is `-1.34984e-4` per step and the
  singular-value ratio falls from `0.846641` to `0.775271`.

Extending the same configuration beyond the published window changes the
picture. The screenshot below keeps the same dataset, model, optimizer, and
initialization but runs for `1000` steps:

:::{figure} ../../_images/app/rank-collapse-figure5-1000steps-results-screenshot.png
:alt: Figure 5 template extended to 1000 steps
:class: cr-reproduction-screenshot

Extending the run to 1000 steps: after the initial compression, RankMe
re-expands and approaches the `d = 2` ceiling. The finite-window collapse
signature is not persistent at this horizon.
:::

The feature-Gram alignment error is `4.68749e-8` in the full evidence,
consistent with the balanced initialization protocol used by this
mechanism-level reconstruction.

## Interpretation

The run qualitatively supports the paper's central mechanism: under
skewed class frequency and a `d < |V|` bottleneck, full-batch softmax
cross-entropy produces a RankMe trajectory with warmup, expansion, and
terminal compression. The template makes the bias-free linear
factorization explicit (`F = Sθ`, `W` direct) so that the
`dσ_i/dt ∝ σ_i` selection-bias relation can be read off the two
eigenvalue histories captured by the `observable_representation_rankme`
node.

## Limitations

- Six-sample toy only; no natural corpus and no test split.
- Calibrated `lr = 0.06`, `seed = 25`, singular values `[1.56, 1.20]`
  are not paper-backed; they were chosen to reconstruct the published
  window.
- The `0-300` published window is reconstructed. Extending the same
  configuration to `10,000` steps re-expands RankMe toward the `d = 2`
  ceiling; **persistent/asymptotic collapse is not claimed**. The late
  decline is finite-window only.
- The calibrated run is a deterministic full-evidence result at seed 25.
  An audit over seeds 0-99 passed the strict Figure-5D landmark contract
  for `1/100` seeds, so this is not a random-seed robustness result.
- Uniform-class, no-bottleneck, and MSE controls do not show the same
  negative terminal slope. They are controls for the disclosed protocol,
  not proof of a universal causal effect.
- Theorem B.2's aligned-initialization invariant `FᵀF = WWᵀ` is
  preserved by construction; Theorem B.3's dominant-class-size bound
  additionally assumes near-uniform initial predictions and `|V| ≫ 1`,
  so its quantitative bound is not exact for this four-class toy.
- The class-wise primacy ordering (Figure 5B/5C) is not separately
  validated by this template - only the Panel-D RankMe signature.
- No Grokking task is used.
- The repository's Template tests lock the core nodes, connections,
  parameters, and RankMe Observable in a baseline test; a separate
  in-memory CI smoke test shortens the training budget and only checks
  that the Trainer reaches `complete` with finite loss and Observable
  values - it does not claim to reproduce the three phases. The full
  evidence above is the 300-step CPU run that supports the phenomenon
  statement.

These limitations mean the result should be read as a mechanism-level
stress test, not a strict numeric replication or a universal causal
claim.
