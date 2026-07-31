---
doc_type: overview
doc_status: stable-core
---

:::{div} cr-eyebrow
Examples
:::

# Reproductions

:::{div} cr-lead
Classic learning-mechanics and physics-of-AI phenomena, reproduced as runnable
graphs. Each article states its experimental boundary, method, result, and
limitations; visual similarity alone is not presented as numeric replication.
:::

:::{div} cr-claim-legend
**Claim legend:** **PHENOMENON** reproduces the reported qualitative behavior;
**PAPER-FAITHFUL** follows the original experimental setup closely enough for
that stronger claim. The current collection contains sixteen phenomenon
reproductions. Author attribution appears inside each article.
:::

## Optimization Stability and Sharpness

How learning rate, batch size, noise, and curvature interact during
optimization.

:::::{grid} 1 1 2 3
:gutter: 3
:class-container: cr-example-grid

::::{grid-item-card} Cyclic Batch Size vs. Cyclic Learning Rate
:link: reproductions/jastrzebski-fig1-cyclic-cbs-vs-clr
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
Matched cyclic noise-scale schedules produce similar five-seed training
dynamics.
:::

:::{div} cr-card-tags
`noise-scale` `optimization`
:::

:::{div} cr-template-id
Template: `repro-jastrzbski-fig1-vgg11`
:::
::::

::::{grid-item-card} Edge of Stability on a Small CPU MLP
:link: reproductions/edge-of-stability-cpu
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
The top Hessian eigenvalue reaches and fluctuates around $2 / \eta$ while
full-batch loss decreases non-monotonically.
:::

:::{div} cr-card-tags
`sharpness` `full-batch`
:::

:::{div} cr-template-id
Template: `edge-of-stability-cpu`
:::
::::

::::{grid-item-card} Edge of Stability (EOS)
:link: reproductions/edge-of-stability-eos
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
Leading curvature approaches 2 / eta while full-batch loss makes
non-monotonic net progress.
:::

:::{div} cr-card-tags
`sharpness` `full-batch`
:::

:::{div} cr-template-id
Template: `97ff01a8-1724-43ec-8c38-fdb62bbe5faf`
:::
::::

::::{grid-item-card} Small-Batch vs. Large-Batch Training
:link: reproductions/keskar-fig2-3-sb-lb
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
A generalization gap appears alongside greater path-wise sharpness near the
large-batch solution.
:::

:::{div} cr-card-tags
`generalization` `sharpness`
:::

:::{div} cr-template-id
Template: `repro-keskar-fig23-sb-lb`
:::
::::

:::::

## Learning Phases and Feature Formation

How networks enter distinct regimes, learn dominant modes, and specialize.

:::::{grid} 1 1 2 3
:gutter: 3
:class-container: cr-example-grid

::::{grid-item-card} Lazy vs. Rich Training Regimes
:link: reproductions/lazy-vs-rich-regime
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
Output scaling controls whether a wide ReLU network learns moving features or
stays close to its initialization.
:::

:::{div} cr-card-tags
`feature-learning` `scaling`
:::

:::{div} cr-template-id
Template: `d2e8cdd7-d14c-42c3-94dc-ba2b419c07f9`
:::
::::

::::{grid-item-card} Staggered Singular-Value Dynamics
:link: reproductions/staggered-singular-value-dynamics
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
Deep linear training learns the strongest effective-map singular modes before
weaker ones.
:::

:::{div} cr-card-tags
`deep-linear` `singular-values`
:::

:::{div} cr-template-id
Template: `a04f21b3-e31c-4ba3-b8b9-d3af752f77d4`
:::
::::

::::{grid-item-card} Spectral Bias: Low Frequencies Are Learned First
:link: reproductions/spectral-bias-figure1a
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
Ten equal-amplitude Fourier components reach their target amplitudes in a
consistent low-to-high frequency order.
:::

:::{div} cr-card-tags
`spectral-bias` `fourier`
:::

:::{div} cr-template-id
Template: `repro-spectral-bias-fig1a`
:::
::::

::::{grid-item-card} Exact Solution for On-Line Learning in Multilayer Neural Networks
:link: reproductions/saad_solla_plateau_reproduction
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
On-line gradient-descent learning exhibits a plateau in generalization error
before specialization eliminates the redundant student node.
:::

:::{div} cr-card-tags
`teacher-student` `generalization`
:::

:::{div} cr-template-id
Template: `e399fd7d-e107-44d0-94b6-7e2159392253`
:::
::::

:::::

## Representation Compression and Information Flow

How internal representations expand, compress, and retain task-relevant
information.

:::::{grid} 1 1 2 3
:gutter: 3
:class-container: cr-example-grid

::::{grid-item-card} Rank Collapse in a Linear Bottleneck
:link: reproductions/rank-collapse-figure5
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
Skewed class frequency and a d=2 bottleneck produce warmup, entropy expansion,
then late-window RankMe compression.
:::

:::{div} cr-card-tags
`rank-collapse` `representation`
:::

:::{div} cr-template-id
Template: `repro-rank-collapse-figure5-linear-ce`
:::
::::

::::{grid-item-card} Rank Collapse on Real Text (TinyShakespeare)
:link: reproductions/rank-collapse-tinyshakespeare
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
A small causal Transformer trained on real TinyShakespeare text shows the same
warmup → expansion → compression RankMe signature as intermediate pretraining
checkpoints.
:::

:::{div} cr-card-tags
`rank-collapse` `language-model`
:::

:::{div} cr-template-id
Template: `repro-rank-collapse-tinyshakespeare-pretraining`
:::
::::

::::{grid-item-card} Information Bottleneck Dynamics Across Training-Set Sizes
:link: reproductions/information-bottleneck-figure3
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
Information-plane trajectories show early label-information growth, later
leftward motion, and more final label information with more training data.
:::

:::{div} cr-card-tags
`information-bottleneck` `representation`
:::

:::{div} cr-template-id
Template: `dafb8339-a932-4b10-b3b6-185fc53a5a4f`
:::
::::

:::::

## Generalization, Memorization, and Reproducibility

How models fit their data, generalize beyond it, and agree across training
runs.

:::::{grid} 1 1 2 3
:gutter: 3
:class-container: cr-example-grid

::::{grid-item-card} Diffusion Reproducibility Across Data Scale
:link: reproductions/diffusion-reproducibility
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
Same-noise diffusion outputs progress from unstable memorization to
fine-grained, less-train-near agreement as CIFAR-10 data scale increases.
:::

:::{div} cr-card-tags
`diffusion` `reproducibility` `generalization`
:::

:::{div} cr-template-id
Template: `repro-diffusion-same-init-different-seed`
:::
::::

::::{grid-item-card} Memorizing Random Labels on CIFAR-10
:link: reproductions/random-label-memorization-figure1a
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
A Small Inception model reaches 100% training accuracy on true labels, fixed
random labels, and three randomized-input conditions.
:::

:::{div} cr-card-tags
`memorization` `generalization`
:::

:::{div} cr-template-id
Template: `repro-random-label-memorization-fig1a`
:::
::::

:::::

## Loss Geometry and Parameter Symmetry

How solutions relate across parameter space and how training breaks geometric
invariants.

:::::{grid} 1 1 2 3
:gutter: 3
:class-container: cr-example-grid

::::{grid-item-card} Linear and Curve Mode Connectivity
:link: reproductions/linear-mode-connectivity
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
Direct interpolation becomes less favorable during training, while a quadratic
Bezier path remains low-loss between the final same-init endpoints.
:::

:::{div} cr-card-tags
`loss-landscape` `mode-connectivity` `CIFAR-10`
:::

:::{div} cr-template-id
Template: `repro-linear-mode-connectivity-cifar10`
:::
::::

::::{grid-item-card} Neural Mechanics: Symmetry and Broken Conservation Laws
:link: reproductions/Neural_Mechanics
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
Finite learning rates and weight decay break the conservation laws of gradient
flow; VGG-11 on CIFAR-10 traces the full centripetal-centrifugal spectrum.
:::

:::{div} cr-card-tags
`symmetry` `conservation-laws`
:::

:::{div} cr-template-id
Template: `b38ae9dd-735b-46c4-973f-a850a2a55544`
:::
::::

:::::

## In-Context Learning

How sequence models acquire task behavior from the current context.

:::::{grid} 1 1 2 3
:gutter: 3
:class-container: cr-example-grid

::::{grid-item-card} In-Context Associative Recall
:link: reproductions/in-context-associative-recall
:link-type: doc
:class-card: cr-example-card

{bdg-secondary}`PHENOMENON`

:::{div} cr-card-summary
A causal Transformer retrieves a paired value from a synthetic context and
exposes the attention patterns used during recall.
:::

:::{div} cr-card-tags
`in-context-learning` `attention`
:::

:::{div} cr-template-id
Template: `5d1a2ab4-825d-4251-8a5a-d7b83b42c1d7`
:::
::::

:::::

```{toctree}
:hidden:

reproductions/jastrzebski-fig1-cyclic-cbs-vs-clr
reproductions/edge-of-stability-cpu
reproductions/lazy-vs-rich-regime
reproductions/staggered-singular-value-dynamics
reproductions/edge-of-stability-eos
reproductions/keskar-fig2-3-sb-lb
reproductions/diffusion-reproducibility
reproductions/linear-mode-connectivity
reproductions/rank-collapse-figure5
reproductions/rank-collapse-tinyshakespeare
reproductions/in-context-associative-recall
reproductions/information-bottleneck-figure3
reproductions/Neural_Mechanics
reproductions/random-label-memorization-figure1a
reproductions/saad_solla_plateau_reproduction
reproductions/spectral-bias-figure1a
```
