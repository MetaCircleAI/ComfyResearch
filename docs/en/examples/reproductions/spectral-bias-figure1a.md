---
doc_type: reproduction
doc_status: phenomenon
template_id: repro-spectral-bias-fig1a
---

:::{div} cr-eyebrow
Learning Phases and Feature Formation · Phenomenon reproduction
:::

# Spectral Bias: Low Frequencies Are Learned First

:::{admonition} Compute cost
:class: cr-compute-warning

The paper-scale aggregate trains ten six-layer networks for 80,000 full-batch
Adam updates each. The bundled graph is a single-phase interactive entry point.
:::

:::{div} cr-article-lead
Ten equal-amplitude Fourier components are learned in frequency order: the
lowest component crosses the 0.8 amplitude threshold at 400 updates; the
highest crosses at 45,200.
:::

::::::{div} cr-article-meta
:::::{div} cr-meta-person
::::{div} cr-avatar
XC
::::
::::{div} cr-meta-copy
:::{div} cr-meta-label
Author
:::
**熊程宇 (Xiong Chengyu)**
::::
:::::
:::::{div} cr-meta-scope
::::{div} cr-meta-copy
:::{div} cr-meta-label
Scope
:::
**Phenomenon reproduction**
::::
:::::
::::::

:::{admonition} Abstract
:class: cr-abstract

This example reproduces the equal-amplitude experiment in Figure 1(a) of
Rahaman et al. Ten independently phased six-layer ReLU networks fit a
one-dimensional target made from ten equal-amplitude sine waves. The aggregated
DFT heatmap shows the same ordering as the paper: low-frequency components
reach their target amplitude before high-frequency components. The layer-weight
panel follows the public notebook's calculation. Its layer-to-layer shape
differs from the target panel and is reported separately.
:::

**Paper:** [On the Spectral Bias of Neural
Networks](https://arxiv.org/abs/1806.08734)

**Template:** `repro: Rahaman Spectral Bias Figure 1(a)`

**Template ID:** `repro-spectral-bias-fig1a`

## Reproduction Goal

The equal-amplitude target is

$$
f(x) = \sum_{i=1}^{10} \sin(2\pi k_i x + \phi_i),
$$

with $k_i \in \{5, 10, \ldots, 50\}$ and independently sampled phases
$\phi_i$. Equal amplitudes isolate frequency as the ordered variable.

The main observable is the normalized Fourier amplitude

$$
\frac{2\left|\operatorname{DFT}(f_\theta)[k_i]\right|}{A_i},
$$

recorded throughout training. Spectral bias predicts a visible low-to-high
frequency progression in this heatmap.

## Paper Experiment and Reproduction Boundary

| Item | Current reproduction |
| --- | --- |
| Input grid | 200 uniformly spaced points on $[0,1)$ |
| Frequencies | 5 through 50 in increments of 5 |
| Amplitudes | 1 for all ten components |
| Repetitions | 10 independent phase draws |
| Model | Six Linear layers, hidden width 256, ReLU between layers |
| Objective | Full-batch mean squared error |
| Optimizer | Adam, learning rate 0.0003 |
| Training | 80,000 updates per phase |
| Recording | Every 100 updates, 800 points per run |
| DFT display | Normalized amplitude clipped to $[0,1]$ |

The paper and appendix specify 80,000 updates. The current public notebook
defaults to 60,000. This reproduction follows the paper horizon and uses the
notebook's endpoint-excluded input grid, network semantics, optimizer, DFT
formula, and layer-weight expression.

## Run in Comfy Research

1. Open **Templates** and load `On the Spectral Bias of Neural Networks`.
2. Confirm the ten-frequency equal-amplitude target and the endpoint-excluded
   200-point grid.
3. Inspect the six-layer ReLU MLP, Adam optimizer, and full-batch Trainer.
4. Run **Train** for one phase and inspect the ten Fourier observables.
5. Compare when low-, middle-, and high-frequency amplitudes begin to rise.
6. Use the batch reproduction runner when a ten-phase mean and standard
   deviation are required.

The template gives an inspectable single-phase run. The values below come from
the completed ten-phase aggregate.

## Results

The mean iteration at which every normalized component first reached 0.8
increased with frequency:

| Frequency | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean update | 400 | 3,500 | 7,800 | 12,600 | 21,200 | 25,700 | 27,100 | 33,300 | 39,800 | 45,200 |

:::{figure} ../../_images/app/spectral-bias-figure1a-frequency-ui.png
:alt: Comfy Research Fourier-component visualization nodes for the lowest and highest frequencies in the ten-phase aggregate.
:class: cr-reproduction-screenshot

**Frequency endpoints.** The $k=5$ component reaches its target near the start
of training. The $k=50$ component rises across most of the 80,000-update run.
:::

:::{figure} ../../_images/app/spectral-bias-figure1a-layer-norm-ui.png
:alt: Comfy Research visualization nodes for the middle-frequency component and the spectral norms of six layer-weight matrices.
:class: cr-reproduction-screenshot

**Middle frequency and layer diagnostic.** The $k=25$ component reaches its
target between the two endpoint frequencies. The adjacent node shows the six
layer-weight spectral-norm series from the same ten-phase aggregate.
:::

At the final recorded point, update 79,900, the ten mean amplitudes ranged from
0.9437 to 1.0069. All target components were nearly fitted by the end, and
their learning onsets remained strongly ordered by frequency.

The public-notebook layer-weight calculation produced overall growth in the
four square hidden layers. Its final means were 537.39, 337.99, 455.25, and
660.57. The scale overlaps the paper. The stronger layer separation and Layer
5 dominance resemble the right panel of Figure 1(b) more than Figure 1(a).
This difference is listed in the limitations.

## Interpretation

Across ten phase draws, the ReLU network fits the smooth, low-frequency
structure before the rapidly varying, high-frequency structure. The aggregate
reproduces the reported spectral ordering.

High-frequency components are eventually fitted, with later threshold
crossings under this configuration.

## Limitations

- The layer-weight panel differs visibly from the target Figure 1(a) panel,
  despite using the public notebook expression.
- One source array contains a non-finite Layer 4 point at update 64,500. It is
  shown as missing, with no replacement or interpolation.
- Figure 1(b), architecture and optimizer ablations, and the paper's
  real-dataset experiments are outside this reproduction.
- Ten runs support the ordering for this controlled target. The causal
  mechanism for spectral bias remains outside the scope.
