---
doc_type: reproduction
doc_status: phenomenon
template_id: edge-of-stability-cpu
---

:::{div} cr-eyebrow
Optimization Stability and Sharpness · Phenomenon reproduction
:::

# Edge of Stability on a Small CPU MLP

:::{div} cr-article-lead
Full-batch gradient descent drives the top Hessian eigenvalue toward
$2 / \eta = 10$, where it fluctuates while training loss falls overall but not
at every update.
:::

::::::{div} cr-article-meta
:::::{div} cr-meta-person
::::{div} cr-avatar
WJ
::::
::::{div} cr-meta-copy
:::{div} cr-meta-label
Author
:::
**王金鑫 (Wang Jinxin)**
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

This example is a small, local CPU demonstration of edge-of-stability
dynamics from Damian, Nichani, and Lee. With full-batch gradient descent at
$\eta = 0.2$, the top Hessian eigenvalue rises toward $2 / \eta = 10$ and then
fluctuates around that cutoff while loss continues to decrease
non-monotonically. It is a qualitative reproduction, not a numeric replication
of the paper's curves.
:::

**Paper:** [Self-Stabilization: The Implicit Bias of Gradient Descent at the
Edge of Stability](https://arxiv.org/abs/2209.15594), Damian, Nichani, and Lee
(ICLR 2023)

**Template:** `Edge of Stability (CPU)`

**Template ID:** `edge-of-stability-cpu`

## Reproduction Goal

For gradient descent with learning rate $\eta$, the classical local stability
cutoff for the largest Hessian eigenvalue $\lambda_1(\theta)$ is

$$
\lambda_1(\theta) \leq \frac{2}{\eta}.
$$

The paper reports two linked observations. First, sharpness progressively
rises to this cutoff. Then, at the edge of stability, it hovers near the
cutoff while training loss keeps decreasing without being monotonic.

This graph records $\lambda_1$, $\lambda_2$, and loss at every update. The
Hessian visualization draws the cutoff as a dashed line at 10.

## Paper Experiment and Reproduction Boundary

The paper establishes edge-of-stability behavior across several standard
neural-network settings and gives a self-stabilization explanation: an
unstable excursion in the top-eigenvector direction can reduce local curvature
and restore stability. This template retains the fixed-learning-rate,
full-batch mechanism but reduces the task so that it runs quickly on a local
CPU.

| Item | Paper | Current Comfy Research template |
| --- | --- | --- |
| Claim | Sharpness reaches and hovers around $2 / \eta$ while loss decreases non-monotonically | Same qualitative signature |
| Data and model | Multiple neural-network settings | Fixed synthetic linear regression and a small ReLU MLP |
| Optimizer | Full-batch or large-batch gradient descent settings | Full-batch SGD with zero momentum and weight decay |
| Sharpness measurement | Largest Hessian eigenvalue | Top two Hessian eigenvalues, $\lambda_1$ and $\lambda_2$ |
| Runtime goal | Empirical and theoretical study | Short CPU demonstration |

The graph therefore tests a mechanism-level pattern, not a point-by-point
match to a paper figure. Its fixed seed makes the expected shape easy to
inspect, but does not establish seed robustness or universality.

## Experiment Configuration

| Item | Template setting |
| --- | --- |
| Dataset | Fixed synthetic linear regression, 80 training samples, input dimension 10, output dimension 1, noise 0.1, seed 0 |
| Model | ReLU MLP, depth 2, width 24, seed 0, 889 trainable parameters |
| Objective | Mean squared error |
| Optimizer | SGD, learning rate 0.2, momentum 0, weight decay 0 |
| Batch size | Full batch (`batchSize = -1`) |
| Device | CPU |
| Training | 80 updates, log every update |
| Observable | Top two Hessian eigenvalues in descending order |
| Reference line | Sharpness cutoff $2 / \eta = 10$ |

With zero momentum and weight decay, the optimizer is full-batch gradient
descent. The graph is intentionally small and fixed so the qualitative pattern
can be inspected on a local CPU.

## Run in Comfy Research

1. Open **Templates** and load `Edge of Stability (CPU)`.
2. Confirm that `Trainer` uses CPU, full batch, 80 updates, and log frequency
   1.
3. Run **Train** on `Trainer`.
4. Inspect loss in `Training viz`.
5. Inspect $\lambda_1$ and $\lambda_2$ in `Hessian viz` against the dashed
   threshold at 10.

## Results

The expected signature has three parts:

1. $\lambda_1$ rises toward 10 in early training.
2. After reaching the cutoff, $\lambda_1$ fluctuates around it while
   $\lambda_2$ remains lower.
3. Loss has local increases after the first cutoff crossing, but ends below
   its value at that crossing.

:::{figure} ../../_images/app/edge-of-stability-cpu.png
:alt: Comfy Research graph with an 80-step CPU MLP run, training loss, and the top two Hessian eigenvalues around the sharpness cutoff.
:class: cr-reproduction-screenshot
:::

## Interpretation

The result is consistent with the paper's edge-of-stability account: leading
curvature repeatedly approaches and crosses the cutoff while loss still makes
net progress. The lower second eigenvalue confines the observed instability to
the leading-curvature direction rather than showing an indiscriminate increase
in all measured curvature.

This small run does not, by itself, verify the paper's cubic
self-stabilization analysis or its projected-gradient-descent prediction. It
only makes the reported qualitative behavior directly inspectable.

## Limitations

- The synthetic task and small MLP are not the paper's empirical settings.
- One fixed seed does not measure seed-to-seed variation.
- The graph records only the top two Hessian eigenvalues, not the full local
  spectrum or the paper's projected-gradient-descent trajectory.

The result should therefore be read as a fast qualitative demonstration of
edge-of-stability behavior, not as a paper-faithful benchmark.
