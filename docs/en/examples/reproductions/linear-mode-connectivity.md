---
doc_type: reproduction
doc_status: phenomenon
---

:::{div} cr-eyebrow
Loss Geometry and Parameter Symmetry · Phenomenon reproduction
:::

# Linear and Curve Mode Connectivity

:::{div} cr-article-lead
As two same-init SGD runs train longer, their direct parameter interpolation
develops a larger loss barrier, while a learned quadratic Bezier path remains
low-loss at 50k steps.
:::

::::::{div} cr-article-meta
:::::{div} cr-meta-person
::::{div} cr-avatar
WX
::::
::::{div} cr-meta-copy
:::{div} cr-meta-label
Author
:::
**王星运 (Wang Xingyun)**
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

This example studies a minimal mode-connectivity contrast on CIFAR-10. Two
ResNet classifiers share an initialization and differ only in training-seed
randomness. Direct interpolation between their parameters becomes increasingly
unfavorable during training, but a quadratic Bezier path optimized at 50k steps
removes most of the barrier. This is a controlled phenomenon reproduction, not
the original paper's multi-architecture, independently initialized benchmark.
:::

**Paper:** [Loss Surfaces, Mode Connectivity, and Fast Ensembling of
DNNs](https://arxiv.org/abs/1802.10026), Timur Garipov, Pavel Izmailov,
Dmitrii Podoprikhin, Dmitry Vetrov, and Andrew Gordon Wilson (NeurIPS 2018)

**Template:** `repro: Linear Mode Connectivity (CIFAR-10, same init)`

**Template ID:** `repro-linear-mode-connectivity-cifar10`

## Reproduction Goal

Garipov et al. show that low-loss neural-network solutions can be connected by
simple curves with nearly constant accuracy. This template tests the most
direct visual consequence: a poor straight parameter path need not mean that
the two endpoints are disconnected.

For endpoint checkpoints $\theta_A$ and $\theta_B$, the linear evaluator
measures

$$
\theta(\alpha) = (1-\alpha)\theta_A + \alpha\theta_B.
$$

At 50k steps, the curve evaluator keeps the endpoints fixed and optimizes one
quadratic Bezier control point $\theta_C$:

$$
\theta(t) = (1-t)^2\theta_A + 2t(1-t)\theta_C + t^2\theta_B.
$$

The phenomenon is a large linear loss barrier paired with a much smaller
Bezier barrier.

## Paper Experiment and Reproduction Boundary

| Item | Paper | Current Comfy Research template |
| --- | --- | --- |
| Data | CIFAR-10, CIFAR-100, and ImageNet | CIFAR-10 only, fixed full split |
| Architectures | VGG, Wide ResNet, and ResNet families | One small self-defined ResNet |
| Endpoints | Independently initialized minima | Same initialization seed 0; trainer seeds 0 and 1 |
| Paths | Quadratic Bezier and polygonal chains | Quadratic Bezier only, plus a 21-point linear baseline |
| Evaluation | Accuracy along curves and fast ensembling | Train/test cross-entropy and accuracy; loss barrier and accuracy drop |
| Scope | Broad geometry and ensembling study | One endpoint pair and a training-time trajectory |

A low Bezier barrier is evidence for the particular curve family and
optimization budget used here. It does not establish a globally optimal path
or a general theorem about all neural-network minima.

## Experiment Configuration

| Item | Setting |
| --- | --- |
| Dataset | CIFAR-10, 50,000 train / 10,000 test, fixed `subsetSeed = 0` |
| Model | Small ResNet, base channels 16, one block in each of four stages |
| Initialization | Shared model seed 0 |
| Optimizer and loss | Adam, learning rate `1e-3`, cross-entropy |
| Endpoint A / B | Trainer seed 0 / 1, batch size 128 |
| Checkpoint pairs | 1k, 5k, 10k, 20k, and 50k steps |
| Linear evaluator | 21 points from $\alpha=0$ to $1$, evaluation batch size 256 |
| BatchNorm | Recompute running statistics with 100 train batches for reported results |
| Bezier evaluator | 500 Adam updates, 4 path samples/update, learning rate `0.01` |

The test-loss barrier is the maximum path loss minus the worse endpoint loss.
The test-accuracy drop is the minimum path accuracy minus the worse endpoint
accuracy, so negative values are losses in percentage points.

## Run in Comfy Research

1. Open **Templates** and load `repro: Linear Mode Connectivity (CIFAR-10,
   same init)`.
2. Keep the dataset, ResNet seed, optimizer, and loss shared. Confirm that
   endpoint A uses trainer seed 0 and endpoint B uses trainer seed 1.
3. Train both endpoints to a matching checkpoint budget and save both model
   checkpoints. The default graph targets 50k steps.
4. Connect the matching pair to **Interpolation loss barrier**, enable
   **Recompute BN stats**, set 100 calibration batches, and click **Run**.
5. At 50k, run **Bezier mode connectivity**. Compare its reported linear and
   Bezier barriers, then inspect the train/test loss and accuracy curves.

## Results

The direct path worsens as the two SGD trajectories progress.

| Training step | Linear test-loss barrier | Linear test-accuracy drop |
| ---: | ---: | ---: |
| 1,000 | 0.2377 | -8.11 pp |
| 5,000 | 0.7414 | -14.16 pp |
| 10,000 | 1.0473 | -13.96 pp |
| 20,000 | 1.3301 | -13.66 pp |
| 50,000 | 1.8985 | -15.86 pp |

At 50k, optimizing one Bezier control point changes the conclusion about the
endpoints' connectivity.

| Path | Test-loss barrier | Test-accuracy drop |
| --- | ---: | ---: |
| Direct linear interpolation | 1.89855 | -15.86 pp |
| Optimized quadratic Bezier | 0.06864 | -3.28 pp |

:::{figure} ../../_images/app/linear-mode-connectivity-50k.png
:alt: Comfy Research canvas for the 50k-step CIFAR-10 linear mode connectivity run, including the two endpoint trainers, linear interpolation barrier, and optimized quadratic Bezier path.
:class: cr-reproduction-screenshot

**50k endpoint pair.** The direct linear path crosses a pronounced loss
barrier, whereas the optimized quadratic Bezier curve preserves low loss and
accuracy along most of the path.
:::

## Interpretation

1. **Linear disconnection emerges during training.** The near-monotonic growth
   from 0.2377 at 1k to 1.8985 at 50k is consistent with two trajectories
   progressively specializing into parameter configurations that are poorly
   combined by direct averaging.
2. **Loss retains information after accuracy saturates.** From 5k onward, the
   accuracy drop stays near 14--16 points while the cross-entropy barrier keeps
   increasing. The interpolated model therefore loses predictive confidence
   and margin even when top-1 error changes less.
3. **The barrier belongs to the straight path.** At 50k, the quadratic Bezier
   path lowers the loss barrier by about 96%. In this setting, the low-loss
   region is connected but strongly non-convex: the Euclidean segment leaves it,
   while a simple curved route remains.

The result is consistent with parameter specialization or misalignment across
the SGD runs, such as incompatible internal feature bases, channel
organizations, or normalization scales. This experiment does not identify the
precise mechanism because it does not include permutation alignment or
representation analysis.

## Limitations

- The endpoints share an initialization; the paper also studies independently
  initialized solutions and more architectures.
- One CIFAR-10 ResNet pair does not establish seed robustness or universality.
- The Bezier result depends on its control-point parameterization and finite
  optimization budget.
- No polygonal path, fast geometric ensemble, weight-distance, or feature
  analysis is included.
