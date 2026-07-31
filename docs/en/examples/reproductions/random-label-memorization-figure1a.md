---
doc_type: reproduction
doc_status: phenomenon
---

:::{div} cr-eyebrow
Generalization, Memorization, and Reproducibility · Phenomenon reproduction
:::

# Memorizing Random Labels on CIFAR-10

:::{admonition} Compute cost
:class: cr-compute-warning

The complete comparison trains five Small Inception models for 25,000 steps
each on CIFAR-10. A CUDA device is strongly recommended.
:::

:::{div} cr-article-lead
A Small Inception network reaches 100% training accuracy on true labels, fixed
random labels, and heavily randomized inputs.
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

This example reproduces the learning-curve experiment in Figure 1(a) of Zhang
et al. Five Small Inception models are trained on true labels, fixed random
labels, globally shuffled pixels, independently shuffled pixels, and Gaussian
inputs. Every condition reaches 100% training accuracy and a final interval
loss below 0.0014. The overall convergence order matches the paper. The
random-pixel and random-label curves fall later than their visual positions in
the original figure.
:::

**Paper:** [Understanding Deep Learning Requires Rethinking
Generalization](https://arxiv.org/abs/1611.03530)

**Template:** `repro: Zhang et al. Figure 1(a) random-label memorization`

**Template ID:** `repro-random-label-memorization-fig1a`

## Reproduction Goal

The experiment asks whether conventional neural networks fit only meaningful
input-label relationships or whether they can also memorize arbitrary
assignments. Figure 1(a) compares training loss under five fixed data
conditions:

- **true labels:** unchanged CIFAR-10 images and labels;
- **random labels:** every training label is replaced by a fixed uniform
  random class;
- **shuffled pixels:** one fixed pixel permutation is shared by all images;
- **random pixels:** every image receives its own fixed pixel permutation; and
- **gaussian:** image values are replaced by fixed Gaussian inputs with matched
  statistics.

The target phenomenon is complete training-set fitting in all five conditions,
with the true-label curve dropping first and the random-label curve last.

## Experiment Configuration

| Item | Current reproduction |
| --- | --- |
| Dataset | CIFAR-10, 50,000 training and 10,000 test examples |
| Preprocessing | Divide by 255, center crop to 28x28, per-image whitening |
| Model | Small Inception, 1,649,402 trainable parameters |
| Objective | Cross-entropy |
| Optimizer | SGD, momentum 0.9, weight decay 0 |
| Learning rate | 0.1 initially, multiplied by 0.95 each epoch |
| Batch size | 128 |
| Training | 25,000 steps per condition |
| Main curve | Sample-weighted online cross-entropy, aggregated every 100 steps |
| Repetition | One fixed seed per condition |

Batch size, initialization, minibatch order, and the exact averaging window
are unspecified for Figure 1(a). The reproduction records deterministic values
for these settings because they can shift the horizontal location and local
smoothness of the curves.

## Run in Comfy Research

1. Open **Templates** and load
   `Understanding Deep Learning Requires Rethinking Generalization`.
2. Inspect the five dataset conditions and confirm that each connects to an
   independent Trainer.
3. Run the true-label Trainer as the ordinary CIFAR-10 baseline.
4. Run the random-label, shuffled-pixel, random-pixel, and Gaussian Trainers.
5. Compare the five loss histories in the curve-series visualization.
6. Use the full-training-set diagnostic to confirm that low interval loss also
   corresponds to complete memorization.

The five runs are independent and may be executed sequentially.

## Results

All five runs reached 100% full-training-set accuracy. Their final losses and
threshold-crossing times were:

:::{figure} ../../_images/app/random-label-memorization-figure1a-results.png
:alt: Comfy Research curve-series table and loss visualization for true labels, random labels, shuffled pixels, random pixels, and Gaussian inputs.
:class: cr-reproduction-screenshot

**Five-condition result view.** The Comfy Research table identifies the five
captured series. The adjacent visualization shows true labels converging first
and fixed random labels converging last.
:::

| Condition | Loss $\leq 0.1$ | Loss $\leq 0.01$ | Final interval loss | Full-train loss |
| --- | ---: | ---: | ---: | ---: |
| True labels | 4.0k steps | 5.7k steps | 0.000270 | 0.000111 |
| Shuffled pixels | 5.6k steps | 6.7k steps | 0.000170 | 0.000083 |
| Gaussian | 8.8k steps | 9.9k steps | 0.000257 | 0.000154 |
| Random pixels | 11.9k steps | 13.0k steps | 0.000803 | 0.000405 |
| Random labels | 16.6k steps | 17.7k steps | 0.001383 | 0.000765 |

The threshold times preserve the paper's overall ordering: meaningful labels
are fitted first, shared input structure helps, and arbitrary labels require
the longest high-loss phase. Random pixels and random labels converge later
than in a visual reading of the original Figure 1(a). The comparison metrics
are ordering and threshold times.

## Interpretation

Training accuracy alone does not distinguish meaningful input-label structure
from memorization. The same architecture and optimizer drive training error to
zero even when labels contain no information about the inputs.

The true-label model reached 86.71% test accuracy. The random-label model
reached 10.31%, close to chance on CIFAR-10. These supplementary values are
outside Figure 1(a)'s learning-curve comparison.

## Limitations

- One deterministic seed is used per condition, so seed-to-seed uncertainty is
  unmeasured.
- Unreported implementation choices can explain part of the horizontal
  difference from the original curves.
- Coverage is limited to Figure 1(a) and the Small Inception CIFAR-10 setting.
- The experiment measures memorization capacity. The generalization mechanism
  of stochastic gradient methods remains outside the scope.
