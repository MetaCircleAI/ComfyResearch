---
doc_type: reproduction
doc_status: phenomenon
---

:::{div} cr-eyebrow
Learning Phases and Feature Formation · Phenomenon reproduction
:::

# Lazy vs. Rich Training Regimes

:::{div} cr-article-lead
Output scaling controls whether a wide ReLU network learns moving features or
stays close to its initialization.
:::

::::::{div} cr-article-meta
:::::{div} cr-meta-person
::::{div} cr-avatar
QQ
::::
::::{div} cr-meta-copy
:::{div} cr-meta-label
Author
:::
**屈清宇 (Qu Qingyu)**
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

This runnable graph contrasts feature learning with lazy, kernel-like training
on a two-dimensional teacher-student problem. It is a qualitative exploration
of the scaling mechanism described by Chizat, Oyallon, and Bach.
:::

**Paper:** [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956), Chizat, Oyallon, and Bach (NeurIPS 2019)

**Template:** `lazy vs rich regime`

**Template ID:** `d2e8cdd7-d14c-42c3-94dc-ba2b419c07f9`

## Reproduction Goal

The output multiplier $\alpha$ controls the size of the network output at
initialization. Small $\alpha$ permits substantial feature motion (the rich
regime); large $\alpha$ keeps weights close to their starting point (the lazy
or tangent-kernel regime). The graph makes that distinction visible through
loss, weight displacement, and hidden-neuron trajectories.

## Experiment Configuration

| Item | Template setting |
| --- | --- |
| Data | Deterministic two-dimensional teacher data; 200 train and 500 test points |
| Teacher and student | ReLU MLPs with 3 and 200 hidden neurons respectively |
| Controlled parameter | Student `outputScale` $\alpha$ (initially 0.01; suitable for a sweep) |
| Optimizer | Full-batch SGD, learning rate 0.01 |
| Training | 10,000 CPU updates; log every 100 updates |
| Observables | Weight norm, train/test gap, relative displacement, and 2-D neuron trajectories |

## Run in Comfy Research

1. Open **Templates** and load `lazy vs rich regime`.
2. Run at a small `outputScale`, such as 0.01, and inspect displacement and
   neuron trajectories.
3. Repeat with larger values and compare final loss and relative weight motion.

:::{figure} ../../_images/app/lazy-vs-rich-template.png
:alt: Loaded Lazy vs. Rich template in the Comfy Research canvas, showing teacher data, the scaled student MLP, Trainer, and observables.
:class: cr-reproduction-screenshot
:::

## Results and Interpretation

Small $\alpha$ should produce visible hidden-feature motion; at larger
$\alpha$, relative displacement should be small and learning should resemble a
fixed-feature model. The transition is not a universal threshold: it depends
on width, initialization, optimizer, and time scaling.

:::{figure} ../../_images/app/lazy-vs-rich-small-output-scale.png
:alt: Neuron trajectory visualization after a Lazy vs. Rich run with a small output scale, showing compact hidden-feature motion.
:class: cr-reproduction-screenshot
:::

:::{figure} ../../_images/app/lazy-vs-rich-large-output-scale.png
:alt: Neuron trajectory visualization after a Lazy vs. Rich run with a large output scale, showing a wider feature trajectory.
:class: cr-reproduction-screenshot
:::

## Limitations

- This is a synthetic two-dimensional teaching setting.
- A sweep does not establish asymptotic scaling laws.
- Fair comparisons across $\alpha$ can require time and loss rescaling.
