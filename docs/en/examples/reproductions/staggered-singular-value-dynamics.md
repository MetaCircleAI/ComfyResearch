---
doc_type: reproduction
doc_status: phenomenon
template_id: a04f21b3-e31c-4ba3-b8b9-d3af752f77d4
---

:::{div} cr-eyebrow
Learning Phases and Feature Formation · Phenomenon reproduction
:::

# Staggered Singular-Value Dynamics

:::{div} cr-article-lead
A small deep linear network learns the strongest modes of a linear mapping
before weaker modes, making effective rank grow in stages.
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

This graph makes the sequential mode-learning dynamics of deep linear networks
inspectable. It is a qualitative reproduction of the mechanism analyzed by
Saxe, McClelland, and Ganguli.
:::

**Paper:** [Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks](https://arxiv.org/abs/1312.6120), Saxe, McClelland, and Ganguli (ICLR 2014)

**Template:** `staggered sv dynamics`

**Template ID:** `a04f21b3-e31c-4ba3-b8b9-d3af752f77d4`

## Reproduction Goal

The effective map of a deep linear model is the product of its layer matrices.
With a small orthogonal initialization, directions with larger target singular
values should emerge first. The graph records leading singular values of the
effective map so their staggered rises can be compared directly.

## Experiment Configuration

| Item | Template setting |
| --- | --- |
| Data | Noiseless 8-to-8 synthetic linear regression; 2,000 train and 400 test examples |
| Model | Three-matrix deep linear MLP: depth 2, width 8, identity activation |
| Initialization | Saxe initialization with scale $\epsilon = 0.2$ |
| Objective and optimizer | MSE; SGD at learning rate 0.005 |
| Training | 8,000 CPU updates, batch size 256, log every 40 updates |
| Observable | Leading singular values of the product of learned weight matrices |

## Run in Comfy Research

1. Open **Templates** and load `staggered sv dynamics`.
2. Run **Train** on the main Trainer.
3. Inspect leading singular values against training loss and checkpoints.

:::{figure} ../../_images/app/staggered-sv-template.png
:alt: Loaded Staggered Singular-Value Dynamics template with the linear dataset, deep linear MLP, Saxe initialization, Trainer, and weight-product singular-value observable.
:class: cr-reproduction-screenshot
:::

## Results and Interpretation

The expected trajectory is staged: the leading singular value rises and levels
off before later values make their main transition. This is a finite-step,
finite-data signature of mode-wise learning, not evidence that every network
learns features one at a time.

:::{figure} ../../_images/app/staggered-sv-results.png
:alt: Weight product singular-value visualization from the completed staggered dynamics run, showing four modes emerging at different times.
:class: cr-reproduction-screenshot
:::

## Limitations

- The theory is for deep linear models; nonlinear models can couple modes.
- Mini-batch noise and a finite learning rate can blur the stages.
- Only the leading part of the spectrum is visualized.
