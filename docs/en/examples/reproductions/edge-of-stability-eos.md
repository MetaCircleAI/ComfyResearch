---
doc_type: reproduction
doc_status: phenomenon
template_id: 97ff01a8-1724-43ec-8c38-fdb62bbe5faf
---

:::{div} cr-eyebrow
Optimization Stability and Sharpness · Phenomenon reproduction
:::

# Edge of Stability (EOS)

:::{div} cr-article-lead
Full-batch gradient descent drives leading curvature toward $2 / \eta$, while
loss continues to make non-monotonic net progress.
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

This graph demonstrates edge-of-stability dynamics with a small deterministic
regression problem and an exact Hessian observable. It is a qualitative,
mechanism-level reproduction, not a numeric match to a paper figure.
:::

**Paper:** [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065), Cohen et al. (ICLR 2022)

**Template:** `Edge of stability(EOS)`

**Template ID:** `97ff01a8-1724-43ec-8c38-fdb62bbe5faf`

## Reproduction Goal

With $\eta = 0.2$, the local linear stability boundary is $2 / \eta = 10$.
The intended signature is a progressive rise in the largest Hessian eigenvalue,
followed by fluctuations near that value and a loss curve with local spikes but
net decrease.

## Experiment Configuration

| Item | Template setting |
| --- | --- |
| Data | Deterministic synthetic regression, 20 inputs to 1 output; 200 train and 50 test examples |
| Model | Two-hidden-layer tanh MLP, width 32, seed 0 |
| Optimizer | Full-batch SGD, learning rate 0.2, no momentum |
| Training | Two CPU Trainers: 10,000 updates, logged every 500 updates or every update |
| Observable | Exact leading Hessian eigenvalue and a reference line at $2 / \eta = 10$ |

## Run in Comfy Research

1. Open **Templates** and load `Edge of stability(EOS)`.
2. Run the Trainer paired with the Hessian visualization.
3. Inspect loss and $\lambda_{\max}$ together; use the every-update Trainer to
   resolve individual oscillations.

:::{figure} ../../_images/app/edge-of-stability-eos-template.png
:alt: Loaded Edge of Stability template with the deterministic regression graph, full-batch SGD, two CPU Trainers, and Hessian observable.
:class: cr-reproduction-screenshot
:::

## Results and Interpretation

$\lambda_{\max}$ should approach the dashed threshold and remain near it
rather than increasing without bound. Loss can rise on individual updates yet
trend down over the run, which is consistent with edge-of-stability behavior
in this fixed, full-batch setting.

:::{figure} ../../_images/app/edge-of-stability-eos-results.png
:alt: Completed EOS visualizations showing the leading Hessian eigenvalue near the 2 over eta threshold together with the training and test loss trajectories.
:class: cr-reproduction-screenshot
:::

## Limitations

- The small MSE regression task is not the paper's full experimental set.
- One seed is illustrative, not a robustness study.
- A leading eigenvalue does not establish the full self-stabilization analysis.
