---
doc_type: reproduction
doc_status: phenomenon
template_id: repro-keskar-fig23-sb-lb
---

:::{div} cr-eyebrow
Optimization Stability and Sharpness · Phenomenon reproduction
:::

# Small-Batch vs. Large-Batch Training

:::{div} cr-article-lead
Both regimes fit the training set, but the large-batch solution generalizes
worse and is sharper along the sampled parameter path.
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

This example uses Comfy Research to reproduce the central experiments from
Keskar et al. on the large-batch training generalization gap and sharp minima.
This is a qualitative reproduction; it does not claim point-by-point or
numeric replication of the paper's curves.
:::

**Paper:** [On Large-Batch Training for Deep Learning](https://arxiv.org/abs/1609.04836)

**Template:** `repro: Keskar Fig 2+3 SB/LB`

**Template ID:** `repro-keskar-fig23-sb-lb`

## Reproduction Goal

Keskar et al. compare small-batch (SB) and large-batch (LB) training. Both can
reach very high training accuracy, but LB commonly shows lower test accuracy,
creating a generalization gap. The paper relates this phenomenon to LB being
more likely to converge to a sharp minimum that is sensitive to parameter
perturbations.

The current template follows the CIFAR-10 with C1 path and contains two parts:

- **Figure 2:** Compare SB and LB train/test accuracy.
- **Figure 3:** Interpolate parameters between the SB and LB checkpoints, then
  evaluate loss and accuracy along the one-dimensional path.

## Experiment Configuration

The two Trainers share the same data, model architecture, initialization seed,
optimizer, and objective. Only the batch size and corresponding number of
optimizer steps differ.

| Item | SB | LB |
| --- | --- | --- |
| Dataset | CIFAR-10: 50,000 train / 10,000 test | Same as SB |
| Model | Keskar C1 CNN, seed 0 | Same as SB, seed 0 |
| Optimizer | Adam, lr=0.001, $\beta_1=0.9$, $\beta_2=0.999$, weight decay=0 | Same as SB |
| Objective | Cross-entropy | Cross-entropy |
| Batch size | 256 | 5,000 (10% of the training set) |
| Training length | 100 epochs, 19,600 steps | 100 epochs, 1,000 steps |
| Log frequency | 196 steps | 10 steps |

Figure 3 uses the checkpoint produced by each Trainer. The parameter path is:

$$
\theta(\alpha) = \theta_{SB} + \alpha(\theta_{LB} - \theta_{SB})
$$

Here, $\alpha=0$ corresponds to SB and $\alpha=1$ corresponds to LB. The
template samples 25 points over $[-1, 2]$. Along this path, trainable
parameters are interpolated; evaluation uses per-batch BatchNorm statistics.

## Run in Comfy Research

1. Open Templates and load `repro: Keskar Fig 2+3 SB/LB`.
2. Run **Train** on `Trainer SB (B=256)` to produce the SB checkpoint and
   train/test accuracy curves.
3. Run **Train** on `Trainer LB (B=5000)` to produce the LB checkpoint and its
   curves.
4. Compare the two sets of train/test accuracy curves in
   `Fig 2 accuracy plot`.
5. After confirming that both checkpoints are ready, run **Run** on
   `Fig 3 parametric path`.
6. In `Fig 3 path plot`, inspect how train/test loss and accuracy change with
   $\alpha$.

The two Trainers are independent; no manual rewiring is needed. They write
separate SB and LB checkpoints and append their curves to the Figure 2 series
table.

## Results

### Figure 2: A Generalization Gap Appears

Both runs eventually reach train accuracy close to 1.0, indicating that SB and
LB both fit the training set well. In the completed run, SB test accuracy is
generally higher and stabilizes at approximately 0.77-0.78, while LB test
accuracy stabilizes at approximately 0.74-0.76.

This result qualitatively reproduces the main observation from Figure 2: LB
does not perform worse because it failed to fit the training data. Its training
accuracy is also high, but its test performance is slightly lower.

:::{figure} ../../_images/app/keskar-sb-lb-results.png
:alt: Comfy Research graph and result plots showing small-batch and large-batch accuracy, plus the Figure 3 parameter path.
:class: cr-reproduction-screenshot
:::

### Figure 3: The LB Endpoint Is Sharper Along the Parameter Path

The parameter-path run produces four series: train/test loss and accuracy. Near
$\alpha=0$ (SB), loss remains low and accuracy is relatively stable. As the
path approaches $\alpha=1$ (LB), both loss and accuracy change more abruptly.
Extrapolating farther to $\alpha>1$ produces a clear rise in loss and a rapid
drop in accuracy.

This asymmetric one-dimensional slice qualitatively reproduces the visual
feature in Figure 3: the low-loss region near the SB solution is wider, while
the LB solution is more sensitive to parameter perturbations along this path.
It supports the interpretation that the LB solution is sharper, but one linear
path cannot represent the complete high-dimensional loss landscape or replace
the paper's sharpness metric.

## Interpretation

This run completes two independent SB/LB training paths, two checkpoints, the
Figure 2 accuracy comparison, and the Figure 3 parameter-interpolation
workflow. The results qualitatively support the paper's two main observations:
LB can fit the training set well while obtaining lower test accuracy than SB,
and the LB solution is sharper than the SB solution along the sampled parameter
path.

## Limitations

- The current result uses only the C1 network and a single initialization,
  seed 0; it does not report variance across seeds.
- The paper also includes F1/F2 and C2-C4; the current template covers only
  CIFAR-10 with C1.
- Figure 3 examines only one linear path between SB and LB. This is not
  equivalent to the complete high-dimensional loss landscape.
- The result verifies the reported qualitative phenomena. It provides no
  point-by-point error or numeric alignment with the paper's original figures.

These limitations mean the result should not be presented as a strict numeric
replication or a universal causal conclusion.
