---
doc_type: reproduction
doc_status: phenomenon
---

:::{div} cr-eyebrow
Optimization Stability and Sharpness · Phenomenon reproduction
:::

# Cyclic Batch Size vs. Cyclic Learning Rate

:::{admonition} Compute cost
:class: cr-compute-warning

This reproduction is computationally expensive. Choose the experiment
configuration with care.
:::

:::{div} cr-article-lead
Matched periodic changes in $\eta/B$ produce similar overall learning
dynamics across five seeds.
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

This example uses Comfy Research to reproduce the central experiment in the
left panel of Figure 1 from Jastrzębski et al.: whether cyclic batch size (CBS)
and cyclic learning rate (CLR) produce similar training dynamics. This is a
qualitative reproduction; it does not claim point-by-point or numeric
replication of the paper's curves.
:::

**Paper:** [Three Factors Influencing Minima in SGD](https://arxiv.org/abs/1711.04623)

**Template:** `repro: Jastrzębski Fig 1 cyclic CBS vs CLR`

**Template ID:** `repro-jastrzbski-fig1-vgg11`

## Reproduction Goal

The paper studies the relationship between learning rate, batch size, and the
final point reached by SGD. The authors focus in particular on the ratio of
learning rate to batch size:

$$
\frac{\eta}{B}
$$

where $\eta$ is the learning rate and $B$ is the batch size. The left panel of
Figure 1 compares two periodic training strategies:

- **CBS:** Keep the learning rate fixed and change the batch size periodically.
- **CLR:** Keep the batch size fixed and change the learning rate periodically.

The two experiments use matching ranges of $\eta/B$. Similar overall accuracy
trajectories would support the paper's qualitative observation that, in this
experimental setting, a learning-rate schedule can be replaced by a matching
batch-size schedule.

## Paper Experiment and Reproduction Boundary

The current template represents the VGG-11 with Batch Normalization experiment
on CIFAR-10 from the left panel of Figure 1. It retains the CBS-versus-CLR
comparison and aggregates results over five initialization seeds.

| Item | Current Comfy Research template |
| --- | --- |
| Dataset | CIFAR-10: 45,000 training samples and 10,000 test samples |
| Data split | `RandomState(777)`, split seed 777 |
| Data processing | Pixel mean/global standard deviation normalization, random crop, horizontal flip, and shuffling each epoch |
| Model | VGG-11 with Batch Normalization |
| Initialization | Seeds 0, 1, 2, 3, and 4 |
| Objective | Cross-entropy |
| Optimizer | SGD, momentum 0, weight decay 0 |
| Training length | 300 data epochs |
| Result aggregation | Mean over the five seeds, computed by Trainer run |

This reproduction checks whether the two schedules yield similar training
dynamics. It does not provide per-epoch numeric errors, confidence intervals,
or pixel-level alignment with the original figure, so its conclusions remain
qualitative.

## Experiment Configuration

CBS and CLR share the data, model, objective, and initialization seeds. Only
the batch-size and learning-rate schedules differ.

| Configuration | CBS | CLR |
| --- | --- | --- |
| Learning rate | Fixed at 0.005 | Cycles between 0.001 and 0.005 |
| Batch size | Cycles between 128 and 640 | Fixed at 128 |
| Schedule | 5 epochs high, 5 epochs low | 5 epochs high, 5 epochs low |
| Cycle length | 10 epochs | 10 epochs |
| Trainer | `Trainer CBS (cyclic batch)` | `Trainer CLR (cyclic LR)` |

The $\eta/B$ values match exactly at both endpoints:

| Noise-scale phase | CBS | CLR | $\eta/B$ |
| --- | --- | --- | --- |
| High | $0.005 / 128$ | $0.005 / 128$ | $3.90625 \times 10^{-5}$ |
| Low | $0.005 / 640$ | $0.001 / 128$ | $7.8125 \times 10^{-6}$ |

Because its batch size changes with the epoch, CBS takes 63,450 optimizer steps
to complete 300 data epochs. CLR uses a fixed batch size of 128 and takes
105,600 steps. The comparison uses data epoch as the horizontal axis; it does
not require equal optimizer-step counts.

## Run in Comfy Research

1. Open Templates and load `repro: Jastrzębski Fig 1 cyclic CBS vs CLR`.
2. Confirm that a training device is available. The template uses CUDA by
   default and can run remotely through Remote GPU.
3. Run **Train series** on `Trainer CBS (cyclic batch)` to complete the CBS
   experiments for seeds 0-4.
4. Run **Train series** on `Trainer CLR (cyclic LR)` to complete the CLR
   experiments for seeds 0-4.
5. Open `Fig 1 acc plot (5-seed mean)` and compare the mean CBS and CLR
   train/test accuracy curves.

The two Trainers can run sequentially. The final `curve_series_table` collects
`train_acc` and `test_acc`, and the result plot aggregates the five seeds by
Trainer run.

## Results

In the completed run, the five-seed mean CBS and CLR training and test
accuracies show similar periodic changes and overall trends. The primary result
is that test accuracy for the two schedules remains consistent rather than
identical at every epoch.

The result is consistent with the central qualitative observation in Figure 1:
when the periodic variation in $\eta/B$ matches, cyclic batch size and cyclic
learning rate can produce similar training dynamics.

:::{figure} ../../_images/app/jastrzebski-cbs-clr-curves.png
:alt: Comfy Research graph and results showing the five-seed cyclic batch-size and cyclic learning-rate accuracy curves.
:class: cr-reproduction-screenshot
:::

## Interpretation

This experiment supports a conclusion constrained by its experimental
conditions: on CIFAR-10, with VGG-11 plus Batch Normalization and the current
SGD configuration, CBS and CLR schedules with matching $\eta/B$ behave
similarly.

## Limitations

This experiment alone cannot establish that $\eta/B$ is sufficient to
determine training dynamics for every model, dataset, and optimizer. Gradient
covariance, momentum, regularization, data order, and model architecture can
still affect the result. A stricter numeric replication would also need to
save the complete curves for both schedules, report variance across seeds, and
compare quantitatively with the paper's data.
