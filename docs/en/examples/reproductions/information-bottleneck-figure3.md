---
doc_type: reproduction
doc_status: phenomenon
template_id: dafb8339-a932-4b10-b3b6-185fc53a5a4f
---

:::{div} cr-eyebrow
Representation Compression and Information Flow · Phenomenon reproduction
:::

# Information Bottleneck Dynamics Across Training-Set Sizes

:::{admonition} Compute cost
:class: cr-compute-warning

The interactive graph exposes one run for each training-set size. The reported
paper-scale result averages 50 independent runs per panel and is substantially
more expensive.
:::

:::{div} cr-article-lead
Across 5%, 45%, and 85% training subsets, hidden representations first gain
label information and later move toward lower input information; the final
deep-layer label information increases with the amount of training data.
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

This example reproduces the three information-plane panels in Figure 3 of
Shwartz-Ziv and Tishby. A 12-10-8-6-4-2-1 network is trained on 5%, 45%, and
85% of the paper's 4,096 binary input patterns. The completed 50-run aggregate
recovers the early increase in $I(T;Y)$, the later leftward movement of several
hidden layers, and the ordering of the final deep-layer $I(T;Y)$ values. The
reported values are diagnostics tied to the selected discretized estimator.
:::

**Paper:** [Opening the Black Box of Deep Neural Networks via
Information](https://arxiv.org/abs/1703.00810)

**Template:** `repro: Information-plane interactive approximation (Figure 3 protocol)`

**Template ID:** `dafb8339-a932-4b10-b3b6-185fc53a5a4f`

## Reproduction Goal

The information plane represents every hidden layer $T$ with two coordinates:

$$
\bigl(I(X;T), I(T;Y)\bigr).
$$

The paper describes two broad stages. During early training, layers primarily
move upward as they acquire label information. Later, several intermediate and
deep layers move left as the binned estimate of input information decreases.
Figure 3 also compares how the amount of training data changes the final
position of the deepest representation.

This reproduction checks three observable claims:

- the three training fractions produce the same broad trajectory direction;
- later layers exhibit a measurable decrease in binned $I(X;T)$; and
- final deep-layer $I(T;Y)$ rises from the 5% panel to the 85% panel.

## Paper Experiment and Reproduction Boundary

The formal batch run follows the target figure's architecture, three sample
fractions, 10,000-epoch horizon, 50 independent initializations and training
subsets, and the released IDNNs binning convention. Where the paper prose,
figure caption, and public code differ, the architecture and training horizon
follow the target figure. Optimizer and initialization details follow the
executable public implementation.

| Item | Current reproduction |
| --- | --- |
| Data | Verified `var_u.mat`: 4,096 binary 12-dimensional inputs |
| Training subsets | 205 (5%), 1,843 (45%), and 3,482 (85%) examples |
| Model | 12-10-8-6-4-2-1 MLP, tanh hidden layers, binary sigmoid output |
| Initialization | Released IDNNs fan-in truncated-normal initialization |
| Objective | Binary cross-entropy |
| Optimizer | Adam, learning rate 0.0004 |
| Training | 10,000 epochs, IDNNs logarithmic recording schedule |
| Information estimate | 30-anchor released-IDNNs discretization |
| Formal aggregation | 50 independent parameter initializations and subsets per panel |

Each graph Trainer runs one seed. The reported aggregate was produced
separately by the reproduction batch runner over 50 seeds.

## Run in Comfy Research

1. Open **Templates** and load
   `Opening the Black Box of Deep Neural Networks via Information`.
2. Choose one of the 5%, 45%, or 85% columns.
3. Confirm the dataset size, 12-10-8-6-4-2-1 network, Adam configuration, and
   information-plane observable.
4. Run **Train** on that column's Trainer.
5. Inspect the corresponding information-plane visualization.
6. Repeat for the other columns if a three-panel comparison is needed.

Use the reproduction batch runner to obtain the 50-repeat aggregate. Each
interactive Trainer produces one run.

## Results

All three formal panels completed 50 independent runs. The final values below
are means with one standard deviation across runs.

:::{figure} ../../_images/app/information-bottleneck-figure3-p05-ui.png
:alt: Comfy Research information-plane visualization for the 5 percent training subset, showing six hidden-layer trajectories.
:class: cr-reproduction-screenshot
:width: 480px

**5% training subset.** The Comfy Research result node shows the six
hidden-layer trajectories from the 50-run aggregate.
:::

:::{figure} ../../_images/app/information-bottleneck-figure3-p45-ui.png
:alt: Comfy Research information-plane visualization for the 45 percent training subset, showing six hidden-layer trajectories.
:class: cr-reproduction-screenshot
:width: 480px

**45% training subset.** The trajectories finish with more label information
than in the 5% condition.
:::

:::{figure} ../../_images/app/information-bottleneck-figure3-p85-ui.png
:alt: Comfy Research information-plane visualization for the 85 percent training subset, showing six hidden-layer trajectories.
:class: cr-reproduction-screenshot
:width: 480px

**85% training subset.** The deepest layer finishes near one bit of label
information while the later epochs move toward lower input information.
:::

| Training fraction | Final deep $I(X;T)$ | Final deep $I(T;Y)$ | Final train accuracy |
| --- | ---: | ---: | ---: |
| 5% | $1.152 \pm 0.023$ bits | $0.452 \pm 0.032$ bits | $0.9978 \pm 0.0043$ |
| 45% | $1.080 \pm 0.101$ bits | $0.824 \pm 0.033$ bits | $0.9981 \pm 0.0027$ |
| 85% | $1.112 \pm 0.108$ bits | $0.964 \pm 0.019$ bits | $0.9991 \pm 0.0026$ |

The most direct cross-panel result is the monotonic rise in final deep-layer
label information:

$$
0.452 \rightarrow 0.824 \rightarrow 0.964\ \text{bits}.
$$

The trajectory geometry also matches the target phenomenon. Early points move
mainly upward; middle and deep layers later move left. Using the maximum
recorded value minus the endpoint as a compression diagnostic, the largest
decreases were 6.637 bits for the 5% panel, 4.710 bits for the 45% panel, and
4.249 bits for the 85% panel.

## Interpretation

Under this data-generation rule, network, optimizer, and discretized estimator,
representations first acquire label information and later show lower estimated
input information. More training samples leave the deepest representation with
more estimated label information.

This run measures representation trajectories. It does not test whether
compression causes generalization.

## Limitations

- The reported coordinates depend on activation, binning, and optimizer
  choices.
- The interactive template is a single-seed entry point; the reported result
  comes from a separate 50-repeat aggregate.
- The paper's drift-diffusion analysis, information-bottleneck bound, and
  critical-point claims are outside this reproduction.
- Point-by-point digitization of the original figure's green transition
  markers was outside the scope.
