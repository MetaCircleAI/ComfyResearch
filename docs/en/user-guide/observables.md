---
doc_type: how-to
doc_status: stable-core
---

:::{div} cr-eyebrow
User Guide
:::

# Record Observables

:::{div} cr-article-lead
Measure training state at deliberate checkpoints without changing the loss
that drives optimization.
:::

## Observable, result, and visualization

An Observable defines what to measure. The Trainer records it at logging
points. A paired visualization consumes the recorded history. Keeping those
roles separate prevents a plotted quantity from being mistaken for an
optimization objective.

1. Add an Observable from the rail or node library.
2. Connect its output to the Trainer `observables` input.
3. Confirm that a paired Observable visualization exists.
4. Run a short graph and check that values reach that visualization.

:::{figure} ../_images/app/observable-and-visualization.png
:alt: Observable connected to a training graph with its recorded values shown in a visualization.
:class: cr-product-screenshot
:::

## Control measurement cost

Observables do not change the primary loss merely because they are connected,
but their measurement cost is real. Accuracy and scalar norms may be cheap;
Hessian, spectral, representation, or attention measurements can dominate a
small training run.

Use log frequency as part of the measurement design:

- log frequently enough to resolve the phenomenon of interest;
- log less often for expensive second-order or high-dimensional measurements;
- reduce model or sample size when validating a new Observable;
- temporarily disable extra Observables when isolating a training failure.

Changing the logging schedule changes the sampled signal. Keep it fixed when
comparing curves and record it with the graph.

## Decide what supports the claim

Attach an Observable because it can distinguish the hypotheses being tested,
not because it creates an interesting chart. For every measurement, be able to
state:

- which training state it reads;
- how it reduces or samples that state;
- when it is recorded;
- what comparison would support or contradict the claim;
- what approximation or truncation limits its interpretation.

Save the settings with the graph, then use
[Make a result reproducible](reproducibility.md) before reporting it.
