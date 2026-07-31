---
doc_type: how-to
doc_status: stable-core
---

:::{div} cr-eyebrow
User Guide
:::

# Make a result reproducible

:::{div} cr-article-lead
Preserve the executable graph and enough scientific context for another person
to understand, rerun, and challenge the result.
:::

## Record the experiment contract

Before sharing a result, record:

- the exact Template ID, graph file, or committed artifact revision;
- dataset source, sizes, split seed, preprocessing, and sampling mode;
- model architecture, initialization seed, and relevant dimensions;
- optimizer, learning-rate and batch schedules, and regularization;
- steps or epochs, log frequency, compute device, and software environment;
- every Observable used to support the conclusion and its settings;
- the number of seeds and how runs were aggregated;
- the expected result and the condition that would count against it.

## Preserve the right evidence

Use Small when the graph and parameters are enough to rerun the experiment.
Use Medium when the exact plotted history is part of the evidence. Use Large
only when trained model state must be inspected or resumed, and document its
size and handling requirements.

Keep machine-local credentials and unrelated workspace state out of all three
tiers. A screenshot can orient a reader or show expected evidence, but it is
not a substitute for numeric data, the graph artifact, or environment details.

## State the claim boundary

A completed graph does not prove paper-faithful reproduction. Similar-looking
curves do not prove numeric agreement. A single seed does not establish a
distributional result, and a saved checkpoint does not describe the data or
measurement protocol that produced it.

Label the result accurately:

- **Smoke test:** the execution and result-routing path works.
- **Phenomenon reproduction:** the setup demonstrates a qualitative behavior.
- **Paper-faithful reproduction:** the documented implementation, data,
  protocol, and evaluation match the target closely enough for the stated
  comparison.

The reproduction pages in [Examples](../examples/index.md) state this boundary
and list their limitations explicitly.
