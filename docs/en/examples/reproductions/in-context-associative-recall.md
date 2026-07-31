---
doc_type: reproduction
doc_status: phenomenon
---

:::{div} cr-eyebrow
In-Context Learning · Phenomenon reproduction
:::

# In-Context Associative Recall

:::{div} cr-article-lead
A small causal Transformer learns to retrieve the value paired with a queried
key from the current token sequence.
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

This graph trains a token Transformer on synthetic key-value sequences and
exposes accuracy, attention maps, and attention-relation scores. It is a
controlled demonstration of in-context associative recall, not a claim about
general-purpose language-model reasoning.
:::

**Template:** `in context associative recall`

**Template ID:** `5d1a2ab4-825d-4251-8a5a-d7b83b42c1d7`

## Reproduction Goal

Each example has four key-value pairs followed by a query key; the target is
the paired value from the same context. The attention views make it possible to
inspect whether query positions attend to relevant key or value positions as
accuracy improves.

## Experiment Configuration

| Item | Template setting |
| --- | --- |
| Data | Fixed synthetic recall sequences; vocabulary size 8 and four pairs per context |
| Data size | 10,000 train and 2,000 test sequences, seed 0 |
| Model | Causal Transformer: dimension 6, two heads, two layers, feed-forward dimension 24 |
| Objective and optimizer | Cross-entropy; Adam at learning rate 0.001 |
| Training | CPU, batch size 256; primary Trainer runs 4,000 updates and logs every 200 |
| Observables | Accuracy, attention maps, attention-relation scores, and metric comparisons |

## Run in Comfy Research

1. Open **Templates** and load `in context associative recall`.
2. Run the primary Trainer and inspect training and accuracy views.
3. Run the separate attention-diagnostics Trainer, then inspect its attention
   maps and relation scores.
4. Compare the two runs qualitatively: the diagnostic Trainer is an
   independent training run, not a checkpoint restored from the primary
   Trainer.

:::{figure} ../../_images/app/in-context-associative-recall-template.png
:alt: Loaded in-context associative recall template showing the synthetic token dataset, causal Transformer, accuracy Trainer, and attention-diagnostics Trainer.
:class: cr-reproduction-screenshot
:::

## Results and Interpretation

The expected outcome is rising held-out recall accuracy accompanied by
attention patterns that connect a query to relevant context items. These are
diagnostic probes for this synthetic task; they do not alone prove a unique
causal role for a particular attention head.

:::{figure} ../../_images/app/in-context-associative-recall-accuracy.png
:alt: Metric comparison visualization showing held-out accuracy alongside attention-relation scores during in-context associative recall training.
:class: cr-reproduction-screenshot
:::

:::{figure} ../../_images/app/in-context-associative-recall-copy-score.png
:alt: Attention relation score visualization for the previous token relation, with scores for the two selected layer-head pairs.
:class: cr-reproduction-screenshot
:::

:::{figure} ../../_images/app/in-context-associative-recall-induction-score.png
:alt: Attention relation score visualization for the induction relation, with scores for the two selected layer-head pairs.
:class: cr-reproduction-screenshot
:::

## Limitations

- Vocabulary, context length, and model size are intentionally tiny.
- Synthetic recall does not cover natural-language understanding.
- Attention scores are useful probes, not a complete causal explanation.
