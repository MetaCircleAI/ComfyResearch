---
doc_type: reproduction
doc_status: phenomenon
template_id: repro-rank-collapse-tinyshakespeare-pretraining
---

:::{div} cr-eyebrow
Representation Compression and Information Flow · Spectral audit
:::

# Rank Collapse on Real Text (TinyShakespeare)

:::{div} cr-article-lead
A small causal Transformer trained from scratch on real TinyShakespeare
 text reveals the same warmup → expansion → compression RankMe signature
 that the paper measures on intermediate Pythia/OLMo checkpoints.
:::

::::::{div} cr-article-meta
:::::{div} cr-meta-person
::::{div} cr-avatar
GS
::::
::::{div} cr-meta-copy
:::{div} cr-meta-label
Author
:::
**郭绍阳 (Guo Shaoyang)**
::::
:::::
:::::{div} cr-meta-scope
::::{div} cr-meta-copy
:::{div} cr-meta-label
Scope
:::
**Phenomenon reproduction (spectral audit)**
::::
:::::
::::::

:::{admonition} Abstract
:class: cr-abstract

This template trains a small causal Transformer from scratch on real
TinyShakespeare text so the entire rank-collapse pipeline remains
editable and runnable inside Comfy Research. RankMe and alphaReQ are
computed on the final-hidden representation of every token position at
each logged step. This is a spectral audit, not the paper's full
intermediate-checkpoint experiment on real pretraining corpora.
:::

**Paper:** [Tracing the Representation Geometry of Language Models from Pretraining to Post-training](https://arxiv.org/abs/2509.23024)

**Template:** `repro: Rank Collapse TinyShakespeare spectral audit`

**Template ID:** `repro-rank-collapse-tinyshakespeare-pretraining`

## Reproduction Goal

Show that the final-hidden RankMe curve **rises, then falls** over
training on real TinyShakespeare text.

## Experiment Configuration

### Paper vs. template settings

| Item | Paper (main experiment) | Template (spectral audit) |
| --- | --- | --- |
| Corpus | real pretraining corpora (FineWeb eval) | real TinyShakespeare word-level corpus |
| Model | Pythia / OLMo (intermediate checkpoints) | small causal Transformer trained from scratch |
| Vocab | large subword BPE | corpus-derived top-256 words + PAD/BOS/EOS/UNK |
| Context | long-context eval | 32-token windows |
| RankMe/alphaReQ | last-token hidden state | every token position as a sample from `lm_head::input` |
| Horizon | full pretraining | 20,000 steps (evidence horizon, not CI) |

### Node graph

```text
tinyshakespeare_lm_dataset  ──►  trainer
transformer_token_model      ──►  trainer
adamw_optimizer              ──►  trainer
cross_entropy_loss           ──►  trainer
observable_representation_rankme ──►  trainer
observable_representation_alpha_req ──►  trainer
trainer  ──► training_visualization
trainer  ──► observable_viz  (RankMe)
trainer  ──► observable_viz  (alphaReQ)
```

### Key parameters

The paper's model, corpus, and horizon are not directly portable to a
from-scratch CI-friendly run, so this template scales every axis down
at once: a `modelDim = 4`, 4-head, 4-layer causal Transformer (paper:
Pythia/OLMo-scale checkpoints) over a `contextLength = 32` window on
the real TinyShakespeare corpus (paper: FineWeb eval), trained for
`20,000` steps (paper: full pretraining) with AdamW
`learningRate = 0.005`, `beta2 = 0.99`, `weightDecay = 0.001`, and
`batchSize = 32`. RankMe and alphaReQ are both read from
`lm_head::input`, sampling every token position rather than only the
last one. The exact per-node values (including `numHeads`, `ffDim`,
`epsilon`, and the observable flags) are locked by the Template's
Baseline Test.

## Run in Comfy Research

1. Open **Templates** and load `repro: Rank Collapse TinyShakespeare spectral audit`.
2. Click **Train** on the `Trainer` node. The first run will download
   the TinyShakespeare corpus if it is not already cached locally.
3. Inspect the `Final-hidden RankMe` panel for the warmup → expansion →
   compression signature over the 20,000-step horizon.
4. Cross-check `Final-hidden alphaReQ` for the power-law tail exponent;
   it should track the RankMe curve inversely during compression.


### Hyperparameter sensitivity: weight decay

The template default uses `learningRate = 0.005` and `weightDecay = 0.001`.
At `weightDecay = 0.01` the collapse signature is suppressed: the decay
flattens the narrow `d_model = 4` spectrum before the transient RankMe
expansion can form, so warmup → expansion → compression never separates
from noise. This makes the optimizer setting part of the phenomenon
boundary, not an arbitrary default.

### Data is downloaded, not synthesized

The `tinyshakespeare_lm_dataset` node downloads the real TinyShakespeare
corpus. **Download failure is an error, never a synthetic fallback** -
this preserves the "real text" claim of the audit.

## Results

The screenshot below comes from the real ComfyResearch UI: the
template-default `modelDim = 4` graph after training, with live
RankMe / alphaReQ / loss panels visible.

:::{figure} ../../_images/app/rank-collapse-tinyshakespeare-results-screenshot.png
:alt: ComfyResearch canvas after the rank-collapse training run completed
:class: cr-reproduction-screenshot

Live ComfyResearch UI after training: optimizer (`lr=0.005`,
`weightDecay=0.001`), Trainer (`100000` steps in this run), and the
RankMe / alphaReQ / Training Viz panels showing the collapse signature.
:::

### Tuning the bottleneck

**`modelDim`** sets the width of the final-hidden bottleneck that
RankMe measures. Everything else (dataset, steps, optimizer) is held
fixed at the template defaults; only `modelDim` (and the matching
`numHeads`) is overridden. Smaller `modelDim` gives a tighter bottleneck,
and the compression phase is easier to see and holds up longer. Try `2`,
`4`, and `8` yourself in the Trainer's `modelDim` field and watch how the
RankMe trajectory changes.

If the phenomenon doesn't show up, shrink `modelDim` or lower
`weightDecay` -- a bottleneck that's too wide or too strongly regularized
hides the compression phase within any practical step budget.

The screenshot below is a real negative example: `modelDim = 8` under the
same final settings (`lr=0.005`, `weightDecay=0.001`, `100000` steps in
this run). The RankMe curve dips briefly around the early steps, but the
bottleneck is too wide to sustain compression; the rank quickly returns
to a high value and stays there, confirming that the lack of collapse is
not a hyperparameter mistake but a width effect.

:::{figure} ../../_images/app/rank-collapse-tinyshakespeare-dmodel8-results-screenshot.png
:alt: TinyShakespeare modelDim=8 results showing no sustained rank collapse
:class: cr-reproduction-screenshot

`modelDim = 8` real-text run: a transient early dip is erased by the
wide bottleneck, so the collapse signature does not persist.
:::

## Interpretation

The run qualitatively supports the paper's central mechanism on real
text: even when a causal Transformer is trained from scratch (rather
than measured across pretraining checkpoints), the final-hidden
representation exhibits the same warmup → expansion → compression
RankMe signature. This suggests the phenomenon is a property of the
learning dynamics under cross-entropy on real language, not an artifact
of the checkpoint-sampling protocol.

Step budget also matters, not just bottleneck width. At the template
default (`modelDim = 4`), RankMe does not stay at its post-compression
trough within the first 20,000 steps: after dipping to `1.3271` at step
400 it recovers to `2.3789` by step 20,000. Extending the same run to
100,000 steps shows the rank continuing to drift downward, reaching
`2.0396` at step 100,000 with a near-flat terminal slope
(`-6.4e-7` per step over the last 5,000 steps). The "partial collapse"
label therefore depends on horizon: the early trough is sharp and real,
but full equilibration happens slowly and the rank does not instantly
plateau at the trough. A narrower bottleneck (`modelDim = 2`) suppresses
this slow dynamics and stays collapsed through 20,000 steps, while a
wide enough bottleneck (`modelDim = 8`) never sustains compression at
this horizon.

## Limitations

- Small model (`d = 4`, 4 layers, 4 heads) on a tiny corpus
  (TinyShakespeare), trained from scratch rather than measured across
  Pythia/OLMo pretraining checkpoints; this is not a claim about
  large-scale pretraining.
- The audit measures every token position as a sample from
  `lm_head::input`, whereas the paper evaluates last-token hidden
  states on FineWeb.
- 20,000 steps is the default evidence horizon, not CI; the smoke test
  only shortens an in-memory copy of the template to verify the pipeline
  runs, it does not assert the phenomenon. A 100,000-step continuation
  was run for `modelDim = 4` to check long-horizon behavior.
- Download failure of the TinyShakespeare corpus is an error, never a
  synthetic fallback.
- No test split is used; the audit is about representation geometry,
  not generalization.
- No Grokking task is used.

These limitations mean the result should be read as a small-scale
spectral audit that corroborates the mechanism, not as a replication of
the paper's full main experiment.