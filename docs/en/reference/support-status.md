---
doc_type: reference
doc_status: stable-core
---

:::{div} cr-eyebrow
Reference
:::

# Support status

## Documented stable workflow

This documentation treats the following path as the stable core for the
current source tree:

- projects and the graph workbench;
- Dataset, Model, Loss, Optimizer, Trainer, and their typed connections;
- local CPU, MPS, and detected CUDA execution;
- configured SSH remote GPU execution;
- Observables and their result visualizations;
- Templates, graph files, and export tiers;
- in-repository Node and Observable definition generation.

Stable here means the docs provide a supported learning and reference path. It
does not create a semantic-version compatibility promise for an untagged
development build.

PVI, LMech, Test, and Architecture IQ were removed ahead of the first release;
the pre-removal tree is preserved under the `archive/pre-v1-trim` tag. Apps,
Workflows, and the assistant chat were removed later in the same pre-release
window: Apps never shipped a panel, chat was already unreachable from the
workbench, and the Workflows rail carried a multi-canvas experiment tree whose
parameter sweeps are expressed as list-valued node fields on a single canvas
instead. The Jekyll blog export hung off that tree and described a sweep as a
grid of canvases, so it was removed with it. The LPD curve-phase algorithms
remain in the source tree because CurveStarer analysis uses them via
`/api/curve-lpd/predict`, and `scripts/graph_report_pdf.py` remains a
standalone graph-to-PDF command.

## Version path

This site is published under `/en/0.1.0/` and describes the `v0.1.0` release.
It should not retroactively describe older or newer releases. Other tagged
releases should build documentation from their tags and publish it under
distinct version paths.

When documentation and source disagree, the validated source contract wins for
this release and the documentation should be corrected.
