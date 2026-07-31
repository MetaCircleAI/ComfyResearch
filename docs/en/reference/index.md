---
doc_type: overview
doc_status: stable-core
---

:::{div} cr-eyebrow
Reference
:::

# Runtime and file contracts

:::{div} cr-article-lead
Exact commands, routes, document versions, generated artifacts, and support
boundaries for the current development source.
:::

Use these pages to look up a contract. For task sequences, return to the
[User Guide](../user-guide/index.md). The interactive FastAPI schema at
`/docs` lists every development endpoint; this reference concentrates on the
supported stable workflow.

::::{grid} 1 1 2 2
:gutter: 3
:class-container: cr-link-grid

:::{grid-item-card} Application
:link: application
:link-type: doc
:class-card: cr-link-card
CLI options, environment variables, remote precedence, and local state paths.
:::

:::{grid-item-card} Training API
:link: training-api
:link-type: doc
:class-card: cr-link-card
Stable workflow routes and newline-delimited training event contracts.
:::

:::{grid-item-card} Data contracts
:link: data-contracts
:link-type: doc
:class-card: cr-link-card
Graph, workspace, and saved-library document shapes and versions.
:::

:::{grid-item-card} Node contracts
:link: node-contracts
:link-type: doc
:class-card: cr-link-card
Definition sources, generated outputs, and validation commands.
:::

:::{grid-item-card} Support status
:link: support-status
:link-type: doc
:class-card: cr-link-card
Stable documentation scope, experimental areas, and development-version policy.
:::

::::

```{toctree}
:hidden:

Application <application>
Training API <training-api>
Data contracts <data-contracts>
Node contracts <node-contracts>
Support status <support-status>
```
