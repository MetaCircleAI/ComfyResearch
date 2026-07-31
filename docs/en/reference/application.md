---
doc_type: reference
doc_status: stable-core
---

:::{div} cr-eyebrow
Reference
:::

# Application

## Server commands

| Command | Effect |
| --- | --- |
| `python app.py` | Serve the built frontend and FastAPI backend using CLI or environment defaults. |
| `python app.py --host 127.0.0.1 --port 8042 --open` | Recommended local documentation command. |
| `python app.py --reload` | Enable Uvicorn source reload. |
| `COMFYRESEARCH_PORT=8042 npm --prefix frontend run dev` | Run Vite and proxy `/api` to backend port 8042. |
| `npm --prefix frontend run build` | Build the static frontend and run generated-contract checks. |

## CLI and environment defaults

CLI arguments override their corresponding environment values.

| CLI option | Environment variable | Source default |
| --- | --- | --- |
| `--host` | `COMFYRESEARCH_HOST` | `0.0.0.0` |
| `--port` | `COMFYRESEARCH_PORT` | `8000` |
| `--reload` / `--no-reload` | `COMFYRESEARCH_RELOAD` | Off |
| `--open` / `--no-open` | `COMFYRESEARCH_OPEN_BROWSER` | Off |

The source host default exposes the service on available interfaces. Use an
explicit `--host 127.0.0.1` for normal local work unless network access has
been deliberately reviewed.

Boolean environment values accept the forms implemented by `app.py`:
`COMFYRESEARCH_RELOAD` accepts `1`, `true`, or `yes` after lowercasing;
`COMFYRESEARCH_OPEN_BROWSER` accepts the same values.

## Remote training environment variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `COMFYRESEARCH_TRAIN_REMOTE_HOST` | SSH host; a non-empty value enables the environment candidate | Empty |
| `COMFYRESEARCH_TRAIN_REMOTE_USER` | SSH user | `ubuntu` |
| `COMFYRESEARCH_TRAIN_REMOTE_PATH` | Repository path on the remote host | Empty |
| `COMFYRESEARCH_TRAIN_REMOTE_PYTHON` | Remote Python executable | `/root/miniconda3/bin/python3` |
| `COMFYRESEARCH_TRAIN_REMOTE_IDENTITY` | SSH identity-file path | Empty |
| `COMFYRESEARCH_TRAIN_REMOTE_PASSWORD` | SSH password | Empty |
| `COMFYRESEARCH_TRAIN_REMOTE_EXTRA_OPTS` | Additional SSH options | Empty |
| `COMFYRESEARCH_TRAIN_REMOTE_UPLOAD_DATASET` | Include local dataset content when `1`, `true`, `yes`, or `on` | Off |

An active stored config takes precedence over environment variables. It is
active only when `enabled`, `host`, and `remote_path` are set. If no stored
config is active, an active environment config is used; otherwise execution is
local.

## Local state paths

Paths are relative to the repository root unless stated otherwise.

| Path | Content |
| --- | --- |
| `data/workspace.json` | Version-3 projects, graph documents, and viewports |
| `data/graph_library/workflows.json` | Combined-model subgraph entries |
| `data/graph_library/templates/` | Canonical one-file-per-ID Template entries |
| `data/user_observables.json` | User-defined Observable records |
| `data/user_linear_datasets.json` | User-defined linear dataset records |
| `data/user_symbolic_func_datasets.json` | User-defined symbolic dataset records |
| `.comfyresearch/remote_train_config.json` | Stored remote settings, including any entered password as plain JSON |
| `.comfyresearch/artifacts/` | Externalized local runtime artifacts |

Most machine-local paths are ignored by Git, while canonical files under
`data/graph_library/templates/` may be committed. Always inspect the exact file
and repository status before sharing.
