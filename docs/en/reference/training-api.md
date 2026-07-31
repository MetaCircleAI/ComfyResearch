---
doc_type: reference
doc_status: stable-core
---

:::{div} cr-eyebrow
Reference
:::

# Training API

The FastAPI development schema is available at `/docs`. The routes below are
the supported stable workflow surface documented here.

## Service and catalog routes

| Method and path | Response or effect |
| --- | --- |
| `GET /api/health` | Health, effective local or remote mode, remote source, and last validation state |
| `GET /api/node-definitions` | Version-1 generated node manifest |
| `GET /api/node-categories` | Version-1 library categories plus dynamic user entries |
| `GET /api/workspace` | Load the version-3 workspace snapshot |
| `POST /api/workspace` | Validate and replace the workspace snapshot |
| `GET /api/graph-library/{kind}` | List `workflows` and `templates` |
| `POST /api/graph-library/{kind}` | Add or replace one validated saved-graph entry |
| `DELETE /api/graph-library/{kind}/{entry_id}` | Delete an entry and return the remaining list |

## Single-run routes

| Method and path | Contract |
| --- | --- |
| `POST /api/train` | Validate a `TrainRequest` and stream local or selected remote training |
| `POST /api/train/control` | Request `pause` or `abort` for a Trainer ID |
| `GET /api/train/cuda-devices` | List detected local CUDA devices |
| `GET /api/train/remote/status` | Report effective remote mode and last validation state |
| `GET /api/train/remote/config` | Return the stored remote configuration |
| `POST /api/train/remote/config` | Replace the stored remote configuration |
| `POST /api/train/remote/validate` | Validate a supplied or effective remote connection |
| `POST /api/train/remote/bootstrap` | Synchronize and verify the remote runtime outside a train request |

`TrainRequest` contains `trainer_node_id`, `nodes`, `edges`, optional `resume`
state, and optional `hessian_oversized_policy` (`skip` or `force`). Nodes and
edges are validated using the graph schema before the run is prepared.

## NDJSON response

`POST /api/train` returns `application/x-ndjson`. Each line is a complete JSON
object. Clients must process the stream incrementally and must not parse the
response as one JSON document.

| Event type | Meaning and stable fields |
| --- | --- |
| `progress` | Run position with `step` and `total` |
| `phase` | Remote bootstrap or execution phase with `phase` and `message` |
| `complete` | Terminal success with checkpoint, loss histories, ticks, visualization targets, and Observable updates |
| `paused` | Terminal pause for this stream with `next_step` plus resumable checkpoint and histories |
| `aborted` | Terminal cooperative abort |
| `error` | Terminal remote-stream error with `detail` |

Non-finite numeric values are converted to JSON `null` before encoding so each
line remains valid RFC 8259 JSON.

For a normal local run the documented sequence begins with one or more
`progress` events and ends in `complete`, `paused`, or `aborted`. Remote runs
can emit `phase` before training and can terminate with `error` if the SSH
process fails without another terminal event.

## Sweep routes

| Method and path | Contract |
| --- | --- |
| `POST /api/train/sweep` | Stream `sweep_started`, `sweep_progress`, `sweep_row`, then `sweep_complete` or `sweep_aborted` |
| `POST /api/train/sweep/control` | Request sweep abort between single-run iterations |
| `POST /api/train/coordinate-descent` | Stream coordinate-descent tuning events |
| `POST /api/train/coordinate-descent/control` | Request coordinate-descent abort by session ID |

These route contracts do not define a scientific acceptance threshold. The
client or experiment protocol must decide what result counts as success.
