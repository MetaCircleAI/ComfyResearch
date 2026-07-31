# Remote GPUs on Lambda Cloud (and SSH)

ComfyResearch can use GPUs in three ways:

## 1. Run the whole app on the GPU instance (simplest)

Use this when the browser can reach the backend through an SSH tunnel.

1. Start an instance on [Lambda Cloud](https://lambda.ai/) and note its IP.
2. SSH in (replace user if your image uses another account):

   ```bash
   ssh ubuntu@<instance-ip>
   ```

3. Clone this repo on the instance, create a venv, install `requirements.txt`, and install a CUDA-enabled PyTorch build that matches the instance (see Lambda's docs for the recommended wheel).
4. On the instance:

   ```bash
   cd /path/to/ComfyResearch
   python app.py --host 127.0.0.1 --port 8042 --no-open
   ```

5. On your laptop, forward the port:

   ```bash
   ssh -L 8042:127.0.0.1:8042 ubuntu@<instance-ip>
   ```

6. Open `http://127.0.0.1:8042` locally. Training runs on the instance; use the trainer **Compute device** control (`auto` uses CUDA when available).

Pause and abort in the UI work normally because the API and training loop share one process.

## 2. Split UI (local) vs training (remote over SSH)

When the FastAPI server runs **locally** but you want **`POST /api/train`** executed on a remote machine (same repo checkout there), set:

| Variable | Meaning |
|----------|---------|
| `COMFYRESEARCH_TRAIN_REMOTE_HOST` | Hostname or IP (required to enable proxying) |
| `COMFYRESEARCH_TRAIN_REMOTE_USER` | SSH user (default `ubuntu`) |
| `COMFYRESEARCH_TRAIN_REMOTE_PATH` | Absolute path to this repo on the remote |
| `COMFYRESEARCH_TRAIN_REMOTE_PYTHON` | Interpreter on remote (default `python3`) |
| `COMFYRESEARCH_TRAIN_REMOTE_IDENTITY` | Optional path to private key (`ssh -i`) |
| `COMFYRESEARCH_TRAIN_REMOTE_EXTRA_OPTS` | Extra `ssh` arguments (quoted shell string, split with whitespace) |

Example:

```bash
export COMFYRESEARCH_TRAIN_REMOTE_HOST=192.0.2.10
export COMFYRESEARCH_TRAIN_REMOTE_PATH=/home/ubuntu/ComfyResearch
export COMFYRESEARCH_TRAIN_REMOTE_USER=ubuntu
python app.py --port 8042
```

Each train run performs **two** SSH sessions: a quick `--validate-only` pass (so invalid graphs return HTTP 400 before streaming starts), then the full NDJSON stream via `python -m comfy_research.remote_train_cli`.

For faster repeated connections, enable SSH multiplexing in `~/.ssh/config` (`ControlMaster`, `ControlPath`, `ControlPersist`) for that host.

### Pause / abort with remote training (SSH proxy)

When training is delegated over SSH, pause and abort use **control files on the remote host**:

1. `remote_train_cli` creates `/tmp/comfyresearch/sessions/<session_id>.control.json` and emits a `remote_session` NDJSON line.
2. The training loop polls that file each step (same cooperative semantics as local `/api/train/control`).
3. UI **Pause** / **Abort** → local `/api/train/control` → a second SSH call (`remote_train_control_cli`) writes `{"action":"pause"|"abort"}` into the control file.
4. Remote training yields `paused` or `aborted`, then exits cleanly.

Re-bootstrap happens automatically on the next remote **Train** (code sync in `POST /api/train`). If the full app runs on the GPU instance (Layer 1 + port forward), in-process control applies with no second SSH call.

## 3. Trainer compute device

On any machine with a GPU, set the trainer node's **Compute device** field:

- `auto` — CUDA if `torch.cuda.is_available()`, else Apple MPS if available, else CPU.
- `cpu`, `cuda`, `cuda:0`, …, `mps` — explicit choice (invalid combinations raise a clear error).

`CUDA_VISIBLE_DEVICES` on the server still restricts which GPU index `cuda` maps to.
