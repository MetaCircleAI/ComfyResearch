# AutoDL remote GPU (local UI + remote train)

This guide matches the flow in `README.md` (Chinese section **如何使用 GPU**).

- Run ComfyResearch locally (`python app.py`).
- In the Trainer node, set **Compute device** to `autodl gpu` and fill the **AutoDL remote GPU** panel.
- Click **Train** — no separate Bootstrap / Connect / Test buttons in the UI.

## 1) Prepare the AutoDL instance

1. Start an instance on [AutoDL](https://www.autodl.com/market/list) and copy the SSH login command and password.
2. You do **not** need to clone this repo or install ComfyResearch on the GPU machine manually.
3. Use a PyTorch image with CUDA (see README for a tested stack).

## 2) Configure Trainer (auto-saved)

In **AutoDL remote GPU**, fill:

| Field | Example |
|-------|---------|
| SSH command | `ssh -p 12033 root@region-42.seetacloud.com` |
| Password | from AutoDL console |
| Remote path | `/root/ComfyResearch` (directory created on first train) |
| Python | `/root/miniconda3/bin/python3` (default on many AutoDL images) |
| SSH key path | optional if using key auth instead of password |

Settings are **auto-saved** (~420ms debounce) to a local file when `autodl gpu` is selected:

- Path: `<repo>/.comfyresearch/remote_train_config.json`
- Git-ignored (per-machine user data, not part of the repo)
- Includes SSH host/user/path, password, `enabled: true`

They are **not** stored in the workspace graph JSON or `frontend/dist`.

## 3) Train

1. Set **Compute device** → `autodl gpu`.
2. Click **Train**.

Each remote **Train** automatically:

1. **Bootstraps** the remote runtime (`bootstrap_remote_train_environment` in `POST /api/train`):
   - Compare local bundle digest to remote `.comfyresearch/bundle.sha256`; **skip tar sync** when unchanged
   - Otherwise tar-sync local `comfy_research/` + `requirements.txt` to **Remote path**
   - Run `pip install -r requirements.txt` only when `requirements.txt` changed since last bootstrap
   - Probe remote Python and `import comfy_research`
2. **Validates** the graph (`remote_train_cli --validate-only`)
3. **Streams** training NDJSON over SSH (`remote_train_cli`)

The Trainer banner shows **Remote: bootstrapping runtime…** during step 1.

## 4) Pause / abort (SSH proxy)

Pause and abort use cooperative control files on the AutoDL instance (see `remote-gpu-lambda.md`). Because code sync runs on every remote Train, local `comfy_research` changes (including control modules) reach the remote host on the next Train without manual steps.

## 5) Notes

- AutoDL instances must already be running; the app does not start/stop them.
- Password auth on macOS may need `sshpass` (`brew install hudochenkov/sshpass/sshpass`).
- Env vars `COMFYRESEARCH_TRAIN_REMOTE_*` still work as a fallback.
- API endpoints `/api/train/remote/bootstrap` and `/api/train/remote/validate` exist for tooling but are **not** wired to Trainer buttons.
