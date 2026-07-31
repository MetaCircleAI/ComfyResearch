---
doc_type: how-to
doc_status: stable-core
---

:::{div} cr-eyebrow
Get Started
:::

# Use development mode

:::{div} cr-article-lead
Run the Python backend and Vite frontend separately while changing the web
application, then verify the static production path before finishing.
:::

## Start both development servers

Activate the Python environment and start the backend in one terminal:

```bash
python app.py --host 127.0.0.1 --port 8042
```

Start Vite in a second terminal and tell it which backend port to proxy:

```bash
COMFYRESEARCH_PORT=8042 npm --prefix frontend run dev
```

Open `http://127.0.0.1:5173/`. Requests under `/api/` are proxied to the backend
on port 8042. A Vite page without a running backend can render the shell while
application data and training actions fail, so verify `/api/health` first.

## Choose the right loop

| Workflow | Frontend | Use it for |
| --- | --- | --- |
| Development | Vite on port 5173 | Fast feedback and hot module replacement |
| Production-like | Static `frontend/dist` served by `app.py` | Final integration and packaging checks |

Backend-only changes do not require Vite. Frontend changes should be checked in
both loops because proxy behavior can hide a stale or missing static build.

## Return to the static application

Stop Vite, rebuild the checked-in frontend sources, and restart the backend:

```bash
npm --prefix frontend run build
python app.py --host 127.0.0.1 --port 8042 --open
```

Confirm that the page loads from `http://127.0.0.1:8042/` and that
`http://127.0.0.1:8042/api/health` still reports `"ok": true`.

:::{note}
Binding to `127.0.0.1` keeps the development service on the local machine. Do
not expose it to a network until authentication, credentials, and stored
research artifacts have been reviewed for that environment.
:::
