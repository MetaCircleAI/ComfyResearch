---
doc_type: tutorial
doc_status: stable-core
---

:::{div} cr-eyebrow
Get Started
:::

# Install and run

:::{div} cr-article-lead
Build Comfy Research from source, verify the local service, and then complete a
small CPU experiment.
:::

The repository currently provides a source installation rather than a prebuilt
package. A first run needs Python, Node.js, and Git:

| Tool | Requirement | Used for |
| --- | --- | --- |
| Python | 3.10 or newer; docs CI uses 3.11 | FastAPI backend and research runtime |
| Node.js | Current LTS; Node 20 or newer recommended | React/Vite frontend build |
| Git | Any maintained version | Source checkout and updates |

CPU is enough for the tutorial. CUDA, Apple MPS, and remote GPU execution can
be configured after the local path works.

## Build from source

Clone the repository and enter it:

```bash
git clone https://github.com/MetaCircleAI/ComfyResearch.git
cd ComfyResearch
```

Create an isolated Python environment and install the backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the pinned frontend dependencies and build the static application:

```bash
npm --prefix frontend ci
npm --prefix frontend run build
```

The frontend build also checks the generated Node metadata. If it reports a
mismatch, regenerate the checked-in Node artifacts before rebuilding.

## Start the application

From the repository root, bind the first local run to the loopback interface:

```bash
python app.py --host 127.0.0.1 --port 8042 --open
```

If the browser does not open, visit `http://127.0.0.1:8042/`. Keep this terminal
running while you use the application.

In a second terminal, verify the backend:

```bash
curl http://127.0.0.1:8042/api/health
```

A successful response contains `"ok": true`. Fix startup or dependency errors
before creating a graph; the tutorial assumes this health check passes.

::::{grid} 1 1 2 2
:gutter: 3
:class-container: cr-link-grid

:::{grid-item-card} Run your first graph
:link: first-graph
:link-type: doc
:class-card: cr-link-card
Load the bundled Edge of Stability (CPU) graph, run 80 CPU steps, and make one controlled change.
:::

:::{grid-item-card} Development mode
:link: development-mode
:link-type: doc
:class-card: cr-link-card
Run the backend with the Vite frontend and return to a verified static build.
:::

::::

## If startup fails

**The backend reports that `frontend/dist` is stale**
: Run `npm --prefix frontend run build`, then restart the server.

**Port 8042 is already in use**
: Choose another port, such as 8842, in both the start command and health URL.

**A compute device is unavailable**
: Use CPU for the first run. Configure accelerators only after the local smoke
  test succeeds.

For additional symptoms, see [Troubleshooting](../user-guide/troubleshooting.md).

```{toctree}
:hidden:

Run your first graph <first-graph>
Development mode <development-mode>
```
