from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from comfy_research.api import (
    activation_tensor,
    collect,
    curve_lpd,
    dataset_tensor,
    diffusion_repro,
    vision_dataset_gallery,
    graph_library,
    health,
    library_source,
    kan_plot,
    notebook_execute,
    node_catalog,
    parametric_path_sampler,
    pca,
    predict,
    svd,
    train,
    user_linear_datasets,
    user_symbolic_func_datasets,
    user_observables,
    weight_tensors,
    workspace,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND = _REPO_ROOT / "frontend"
_DIST = _FRONTEND / "dist"

_HTML_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "X-ComfyResearch-App": "1",
}


def _assert_frontend_dist_current(frontend: Path = _FRONTEND) -> None:
    index = frontend / "dist" / "index.html"
    src = frontend / "src"
    if not index.is_file() or not src.is_dir():
        return

    inputs = [path for path in src.rglob("*") if path.is_file()]
    for name in ("index.html", "package.json", "package-lock.json", "vite.config.ts"):
        path = frontend / name
        if path.is_file():
            inputs.append(path)
    if not inputs:
        return

    newest = max(inputs, key=lambda path: path.stat().st_mtime_ns)
    if newest.stat().st_mtime_ns <= index.stat().st_mtime_ns:
        return
    raise RuntimeError(
        f"frontend/dist is stale: {newest.relative_to(frontend)} is newer than dist/index.html. "
        "Run 'cd frontend && npm run build' before starting ComfyResearch."
    )


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _assert_frontend_dist_current()
    added = user_observables.ensure_grokking_demo_user_observables()
    if added:
        print(f"[ComfyResearch] Seeded {added} grokking demo user observable(s) from bundled data.")
    idx = _DIST / "index.html"
    print(f"[ComfyResearch] SPA index: {idx} (exists={idx.is_file()})")
    print(
        "[ComfyResearch] Expect browser tab title 'ComfyResearch' and no .vue files in DevTools. "
        "If you see GraphView.vue, [ComfyUI], or /api/userdata 404s, another app or a cached page is on this origin—try another port (python app.py --port 8842) or hard-refresh (Cmd+Shift+R).",
    )
    print(
        "[ComfyResearch] This server serves the built SPA from frontend/dist (not Vite HMR). "
        "After editing frontend/src/, run: cd frontend && npm run build — then reload the tab (Cmd+Shift+R). "
        "For live dev, run COMFYRESEARCH_PORT=<this port> npm run dev in frontend/ and open http://127.0.0.1:5173.",
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="ComfyResearch", version="0.1.0", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(curve_lpd.api_router)
    app.include_router(library_source.router)
    app.include_router(notebook_execute.router)
    app.include_router(workspace.router)
    app.include_router(graph_library.router)
    app.include_router(node_catalog.router)
    app.include_router(user_observables.router)
    app.include_router(user_linear_datasets.router)
    app.include_router(user_symbolic_func_datasets.router)
    app.include_router(train.router)
    app.include_router(collect.router)
    app.include_router(parametric_path_sampler.router)
    app.include_router(activation_tensor.router)
    app.include_router(dataset_tensor.router)
    app.include_router(diffusion_repro.router)
    app.include_router(vision_dataset_gallery.router)
    app.include_router(pca.router)
    app.include_router(predict.router)
    app.include_router(svd.router)
    app.include_router(weight_tensors.router)
    app.include_router(kan_plot.router)

    if _DIST.is_dir() and (_DIST / "index.html").is_file():
        assets_dir = _DIST / "assets"
        if assets_dir.is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=assets_dir),
                name="assets",
            )

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str) -> FileResponse:
            if full_path.startswith("api"):
                from fastapi import HTTPException

                raise HTTPException(status_code=404, detail="Not found")
            return FileResponse(_DIST / "index.html", headers=_HTML_NO_CACHE)

    return app


app = create_app()
