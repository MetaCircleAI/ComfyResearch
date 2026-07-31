# End-to-end tests (Playwright)

The backend (`comfy_research.main:app`) serves both the JSON API and the built
SPA from `frontend/dist` on one origin, so E2E only needs the backend running.

## Run locally

```bash
cd frontend
npm run build                         # produces frontend/dist
npm i -D @playwright/test             # if not already installed
npx playwright install chromium
npx playwright test -c e2e/playwright.config.ts
```

The Playwright `webServer` config starts `uvicorn ... :8000` from the repo root
and waits for it before running the specs.

## What's covered today (`smoke.spec.ts`)

1. **Backend HTTP smoke** — `GET /api/node-categories` returns the catalog with
   all optimizer nodes (`adam/adamw/sgd/shampoo/soap`). Exercises the real app.
2. **SPA shell smoke** — the built frontend loads, mounts into `#root`, and
   raises no uncaught page errors.

## TODO — full drag → train flow (currently non-blocking in CI)

The CI `e2e` job runs with `continue-on-error: true`. To make it a real gate:

1. Add stable `data-testid` attributes to the sidebar palette items and to the
   canvas nodes (React Flow nodes), plus the Train button and the loss chart.
2. Implement a spec that: drags `linear_dataset` + `mlp_model` + `adam_optimizer`
   + `trainer` onto the canvas, connects the four handles into the trainer,
   clicks **Train**, and asserts a loss series/point renders.
3. Once it passes in CI, remove `continue-on-error` from the `e2e` job in
   `.github/workflows/ci.yml`.
