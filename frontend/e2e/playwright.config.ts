import { defineConfig } from "@playwright/test";

// The backend (comfy_research.main:app) serves BOTH the JSON API and the built
// SPA from frontend/dist on a single origin — so E2E only needs to start the
// backend. CI runs `npm run build` first to produce frontend/dist.
const PORT = Number(process.env.E2E_PORT ?? 8000);
const BASE = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: ".",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  retries: process.env.CI ? 1 : 0,
  reporter: [["html", { outputFolder: "playwright-report", open: "never" }], ["list"]],
  use: {
    baseURL: BASE,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: `python -m uvicorn comfy_research.main:app --host 127.0.0.1 --port ${PORT}`,
    cwd: "../..", // repo root, relative to frontend/e2e/
    url: BASE,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
