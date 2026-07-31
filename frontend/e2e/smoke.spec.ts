import { expect, test } from "@playwright/test";

// --- Backend-through-HTTP smoke: the server boots and serves the node catalog,
// including the four optimizer nodes. This exercises the real FastAPI app
// end-to-end (routing + catalog assembly), no browser interaction needed.
test("GET /api/node-categories serves the catalog with all optimizer nodes", async ({ request }) => {
  const res = await request.get("/api/node-categories");
  expect(res.ok(), `status ${res.status()}`).toBeTruthy();
  const body = (await res.json()) as {
    categories: Array<{ id: string; children: Array<{ id: string; label: string }> }>;
  };
  const optimizer = body.categories.find((c) => c.id === "optimizer");
  expect(optimizer, "optimizer category present").toBeTruthy();
  const ids = optimizer!.children.map((c) => c.id);
  for (const opt of ["adam_optimizer", "adamw_optimizer", "sgd_optimizer", "shampoo_optimizer", "soap_optimizer"]) {
    expect(ids, `optimizer palette contains ${opt}`).toContain(opt);
  }
});

// --- SPA shell smoke: the built frontend is served and mounts without a blank
// screen or a hard crash (catches broken bundles / missing static assets).
test("SPA shell loads and mounts into #root", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/");
  await expect(page.locator("#root")).toBeVisible();
  // React should have rendered *something* into #root.
  await expect(page.locator("#root")).not.toBeEmpty();
  expect(errors, `uncaught page errors: ${errors.join("; ")}`).toHaveLength(0);
});

// TODO(e2e, full flow): drag dataset + model + optimizer + trainer from the
// sidebar palette onto the canvas, connect their handles, click Train, and
// assert a loss series appears. This needs stable selectors — add `data-testid`
// to the palette items and canvas nodes first, then implement here. Once it's
// green in CI, remove `continue-on-error` from the `e2e` job in
//.github/workflows/ci.yml to make E2E blocking. See./README.md.
