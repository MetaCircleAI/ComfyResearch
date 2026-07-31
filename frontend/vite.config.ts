/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import { configDefaults } from "vitest/config";
import react from "@vitejs/plugin-react";

/** Must match the ComfyResearch API (e.g. `python app.py --port 8042` → use 8042 here). */
const apiTarget =
  process.env.COMFYRESEARCH_API_ORIGIN ??
  `http://127.0.0.1:${process.env.COMFYRESEARCH_PORT ?? "8000"}`;

export default defineConfig({
  plugins: [react()],
  base: "/",
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
      "/lmech": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  // Vitest reads this config. Playwright E2E specs under e2e/ have their own
  // runner (@playwright/test) and must not be collected by vitest.
  test: {
    exclude: [...configDefaults.exclude, "e2e/**"],
  },
});
