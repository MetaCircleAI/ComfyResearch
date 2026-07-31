import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("remote dataset upload seam", () => {
  it("exposes an opt-in checkbox in the AutoDL panel", () => {
    const source = readFileSync(
      resolve(__dirname, "../../components/nodes/TrainerNode.tsx"),
      "utf8",
    );

    expect(source).toContain('aria-label="Upload local dataset"');
    expect(source).toContain("checked={Boolean(remoteCfg.upload_dataset)}");
  });
});
