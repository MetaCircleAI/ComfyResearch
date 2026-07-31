import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("train execution graph boundary", () => {
  it("routes every browser training entry point through the shared target closure", () => {
    const entrypoints = [
      "../../components/nodes/TrainerNode.tsx",
      "../../components/nodes/CrlTrainerNode.tsx",
    ];

    for (const entrypoint of entrypoints) {
      const source = readFileSync(resolve(__dirname, entrypoint), "utf8");
      expect(source, entrypoint).toContain("serializeExecutionGraphForTarget(");
    }
  });
});
