/** Resolved-hint contracts: definitions are the single source of hint text. */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { computeHintGolden, HINT_GOLDEN_PATH } from "../../../scripts/generate-hint-golden";
import { GENERATED_NODE_SPECS } from "../../generated/generatedNodeSpecs";
import { nodeRegistryHint } from "../nodeRegistrySpec";

describe("hint golden", () => {
  it("resolved hints match the committed snapshot byte-for-byte", () => {
    const committed = JSON.parse(readFileSync(HINT_GOLDEN_PATH, "utf8"));
    expect(computeHintGolden()).toEqual(committed);
  });
});

describe("hint single-source", () => {
  it("every hinted generated spec resolves through the registry", () => {
    for (const [t, g] of Object.entries(GENERATED_NODE_SPECS)) {
      if (g.hint != null) expect(nodeRegistryHint(t)).toBe(g.hint);
    }
  });
});
