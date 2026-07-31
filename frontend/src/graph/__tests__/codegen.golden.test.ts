import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildCodegenGoldenManifest,
  CODEGEN_GOLDEN_PATH,
  CODEGEN_GOLDEN_VERSION,
} from "../../../scripts/generate-codegen-golden";

// The repository currently ships the default graph plus 41 library templates.
// Keep this as a floor so an accidental template-discovery regression cannot
// silently produce a much smaller golden manifest.
const MINIMUM_CODEGEN_FIXTURES = 40;

describe("codegen golden snapshots", () => {
  it("matches the committed default graph and template codegen hashes", () => {
    const current = buildCodegenGoldenManifest();
    const snapshot = JSON.parse(readFileSync(resolve(process.cwd(), CODEGEN_GOLDEN_PATH), "utf8"));

    expect(current.version).toBe(CODEGEN_GOLDEN_VERSION);
    expect(Object.keys(current.fixtures).length).toBeGreaterThanOrEqual(MINIMUM_CODEGEN_FIXTURES);
    expect(current).toEqual(snapshot);
  });
});
