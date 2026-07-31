import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

describe("connection golden generator script", () => {
  it("explicitly invokes the bundled writer instead of relying on source-module main detection", () => {
    const pkg = JSON.parse(readFileSync(resolve(process.cwd(), "package.json"), "utf8"));
    expect(pkg.scripts["generate:connection-golden"]).toContain("writeGolden");
  });
});
