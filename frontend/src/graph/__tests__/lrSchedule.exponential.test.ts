import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { defaultLrScheduleData } from "../../components/nodes/lrScheduleDefaults";
import { GENERATED_NODE_SPECS } from "../../generated/generatedNodeSpecs";

describe("exponential_epoch LR schedule", () => {
  it("keeps hand defaults aligned with the generated node and codegen contract", () => {
    const defaults = defaultLrScheduleData();
    const spec = GENERATED_NODE_SPECS.lr_schedule;
    expect(defaults).toMatchObject({
      lrSchedule: "constant",
      exponentialDecayFactor: 0.95,
      exponentialDecayEpochs: 1,
    });
    expect(spec.frontend.codegenKey).toBe("lr_schedule");
    expect(spec.defaults).toMatchObject({
      exponentialDecayFactor: 0.95,
      exponentialDecayEpochs: 1,
    });
    expect(spec.fields.find((field) => field.key === "lrSchedule")?.options).toContain("exponential_epoch");
  });

  it("exposes the factor and epoch interval controls only for exponential_epoch", () => {
    const source = readFileSync(resolve(__dirname, "../../components/nodes/LrScheduleNode.tsx"), "utf8");
    expect(source).toContain('id: "exponential_epoch"');
    expect(source).toContain('d.lrSchedule === "exponential_epoch"');
    expect(source).toContain('label="exponential decay factor"');
    expect(source).toContain('label="decay every epochs"');
  });
});
