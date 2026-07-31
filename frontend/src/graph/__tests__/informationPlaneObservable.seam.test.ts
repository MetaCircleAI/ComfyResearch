import { describe, expect, it } from "vitest";
import {
  informationPlaneLimits,
  informationPlaneProgress,
  informationPlaneTicks,
} from "../../components/nodes/InformationPlaneVizNode";
import { GENERATED_NODE_SPECS } from "../../generated/generatedNodeSpecs";
import { NODE_REGISTRY } from "../nodeRegistry";
import { getObservableVizVariant } from "../observableVizVariant";

describe("information-plane observable seam", () => {
  it("exposes bounded sampling defaults and its dedicated component", () => {
    const spec = GENERATED_NODE_SPECS.observable_information_plane;
    expect(spec.defaults).toEqual({
      bins: 30,
      maxSamples: 512,
      includeOutput: true,
      binning: "uniform_intervals",
      outputMapping: "tanh",
    });
    expect(spec.hint).toContain("maxSamples");
    expect(spec.observable?.vizVariant).toBe("information_plane");
    expect(NODE_REGISTRY.observable_information_plane.component).toBeTypeOf("function");
  });

  it("recognizes the information-plane visualization variant", () => {
    expect(getObservableVizVariant({ type: "observable_viz", data: { vizVariant: "information_plane" } } as never)).toBe("information_plane");
  });

  it("scales I(X;T) and I(T;Y) independently", () => {
    expect(informationPlaneLimits([
      [[12, 0.999], [4.5, 0.94], [1.2, 0.9]],
      [[11.8, 0.998], [3.1, 0.96], [1.1, 0.93]],
    ])).toEqual({ x: 12, y: 1 });

    expect(informationPlaneLimits([[[15.2, 1.24]]])).toEqual({ x: 16, y: 1.3 });
  });

  it("provides scientific axis ticks and colors by actual training progress", () => {
    expect(informationPlaneTicks(12, 6)).toEqual([0, 2, 4, 6, 8, 10, 12]);
    expect(informationPlaneTicks(1, 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(informationPlaneProgress(1, 3, [0, 10, 100])).toBe(0.1);
    expect(informationPlaneProgress(1, 3)).toBe(0.5);
  });
});
