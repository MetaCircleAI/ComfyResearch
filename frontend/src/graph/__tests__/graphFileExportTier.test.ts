import { describe, expect, it } from "vitest";
import { applyGraphFileExportTier } from "../graphFileExportTier";
import type { GraphDocument } from "../../types/graph";

describe("small graph export tier", () => {
  it("strips diffusion sampling and observable runtime assets", () => {
    const document = {
      version: 1,
      nodes: [
        { id: "sampler", type: "deterministic_diffusion_sampler", position: { x: 0, y: 0 }, data: { numSteps: 50, runId: "run", previewGrid: "image", metadata: { device: "cuda" }, device: "cuda" } },
        { id: "paired", type: "observable_paired_generation_similarity", position: { x: 0, y: 0 }, data: { meanMae: 0.1, mae: [0.1], histogramPng: "histogram", imageGrid: "grid" } },
        { id: "rp", type: "observable_rp_score_sscd", position: { x: 0, y: 0 }, data: { rp: 1, meanSimilarity: 0.99, similarities: [0.99], histogramPng: "histogram" } },
        { id: "nearest", type: "observable_nearest_train_gl", position: { x: 0, y: 0 }, data: { glScore: 0.4, nearestSimilarity: [0.9], nearestIndex: [1], histogramPng: "histogram", imageGrid: "grid", backend: "pixel_cosine_exact" } },
        { id: "lmc", type: "observable_linear_interpolation_barrier", position: { x: 0, y: 0 }, data: { alphaMin: 0, alphaMax: 1, alphaSteps: 21, alphaSeries: [0, 1], trainLossAlongPath: [1, 1], testLossAlongPath: [1, 1], lossBarrier: 0, accuracyDrop: 0, interpolationCurvePng: "curve", runSummary: "done" } },
        { id: "bezier", type: "observable_bezier_mode_connectivity", position: { x: 0, y: 0 }, data: { alphaSteps: 21, alphaSeries: [0, 1], linearTestLoss: [2, 1], bezierTestLoss: [1, 1], linearLossBarrier: 1, bezierLossBarrier: 0, runSummary: "done" } },
      ],
      edges: [],
      viewport: { x: 0, y: 0, zoom: 1 },
    } as unknown as GraphDocument;

    const small = applyGraphFileExportTier(document, "small");
    const dataById = Object.fromEntries(small.nodes.map((node) => [node.id, node.data as Record<string, unknown>]));

    expect(dataById.sampler).toEqual({ numSteps: 50 });
    expect(dataById.paired).toEqual({});
    expect(dataById.rp).toEqual({});
    expect(dataById.nearest).toEqual({});
    expect(dataById.lmc).toEqual({ alphaMin: 0, alphaMax: 1, alphaSteps: 21 });
    expect(dataById.bezier).toEqual({ alphaSteps: 21 });
  });
});
