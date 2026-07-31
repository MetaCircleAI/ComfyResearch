import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Edge, Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { serializeExecutionGraphForTarget } from "../trainSeriesPlan";

describe("parametric path sampler protocol", () => {
  it("documents the paper BatchNorm evaluation behavior", () => {
    const source = readFileSync(
      resolve(__dirname, "../../components/nodes/ParametricPathSamplerNode.tsx"),
      "utf8",
    );

    expect(source).toContain("per-batch BatchNorm statistics");
  });

  it("sends only the sampler dependency closure", () => {
    const checkpoint = "x".repeat(1024);
    const nodes = [
      { id: "dataset", type: "cifar10_dataset", data: {} },
      { id: "model", type: "vgg11_cifar_model", data: {} },
      { id: "loss", type: "cross_entropy_loss", data: {} },
      { id: "trainer-sb", type: "trainer", data: { lossHistory: [1, 2, 3] } },
      { id: "trainer-lb", type: "trainer", data: { lossHistory: [1, 2, 3] } },
      {
        id: "checkpoint-sb",
        type: "model_checkpoint",
        data: { checkpoint_b64: checkpoint, memoryCheckpoint_b64: checkpoint },
      },
      {
        id: "checkpoint-lb",
        type: "model_checkpoint",
        data: { checkpoint_b64: checkpoint, memoryCheckpoint_b64: checkpoint },
      },
      { id: "sampler", type: "parametric_path_sampler", data: {} },
      { id: "viz", type: "curve_series_visualization", data: {} },
    ].map((node, index) => ({ ...node, position: { x: index, y: 0 } })) as Node[];
    const edges = [
      { id: "dataset-sb", source: "dataset", target: "trainer-sb", targetHandle: "dataset" },
      { id: "model-sb", source: "model", target: "trainer-sb", targetHandle: "model" },
      { id: "loss-sb", source: "loss", target: "trainer-sb", targetHandle: "loss" },
      { id: "sb-checkpoint", source: "trainer-sb", target: "checkpoint-sb" },
      { id: "dataset-lb", source: "dataset", target: "trainer-lb", targetHandle: "dataset" },
      { id: "model-lb", source: "model", target: "trainer-lb", targetHandle: "model" },
      { id: "loss-lb", source: "loss", target: "trainer-lb", targetHandle: "loss" },
      { id: "lb-checkpoint", source: "trainer-lb", target: "checkpoint-lb" },
      { id: "sb-sampler", source: "checkpoint-sb", target: "sampler", targetHandle: "checkpoint_sb" },
      { id: "lb-sampler", source: "checkpoint-lb", target: "sampler", targetHandle: "checkpoint_lb" },
      { id: "model-sampler", source: "model", target: "sampler", targetHandle: "model" },
      { id: "dataset-sampler", source: "dataset", target: "sampler", targetHandle: "dataset" },
      { id: "loss-sampler", source: "loss", target: "sampler", targetHandle: "loss" },
      { id: "sampler-viz", source: "sampler", target: "viz" },
    ] as Edge[];

    const graph = serializeExecutionGraphForTarget(nodes, edges, "sampler");

    expect(graph.nodes.map((node) => node.id)).toEqual([
      "dataset", "model", "loss", "checkpoint-sb", "checkpoint-lb", "sampler",
    ]);
    expect(graph.nodes.map((node) => node.id)).not.toContain("trainer-sb");
    expect(graph.nodes.map((node) => node.id)).not.toContain("trainer-lb");
    expect(graph.nodes.map((node) => node.id)).not.toContain("viz");
    for (const id of ["checkpoint-sb", "checkpoint-lb"]) {
      const data = graph.nodes.find((node) => node.id === id)!.data;
      expect(data.checkpoint_b64).toBe(checkpoint);
      expect(data).not.toHaveProperty("memoryCheckpoint_b64");
    }
  });
});
