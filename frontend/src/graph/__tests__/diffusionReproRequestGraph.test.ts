import { describe, expect, it } from "vitest";
import { diffusionReproRequestGraph } from "../diffusionReproRequestGraph";

describe("diffusion reproducibility request graph", () => {
  const checkpoint = "checkpoint-payload";
  const nodes = [
    { id: "dataset", type: "cifar10_dataset", position: { x: 0, y: 0 }, data: {} },
    { id: "model-a", type: "unet_ddpm_model", position: { x: 0, y: 0 }, data: {} },
    { id: "optimizer", type: "adam_optimizer", position: { x: 0, y: 0 }, data: {} },
    { id: "trainer-a", type: "trainer", position: { x: 0, y: 0 }, data: { memoryCheckpoint_b64: checkpoint } },
    { id: "checkpoint-a", type: "model_checkpoint", position: { x: 0, y: 0 }, data: { checkpoint_b64: checkpoint, memoryCheckpoint_b64: checkpoint, checkpointSource: "memory" } },
    { id: "sampler-a", type: "deterministic_diffusion_sampler", position: { x: 0, y: 0 }, data: { runId: "run-a" } },
    { id: "checkpoint-b", type: "model_checkpoint", position: { x: 0, y: 0 }, data: { checkpoint_b64: "other", memoryCheckpoint_b64: "other", checkpointSource: "memory" } },
    { id: "sampler-b", type: "deterministic_diffusion_sampler", position: { x: 0, y: 0 }, data: { runId: "run-b" } },
    { id: "paired", type: "observable_paired_generation_similarity", position: { x: 0, y: 0 }, data: {} },
  ] as any[];
  const edges = [
    { id: "dataset-trainer", source: "dataset", target: "trainer-a", sourceHandle: "dataset", targetHandle: "dataset" },
    { id: "model-trainer", source: "model-a", target: "trainer-a", sourceHandle: "model", targetHandle: "model" },
    { id: "optimizer-trainer", source: "optimizer", target: "trainer-a", sourceHandle: "optimizer", targetHandle: "optimizer" },
    { id: "trainer-checkpoint", source: "trainer-a", target: "checkpoint-a", sourceHandle: "checkpoint", targetHandle: "model_checkpoint" },
    { id: "checkpoint-sampler", source: "checkpoint-a", target: "sampler-a", sourceHandle: "model", targetHandle: "checkpoint" },
    { id: "a-paired", source: "sampler-a", target: "paired", sourceHandle: "samples", targetHandle: "sampler_a" },
    { id: "b-paired", source: "sampler-b", target: "paired", sourceHandle: "samples", targetHandle: "sampler_b" },
  ] as any[];

  it("sends one active checkpoint for sampling instead of every canvas checkpoint", () => {
    const graph = diffusionReproRequestGraph(nodes, edges, "sampler-a", "sampler");
    const ids = graph.nodes.map((node) => node.id);
    const checkpointNode = graph.nodes.find((node) => node.id === "checkpoint-a")!;

    expect(ids).toContain("checkpoint-a");
    expect(ids).not.toContain("checkpoint-b");
    expect(checkpointNode.data).toMatchObject({ checkpoint_b64: checkpoint, checkpointSource: "file" });
    expect(checkpointNode.data).not.toHaveProperty("memoryCheckpoint_b64");
    expect(graph.nodes.find((node) => node.id === "trainer-a")!.data).not.toHaveProperty("memoryCheckpoint_b64");
  });

  it("does not resend checkpoints when an observable consumes saved sampler runs", () => {
    const graph = diffusionReproRequestGraph(nodes, edges, "paired", "observable");

    expect(graph.nodes.map((node) => node.id).sort()).toEqual(["paired", "sampler-a", "sampler-b"]);
    expect(graph.nodes.some((node) => node.type === "model_checkpoint")).toBe(false);
  });
});
