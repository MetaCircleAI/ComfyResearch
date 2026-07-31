import type { Edge, Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import {
  applyAssignmentsToNodes,
  getSweptAxisIdSet,
  planTrainSeriesAssignments,
  serializeExecutionGraphForTarget,
  trainSeriesAxisKey,
} from "../trainSeriesPlan";

const NODES = [
  { id: "ds", type: "linear_dataset", data: { inputDim: 8, outputDim: 1, trainSize: 64, seed: 0 } },
  {
    id: "m",
    type: "mlp_model",
    data: { inputDim: 8, outputDim: 1, depth: 1, width: [8, 16], outputScale: [0.01, 1], activation: "relu", seed: 0 },
  },
  { id: "o", type: "sgd_optimizer", data: { learningRate: 0.01, momentum: 0, weightDecay: 0 } },
  { id: "l", type: "mse_loss", data: {} },
  { id: "t", type: "trainer", data: { trainingSteps: 4, logFrequency: 1, batchSize: -1 } },
] as unknown as Node[];
const EDGES = [
  { id: "e1", source: "ds", target: "t", sourceHandle: "dataset", targetHandle: "dataset" },
  { id: "e2", source: "m", target: "t", sourceHandle: "model", targetHandle: "model" },
  { id: "e3", source: "o", target: "t", sourceHandle: "optimizer", targetHandle: "optimizer" },
  { id: "e4", source: "l", target: "t", sourceHandle: "loss", targetHandle: "loss" },
] as unknown as Edge[];

describe("collectAxes seam (empty-set regression)", () => {
  it("legacy sweep axes are untouched by the generated-first lookup", () => {
    const swept = getSweptAxisIdSet(NODES, EDGES, "t");
    expect(swept.has(trainSeriesAxisKey("m", "width"))).toBe(true);
    expect(swept.has(trainSeriesAxisKey("m", "outputScale"))).toBe(true);
    expect(swept.size).toBe(2);
  });

  it("cyclic_lr_schedule wired to the optimizer lr_schedule handle produces sweep axes", () => {
    // 与 lr_schedule 同一收集位:optimizer 的 lr_schedule handle 上的 generic
    // generated-spec 轴(cyclic 的 lrMin 多值 → 出轴)。
    const nodes = [
      ...NODES,
      { id: "clr", type: "cyclic_lr_schedule", data: { lrMin: [0.001, 0.002], lrMax: 0.005 } },
    ] as unknown as Node[];
    const edges = [
      ...EDGES,
      { id: "e5", source: "clr", target: "o", sourceHandle: "lr_schedule", targetHandle: "lr_schedule" },
    ] as unknown as Edge[];
    const swept = getSweptAxisIdSet(nodes, edges, "t");
    expect(swept.has(trainSeriesAxisKey("clr", "lrMin"))).toBe(true);
    expect(swept.size).toBe(3);
  });

  it("keeps attention relation layer/head lists inside one multi-curve run", () => {
    const nodes = [
      ...NODES,
      { id: "score", type: "observable_attention_relation_score", data: { layerIndex: [1, 0], headIndex: [0, 1] } },
    ] as unknown as Node[];
    const edges = [
      ...EDGES,
      { id: "e5", source: "score", target: "t", sourceHandle: "observables", targetHandle: "observables" },
    ] as unknown as Edge[];
    const swept = getSweptAxisIdSet(nodes, edges, "t");
    expect(swept.has(trainSeriesAxisKey("score", "layerIndex"))).toBe(false);
    expect(swept.has(trainSeriesAxisKey("score", "headIndex"))).toBe(false);
    expect(planTrainSeriesAssignments(nodes, edges, "t")).toHaveLength(4);
  });
  it("runs the VGG-11 paper template once per model seed", () => {
    const nodes = NODES.map((node) => node.id === "m"
      ? { ...node, type: "vgg11_cifar_model", data: { seed: [0, 1, 2, 3, 4] } }
      : node) as Node[];
    const swept = getSweptAxisIdSet(nodes, EDGES, "t");
    expect(swept.has(trainSeriesAxisKey("m", "seed"))).toBe(true);
    expect(planTrainSeriesAssignments(nodes, EDGES, "t")).toHaveLength(5);
  });
});

describe("trainer request graph serialization", () => {
  const checkpoint = "x".repeat(1024);
  const nodes = [
    { id: "ds1", type: "linear_dataset", data: {} },
    { id: "m1", type: "mlp_model", data: {} },
    { id: "o1", type: "sgd_optimizer", data: {} },
    { id: "l1", type: "mse_loss", data: {} },
    { id: "t1", type: "trainer", data: {} },
    {
      id: "ck1",
      type: "model_checkpoint",
      data: { checkpoint_b64: checkpoint, memoryCheckpoint_b64: checkpoint, checkpointSource: "memory" },
    },
    { id: "viz1", type: "training_visualization", data: { lossHistory: [1, 2, 3] } },
    { id: "ds2", type: "linear_dataset", data: {} },
    { id: "combined", type: "combined_model", data: {} },
    { id: "layer1", type: "linear_layer", parentId: "combined", data: {} },
    { id: "layer2", type: "activation_layer", parentId: "combined", data: {} },
    { id: "o2", type: "sgd_optimizer", data: {} },
    { id: "schedule", type: "cyclic_lr_schedule", data: {} },
    { id: "l2", type: "mse_loss", data: {} },
    {
      id: "needed-ck",
      type: "model_checkpoint",
      data: { checkpoint_b64: checkpoint, memoryCheckpoint_b64: checkpoint, checkpointSource: "memory" },
    },
    { id: "obs", type: "observable_user", data: {} },
    {
      id: "t2",
      type: "trainer",
      data: {
        trainingSteps: [2, 4],
        memoryCheckpoint_b64: checkpoint,
        lossHistory: [1, 2],
        epochTicks: [0, 1],
        targetCurveLossHistory: [3, 2, 1],
      },
    },
    { id: "viz2", type: "training_visualization", data: {} },
  ].map((node, index) => ({ ...node, position: { x: index * 10, y: 0 } })) as unknown as Node[];
  const edges = [
    { id: "ds1-t1", source: "ds1", target: "t1", targetHandle: "dataset" },
    { id: "m1-t1", source: "m1", target: "t1", targetHandle: "model" },
    { id: "o1-t1", source: "o1", target: "t1", targetHandle: "optimizer" },
    { id: "l1-t1", source: "l1", target: "t1", targetHandle: "loss" },
    { id: "t1-ck1", source: "t1", target: "ck1", targetHandle: "model_checkpoint" },
    { id: "t1-needed-ck", source: "t1", target: "needed-ck", targetHandle: "model_checkpoint" },
    { id: "t1-viz1", source: "t1", target: "viz1", targetHandle: "tensor_list" },
    { id: "layer-chain", source: "layer1", target: "layer2", targetHandle: "tensor_in" },
    { id: "ds2-t2", source: "ds2", target: "t2", targetHandle: "dataset" },
    { id: "combined-t2", source: "combined", target: "t2", targetHandle: "model" },
    { id: "schedule-o2", source: "schedule", target: "o2", targetHandle: "lr_schedule" },
    { id: "o2-t2", source: "o2", target: "t2", targetHandle: "optimizer" },
    { id: "l2-t2", source: "l2", target: "t2", targetHandle: "loss" },
    { id: "needed-ck-obs", source: "needed-ck", target: "obs", targetHandle: "model" },
    { id: "obs-t2", source: "obs", target: "t2", targetHandle: "observables" },
    { id: "t2-viz2", source: "t2", target: "viz2", targetHandle: "tensor_list" },
  ] as unknown as Edge[];

  it("keeps the target trainer dependency closure and excludes unrelated trainers and outputs", () => {
    const graph = serializeExecutionGraphForTarget(nodes, edges, "t2");
    expect(graph.nodes.map((n) => n.id)).toEqual([
      "ds2", "combined", "layer1", "layer2", "o2", "schedule", "l2", "needed-ck", "obs", "t2",
    ]);
    expect(graph.edges.map((e) => e.id)).toEqual([
      "layer-chain", "ds2-t2", "combined-t2", "schedule-o2", "o2-t2", "l2-t2", "needed-ck-obs", "obs-t2",
    ]);
  });

  it("keeps a required checkpoint once under the backend canonical field", () => {
    const graph = serializeExecutionGraphForTarget(nodes, edges, "t2");
    const data = graph.nodes.find((n) => n.id === "needed-ck")!.data;
    expect(data.checkpoint_b64).toBe(checkpoint);
    expect(data).not.toHaveProperty("memoryCheckpoint_b64");
    const trainerData = graph.nodes.find((n) => n.id === "t2")!.data;
    expect(trainerData).not.toHaveProperty("memoryCheckpoint_b64");
    expect(trainerData).not.toHaveProperty("lossHistory");
    expect(trainerData).not.toHaveProperty("epochTicks");
    expect(trainerData).not.toHaveProperty("targetCurveLossHistory");
  });

  it("keeps distinct file and memory checkpoints because they have different UI semantics", () => {
    const changed = nodes.map((node) =>
      node.id === "needed-ck"
        ? {
            ...node,
            data: {
              checkpoint_b64: "file-checkpoint",
              memoryCheckpoint_b64: "memory-checkpoint",
              checkpointSource: "file",
            },
          }
        : node,
    ) as Node[];
    const data = serializeExecutionGraphForTarget(changed, edges, "t2").nodes.find(
      (node) => node.id === "needed-ck",
    )!.data;
    expect(data.checkpoint_b64).toBe("file-checkpoint");
    expect(data.memoryCheckpoint_b64).toBe("memory-checkpoint");
  });

  it("uses the same closure for manual Train, Train Series, and resume", () => {
    const manual = serializeExecutionGraphForTarget(nodes, edges, "t2");
    const seriesNodes = applyAssignmentsToNodes(manual.nodes, [
      { nodeId: "t2", key: "trainingSteps", value: 4 },
    ]);
    const resume = serializeExecutionGraphForTarget(nodes, edges, "t2");
    const ids = manual.nodes.map((n) => n.id);
    expect(seriesNodes.map((n) => n.id)).toEqual(ids);
    expect(resume.nodes.map((n) => n.id)).toEqual(ids);
    expect(manual.edges).toEqual(resume.edges);
    expect(ids).not.toContain("ck1");
  });
});
