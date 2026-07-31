import type { Edge, Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import type { CurveSeriesTableNodeData } from "../../components/nodes/curveSeriesDefaults";
import { appendCurveSeriesOnTrainComplete } from "../curveSeriesAppend";
import { dualAxisWarranted, inferSeriesYAxis, logYForAxisSide } from "../sweepVizPlot";

/** 缝测试:curve-series 多源表 + 通用观测量捕获 + 双轴语义。
 * Fig-1 图形(单 trainer:training_visualization + observable_viz 双源一表)
 * 双 trainer 图形各配对一源，防止多源改动互相影响。 */

const N = (id: string, type: string, data: Record<string, unknown> = {}): Node =>
  ({ id, type, position: { x: 0, y: 0 }, data }) as unknown as Node;
const E = (id: string, source: string, target: string, sourceHandle: string, targetHandle: string): Edge =>
  ({ id, source, target, sourceHandle, targetHandle }) as Edge;

function runAppend(nodes: Node[], edges: Edge[], trainerId: string): Node[] {
  let out = nodes;
  const setNodes = (updater: (n: Node[]) => Node[]) => {
    out = updater(out);
  };
  appendCurveSeriesOnTrainComplete(setNodes, nodes, edges, trainerId);
  return out;
}

function tableRows(nodes: Node[], tableId: string) {
  const table = nodes.find((n) => n.id === tableId);
  return ((table?.data ?? {}) as CurveSeriesTableNodeData).rows ?? [];
}

/** Fig-1 确切图形:trainer → training_viz(loss)+ observable_viz(paired norm)→ 同一张表。 */
function fig1Graph() {
  const nodes = [
    N("t", "trainer", {}),
    N("obs", "observable_last_layer_weight_norm", {}),
    N("tv", "training_visualization", {
      lossHistory: [1, 0.1, 0.01],
      stepTicks: [1, 2, 3],
    }),
    N("ov", "observable_viz", {
      pairedObservableId: "obs",
      pairedTrainerId: "t",
      valueHistory: [0.07, 0.075, 0.078],
      stepTicks: [1, 2, 3],
    }),
    N("tbl", "curve_series_table", {
      rows: [],
      selectedSeriesIds: null,
      captureMetrics: ["train_loss", "observable"],
      paramKeyOrder: null,
    }),
  ];
  const edges = [
    E("e1", "t", "tv", "loss_results", "tensor_list"),
    E("e2", "t", "ov", "observable_results", "tensor"),
    E("e3", "obs", "t", "observables", "observables"),
    E("e4", "tv", "tbl", "series", "stream"),
    E("e5", "ov", "tbl", "series", "stream"),
  ];
  return { nodes, edges };
}

describe("curve-series multi-source capture (Fig-1 shape)", () => {
  it("captures train_loss AND the paired observable into the same table", () => {
    const { nodes, edges } = fig1Graph();
    const rows = tableRows(runAppend(nodes, edges, "t"), "tbl");
    const metricIds = rows.map((r) => r.metricId).sort();
    expect(metricIds).toEqual(["observable:observable_last_layer_weight_norm", "train_loss"]);
    const obsRow = rows.find((r) => r.metricId?.startsWith("observable:"))!;
    expect(obsRow.label.startsWith("Last layer weight norm")).toBe(true);
    expect(obsRow.y).toEqual([0.07, 0.075, 0.078]);
    const lossRow = rows.find((r) => r.metricId === "train_loss")!;
    expect(lossRow.y).toEqual([1, 0.1, 0.01]);
  });

  it("does not duplicate rows when append runs twice for the same run", () => {
    const { nodes, edges } = fig1Graph();
    const once = runAppend(nodes, edges, "t");
    const twice = runAppend(once, edges, "t");
    expect(tableRows(twice, "tbl").length).toBe(tableRows(once, "tbl").length);
  });
});

describe("curve-series same-type multi-observable", () => {
  it("two same-type observables land as distinct rows with distinct labels", () => {
    const nodes = [
      N("t", "trainer", {}),
      N("o1", "observable_last_layer_weight_norm", { instanceTitle: "Norm A" }),
      N("o2", "observable_last_layer_weight_norm", { instanceTitle: "Norm B" }),
      N("ov1", "observable_viz", {
        pairedObservableId: "o1", pairedTrainerId: "t",
        valueHistory: [1, 2], stepTicks: [1, 2],
      }),
      N("ov2", "observable_viz", {
        pairedObservableId: "o2", pairedTrainerId: "t",
        valueHistory: [3, 4], stepTicks: [1, 2],
      }),
      N("tbl", "curve_series_table", {
        rows: [], selectedSeriesIds: null,
        captureMetrics: ["observable"], paramKeyOrder: null,
      }),
    ];
    const edges = [
      E("e1", "t", "ov1", "observable_results", "tensor"),
      E("e2", "t", "ov2", "observable_results", "tensor"),
      E("e3", "ov1", "tbl", "series", "stream"),
      E("e4", "ov2", "tbl", "series", "stream"),
    ];
    const rows = tableRows(runAppend(nodes, edges, "t"), "tbl");
    expect(rows.length).toBe(2);
    expect(rows.every((r) => r.metricId === "observable:observable_last_layer_weight_norm")).toBe(true);
    const labels = rows.map((r) => r.label).sort();
    expect(labels[0]!.startsWith("Norm A")).toBe(true);
    expect(labels[1]!.startsWith("Norm B")).toBe(true);
    expect(rows.map((r) => r.y).sort()).toEqual([[1, 2], [3, 4]]);
  });
});

describe("curve-series dual-trainer isolation (shape regression)", () => {
  it("trainer A completion appends only A's paired source", () => {
    const nodes = [
      N("tA", "trainer", {}),
      N("tB", "trainer", {}),
      N("ovA", "observable_viz", {
        pairedObservableId: "oA", pairedTrainerId: "tA",
        valueHistory: [0.5, 0.6], stepTicks: [1, 2],
      }),
      N("ovB", "observable_viz", {
        pairedObservableId: "oB", pairedTrainerId: "tB",
        valueHistory: [0.1, 0.2], stepTicks: [1, 2],
      }),
      N("oA", "observable_accuracy", {}),
      N("oB", "observable_accuracy", {}),
      N("tbl", "curve_series_table", {
        rows: [], selectedSeriesIds: null,
        captureMetrics: ["train_acc"], paramKeyOrder: null,
      }),
    ];
    const edges = [
      E("e1", "tA", "ovA", "observable_results", "tensor"),
      E("e2", "tB", "ovB", "observable_results", "tensor"),
      E("e3", "ovA", "tbl", "series", "stream"),
      E("e4", "ovB", "tbl", "series", "stream"),
    ];
    const rows = tableRows(runAppend(nodes, edges, "tA"), "tbl");
    expect(rows.length).toBe(1);
    expect(rows[0]!.y).toEqual([0.5, 0.6]);
  });
});

describe("dual-axis semantics", () => {
  it("train_loss → left axis, observable:* → right axis, acc unchanged", () => {
    expect(inferSeriesYAxis("train loss", "train_loss")).toBe("left");
    expect(inferSeriesYAxis("train acc", "train_acc")).toBe("right");
    expect(
      inferSeriesYAxis("Last layer weight norm", "observable:observable_last_layer_weight_norm"),
    ).toBe("right");
  });

  it("loss + observable warrants dual axes", () => {
    const mk = (label: string, metricId: string) => ({
      id: label, label, metricId, color: "#000", points: [],
    });
    expect(
      dualAxisWarranted([
        mk("train loss", "train_loss"),
        mk("Last layer weight norm", "observable:observable_last_layer_weight_norm"),
      ]),
    ).toBe(true);
  });

  it("log-y applies to the left axis only under dual axes (Fig-1 form)", () => {
    expect(logYForAxisSide("left", true, true)).toBe(true);
    expect(logYForAxisSide("right", true, true)).toBe(false);
    expect(logYForAxisSide("right", false, true)).toBe(true);
    expect(logYForAxisSide("left", true, false)).toBe(false);
  });
});
