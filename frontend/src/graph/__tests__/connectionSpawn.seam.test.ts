import type { Connection, Edge, Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { GENERATED_NODE_SPECS } from "../../generated/generatedNodeSpecs";
import { applyCanvasConnection } from "../connectionRules";
import { spawnConfigFor } from "../observableVizTrainerSpawn";

/** Issue #137 缝测试:onConnect(applyCanvasConnection)必须经由 spawnConfigFor
 * 取 spawn 配置。直接查询旧手写表会让 generated observable 失去 auto-spawn；
 * spawnConfigFor 与 onConnect 各自已有测试，此处覆盖两者的连接点。 */

const N = (id: string, type: string, data: Record<string, unknown> = {}): Node =>
  ({ id, type, position: { x: 0, y: 0 }, data }) as unknown as Node;

function connectObservable(nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  return applyCanvasConnection(
    { source: "o", target: "t", sourceHandle: "observables", targetHandle: "observables" } as Connection,
    nodes,
    edges,
  );
}

function spawnedViz(res: { nodes: Node[]; edges: Edge[] }): Node | undefined {
  return res.nodes.find((n) => n.type === "observable_viz" && n.id !== "o" && n.id !== "t");
}

describe("observable auto-spawn on canvas connect (issue #137)", () => {
  it("observable_accuracy → trainer spawns paired observable_viz + results edge", () => {
    const res = connectObservable([N("o", "observable_accuracy"), N("t", "trainer")], []);
    const viz = spawnedViz(res);
    expect(viz).toBeDefined();
    const d = (viz!.data ?? {}) as Record<string, unknown>;
    expect(d.pairedObservableId).toBe("o");
    expect(d.pairedTrainerId).toBe("t");
    expect(d.vizVariant).toBe("accuracy");
    expect(
      res.edges.some(
        (e) =>
          e.source === "t" &&
          e.target === viz!.id &&
          e.sourceHandle === "observable_results" &&
          e.targetHandle === "tensor",
      ),
    ).toBe(true);
  });

  it("kan_reg spawns via the generated channel (hand table deleted)", () => {
    const res = connectObservable([N("o", "kan_reg"), N("t", "trainer")], []);
    const viz = spawnedViz(res);
    expect(viz).toBeDefined();
    expect(((viz!.data ?? {}) as Record<string, unknown>).vizVariant).toBe("kan_reg");
  });

  it("observable_attention_map → trainer spawns one paired heatmap viz", () => {
    const first = connectObservable([N("o", "observable_attention_map"), N("t", "trainer")], []);
    const viz = spawnedViz(first);
    expect(viz).toBeDefined();
    expect(((viz!.data ?? {}) as Record<string, unknown>).vizVariant).toBe("attention_map");
    expect(first.edges.some((e) => e.source === "t" && e.target === viz!.id && e.sourceHandle === "observable_results")).toBe(true);
    const again = connectObservable(first.nodes, first.edges);
    expect(again.nodes.length).toBe(first.nodes.length);
    expect(again.edges.length).toBe(first.edges.length);
  });

  it("reconnecting does not duplicate the spawned viz", () => {
    const first = connectObservable([N("o", "observable_accuracy"), N("t", "trainer")], []);
    const again = connectObservable(first.nodes, first.edges);
    expect(again.nodes.length).toBe(first.nodes.length);
    expect(again.edges.length).toBe(first.edges.length);
  });

  it("every generated observable with a spawn block auto-spawns on connect", () => {
    const spawnTypes = Object.entries(GENERATED_NODE_SPECS)
      .filter(([, s]) => s.observable?.spawnsVizNode && s.spawn)
      .map(([type]) => type);
    expect(spawnTypes.length).toBeGreaterThan(0);
    const missed = spawnTypes.filter((type) => {
      expect(spawnConfigFor(type)).toBeDefined();
      const res = connectObservable([N("o", type), N("t", "trainer")], []);
      return spawnedViz(res) === undefined;
    });
    expect(missed).toEqual([]);
  });
});
