import type { Edge, Node } from "@xyflow/react";
import { layoutResearchGraphNodes } from "../src/graph/graphAutoLayout";

function node(id: string, type: string, x: number, y: number, extra: Partial<Node> = {}): Node {
  return {
    id,
    type,
    position: { x, y },
    data: {},
    ...extra,
  } as Node;
}

function edge(source: string, target: string): Edge {
  return { id: `${source}-${target}`, source, target } as Edge;
}

function pos(nodes: Node[], id: string): { x: number; y: number } {
  const n = nodes.find((item) => item.id === id);
  if (!n) throw new Error(`Missing node ${id}`);
  return n.position;
}

function assert(cond: unknown, message: string) {
  if (!cond) throw new Error(message);
}

function readNumeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function nodeSize(n: Node): { width: number; height: number } {
  const raw = n as Node & {
    measured?: { width?: number; height?: number };
    width?: number;
    height?: number;
  };
  const style = (n.style ?? {}) as Record<string, unknown>;
  return {
    width:
      readNumeric(raw.measured?.width) ??
      readNumeric(raw.width) ??
      readNumeric(style.width) ??
      readNumeric(style.minWidth) ??
      320,
    height:
      readNumeric(raw.measured?.height) ??
      readNumeric(raw.height) ??
      readNumeric(style.height) ??
      readNumeric(style.minHeight) ??
      180,
  };
}

function rectsOverlap(a: Node, b: Node): boolean {
  const { width: aw, height: ah } = nodeSize(a);
  const { width: bw, height: bh } = nodeSize(b);
  return (
    a.position.x < b.position.x + bw &&
    a.position.x + aw > b.position.x &&
    a.position.y < b.position.y + bh &&
    a.position.y + ah > b.position.y
  );
}

function assertNoOverlaps(nodes: Node[]) {
  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      assert(!rectsOverlap(nodes[i]!, nodes[j]!), `Nodes overlap after layout: ${nodes[i]!.id} / ${nodes[j]!.id}`);
    }
  }
}

function assertPackedRows(nodes: Node[], ids: string[], message: string) {
  const sorted = ids
    .map((id) => ({ id, ...pos(nodes, id) }))
    .sort((a, b) => a.y - b.y || a.x - b.x || a.id.localeCompare(b.id));
  let rowY = sorted[0]!.y;
  let lastX = -Infinity;
  for (const item of sorted) {
    if (item.y !== rowY) {
      assert(item.y > rowY, `${message}: row y positions should increase`);
      rowY = item.y;
      lastX = -Infinity;
    }
    assert(item.x > lastX, `${message}: nodes should increase left-to-right within each row`);
    lastX = item.x;
  }
}

function seededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function pick<T>(items: readonly T[], rng: () => number): T {
  return items[Math.floor(rng() * items.length)]!;
}

{
  const nodes = [
    node("data", "linear_dataset", 40, 40, { style: { width: 260, height: 210 } }),
    node("model", "mlp_model", 40, 160, { style: { width: 260, height: 180 } }),
    node("loss", "mse_loss", 400, 320, { style: { width: 300, height: 150 } }),
    node("opt", "sgd_optimizer", 40, 320, { style: { width: 260, height: 140 } }),
    node("trainer", "trainer", 720, 160, { style: { width: 340, height: 230 } }),
    node("training-viz", "training_visualization", 1020, 160, { style: { width: 360, height: 260 } }),
    node("obs", "observable_train_test_gap", 660, 480, { style: { width: 300, height: 120 } }),
    node("obs-viz", "observable_viz", 1020, 480, {
      style: { width: 360, height: 260 },
      data: { pairedObservableId: "obs", pairedTrainerId: "trainer" },
    }),
    node("endpoint-gap", "series_endpoint_gap", 1280, 480, { style: { width: 320, height: 130 } }),
    node("tensor-viz", "tensor_viz_0d", 1540, 480, { style: { width: 320, height: 180 } }),
    node("sweep", "sweep_data_table", 1800, 480, { style: { width: 560, height: 220 } }),
    node("table", "table_viz", 2060, 480, { style: { width: 480, height: 160 } }),
  ];
  const edges = [
    edge("data", "trainer"),
    edge("model", "trainer"),
    edge("loss", "trainer"),
    edge("opt", "trainer"),
    edge("trainer", "training-viz"),
    edge("obs", "trainer"),
    edge("trainer", "obs-viz"),
    edge("obs-viz", "endpoint-gap"),
    edge("endpoint-gap", "tensor-viz"),
    edge("tensor-viz", "sweep"),
    edge("sweep", "table"),
  ];
  const result = layoutResearchGraphNodes(nodes, edges);
  assert(result.changed, "Expected S2.4-style graph to move nodes");
  const out = result.nodes;
  const chain = ["obs", "obs-viz", "endpoint-gap", "tensor-viz", "sweep", "table"];
  for (let i = 1; i < chain.length; i += 1) {
    assert(pos(out, chain[i - 1]!).x < pos(out, chain[i]!).x, "S2.4 analysis chain should flow left-to-right");
    assert(pos(out, chain[i]!).y === pos(out, "obs").y, "S2.4 analysis chain should stay on one row");
  }
  assert(pos(out, "endpoint-gap").y > pos(out, "trainer").y, "Series endpoint gap must stay below the main trainer flow");
  assert(pos(out, "training-viz").y > pos(out, "obs").y, "Loose training viz should sit on a separate row below the observable chain");
  assertNoOverlaps(out);
}

{
  const rng = seededRng(0x5_24_2026);
  const nodes: Node[] = [
    node("data", "linear_dataset", 800, 20, { style: { width: 260, height: 210 } }),
    node("model", "mlp_model", 120, 540, { style: { width: 300, height: 190 } }),
    node("loss", "mse_loss", 1600, 40, { style: { width: 300, height: 150 } }),
    node("opt", "adamw_optimizer", -100, 460, { style: { width: 280, height: 140 } }),
    node("trainer", "trainer", 600, 380, { style: { width: 360, height: 240 } }),
    node("training-viz", "training_visualization", 480, 900, { style: { width: 380, height: 260 } }),
  ];
  const edges: Edge[] = [
    edge("data", "trainer"),
    edge("model", "trainer"),
    edge("loss", "trainer"),
    edge("opt", "trainer"),
    edge("trainer", "training-viz"),
  ];
  const commentLinks: Array<{ source: string; comment: string }> = [];
  const observableTypes = [
    "observable_accuracy",
    "observable_gradient_norm",
    "observable_train_test_gap",
    "observable_weight_l2",
  ] as const;
  const attachmentSpecs = [
    { type: "series_endpoint_gap", width: 320, height: 130 },
    { type: "tensor_selector", width: 320, height: 150 },
    { type: "smoothing_curve", width: 320, height: 140 },
    { type: "derivative_curve", width: 320, height: 140 },
    { type: "curve_annotator", width: 360, height: 230 },
    { type: "tensor_viz_0d", width: 320, height: 180 },
    { type: "tensor_viz_1d", width: 360, height: 230 },
    { type: "sweep_data_table", width: 560, height: 220 },
    { type: "table_viz", width: 480, height: 170 },
  ] as const;

  const chains: string[][] = [];
  for (let i = 0; i < 20; i += 1) {
    const obsId = `stress-obs-${i}`;
    const vizId = `stress-obs-viz-${i}`;
    const obsType = observableTypes[i % observableTypes.length]!;
    nodes.push(
      node(obsId, obsType, Math.floor(rng() * 2000) - 400, Math.floor(rng() * 1000), {
        style: { width: 280, height: 120 },
      }),
    );
    nodes.push(
      node(vizId, "observable_viz", Math.floor(rng() * 2000) - 400, Math.floor(rng() * 1000), {
        style: { width: 380, height: 250 },
        data: { pairedObservableId: obsId, pairedTrainerId: "trainer" },
      }),
    );
    edges.push(edge(obsId, "trainer"), edge("trainer", vizId));

    const chain = [obsId, vizId];
    let upstream = vizId;
    const attachmentCount = 1 + Math.floor(rng() * 3);
    for (let j = 0; j < attachmentCount; j += 1) {
      const spec = pick(attachmentSpecs, rng);
      const id = `stress-attach-${i}-${j}`;
      nodes.push(
        node(id, spec.type, Math.floor(rng() * 2000) - 400, Math.floor(rng() * 1200), {
          style: { width: spec.width, height: spec.height },
        }),
      );
      edges.push(edge(upstream, id));
      chain.push(id);
      upstream = id;
    }
    chains.push(chain);

    if (i % 5 === 0) {
      const commentId = `stress-comment-${i}`;
      nodes.push(
        node(commentId, "comment", Math.floor(rng() * 2000) - 400, Math.floor(rng() * 1200), {
          style: { width: 340, height: 150 },
        }),
      );
      edges.push({
        id: `${chain[chain.length - 1]}-${commentId}`,
        source: chain[chain.length - 1]!,
        target: commentId,
        sourceHandle: "comment",
        targetHandle: "comment",
      } as Edge);
      commentLinks.push({ source: chain[chain.length - 1]!, comment: commentId });
    }
  }

  const trainingVizChain = ["training-viz"];
  let upstream = "training-viz";
  for (const [i, spec] of [
    { type: "tensor_selector", width: 320, height: 150 },
    { type: "smoothing_curve", width: 320, height: 140 },
    { type: "curve_annotator", width: 360, height: 230 },
  ].entries()) {
    const id = `training-viz-attach-${i}`;
    nodes.push(node(id, spec.type, 2000 + i * 120, 800 - i * 140, { style: { width: spec.width, height: spec.height } }));
    edges.push(edge(upstream, id));
    trainingVizChain.push(id);
    upstream = id;
  }
  chains.push(trainingVizChain);

  const result = layoutResearchGraphNodes(nodes, edges);
  assert(result.changed, "Expected observable stress graph to move nodes");
  const out = result.nodes;
  const trainerY = pos(out, "trainer").y;
  for (const chain of chains) {
    const rowY = pos(out, chain[0]!).y;
    assert(rowY > trainerY, `Chain ${chain[0]} should be below the main trainer flow`);
    for (let i = 1; i < chain.length; i += 1) {
      assert(pos(out, chain[i - 1]!).x < pos(out, chain[i]!).x, `Chain ${chain[0]} should flow left-to-right`);
      assert(pos(out, chain[i]!).y === rowY, `Chain ${chain[0]} should stay on one row`);
    }
  }
  for (const link of commentLinks) {
    assert(pos(out, link.comment).x > pos(out, link.source).x, `Comment ${link.comment} should be right of its source`);
    assert(pos(out, link.comment).y >= pos(out, link.source).y, `Comment ${link.comment} should align at or below its source row`);
  }
  assertNoOverlaps(out);
}

{
  const observableStyle = { style: { width: 240, height: 120 } };
  const vizStyle = { style: { width: 360, height: 220 } };
  const nodes = [
    node("trainer", "trainer", 200, 20),
    node("obs-accuracy", "observable_accuracy", 40, 40, observableStyle),
    node("obs-weight-l2", "observable_weight_l2", 80, 70, observableStyle),
    node("obs-grad", "observable_gradient_norm", 120, 100, observableStyle),
    node("obs-gap", "observable_train_test_gap", 160, 130, observableStyle),
    node("obs-noise", "observable_singular_noise", 200, 160, observableStyle),
    node("model", "mlp_model", 900, 500),
    node("data", "modular_addition_dataset", 800, 0),
    node("opt", "adamw_optimizer", -100, 500),
    node("loss", "cross_entropy_loss", 30, 500),
    node("viz-loss", "training_visualization", 220, 500, vizStyle),
    node("viz-accuracy", "observable_viz", 260, 540, vizStyle),
  ];
  const edges = [
    edge("data", "trainer"),
    edge("model", "trainer"),
    edge("opt", "trainer"),
    edge("loss", "trainer"),
    edge("obs-accuracy", "trainer"),
    edge("obs-weight-l2", "trainer"),
    edge("obs-grad", "trainer"),
    edge("obs-gap", "trainer"),
    edge("obs-noise", "trainer"),
    edge("trainer", "viz-loss"),
    edge("trainer", "viz-accuracy"),
  ];
  const result = layoutResearchGraphNodes(nodes, edges);
  assert(result.changed, "Expected scrambled trainer graph to move nodes");

  const out = result.nodes;
  assert(pos(out, "data").x < pos(out, "model").x, "Dataset should be left of model");
  assert(pos(out, "model").x < pos(out, "trainer").x, "Model should be left of trainer");
  assert(pos(out, "opt").x < pos(out, "trainer").x, "Optimizer should be left of trainer");
  assert(pos(out, "loss").x < pos(out, "trainer").x, "Loss should be left of trainer");

  const obsIds = ["obs-accuracy", "obs-weight-l2", "obs-grad", "obs-gap", "obs-noise"];
  const obsY = pos(out, obsIds[0]!).y;
  for (let i = 0; i < obsIds.length; i += 1) {
    assert(pos(out, obsIds[i]!).y === obsY, "Observable nodes should fill the first bottom row");
    assert(pos(out, obsIds[i]!).y > pos(out, "trainer").y, "Observable nodes should move below main flow");
    if (i > 0) {
      assert(
        pos(out, obsIds[i - 1]!).x < pos(out, obsIds[i]!).x,
        "Observable nodes should be laid out left-to-right",
      );
    }
  }

  assert(pos(out, "viz-loss").y > obsY, "Training visualization should start below observable rows");
  assert(pos(out, "viz-accuracy").y === pos(out, "viz-loss").y, "Visualization nodes should share a packed row");
  assert(pos(out, "viz-loss").x < pos(out, "viz-accuracy").x, "Visualization nodes should pack left-to-right");
  assertNoOverlaps(out);

  const expandedNodes = [
    ...out,
    node("extra-data", "random_noise_dataset", pos(out, "trainer").x + 25, pos(out, "trainer").y + 25),
    node("extra-model", "transformer_token_model", pos(out, "model").x + 35, pos(out, "model").y + 35),
    node("extra-opt", "sgd_optimizer", pos(out, "opt").x + 45, pos(out, "opt").y + 45),
    node("extra-loss", "mse_loss", pos(out, "loss").x + 55, pos(out, "loss").y + 55),
    node("extra-tensor", "tensor_concat", pos(out, "trainer").x - 45, pos(out, "trainer").y + 5),
    node("extra-obs-a", "observable_accuracy", pos(out, "obs-accuracy").x + 20, pos(out, "obs-accuracy").y + 20, observableStyle),
    node("extra-obs-b", "observable_train_test_gap", pos(out, "obs-gap").x + 30, pos(out, "obs-gap").y + 30, observableStyle),
    node("extra-viz-a", "observable_viz", pos(out, "viz-loss").x + 30, pos(out, "viz-loss").y + 30, vizStyle),
    node("extra-viz-b", "tensor_viz_1d", pos(out, "viz-accuracy").x + 40, pos(out, "viz-accuracy").y + 40, vizStyle),
    node("extra-note", "comment", pos(out, "trainer").x, pos(out, "trainer").y, { style: { width: 340, height: 130 } }),
  ];
  const expandedEdges = [
    ...edges,
    edge("extra-data", "trainer"),
    edge("extra-model", "trainer"),
    edge("extra-opt", "trainer"),
    edge("extra-loss", "trainer"),
    edge("extra-tensor", "trainer"),
    edge("extra-obs-a", "trainer"),
    edge("extra-obs-b", "trainer"),
    edge("trainer", "extra-viz-a"),
    edge("trainer", "extra-viz-b"),
  ];
  const second = layoutResearchGraphNodes(expandedNodes, expandedEdges);
  assert(second.changed, "Expected second layout pass to move newly added nodes");
  const secondOut = second.nodes;
  assert(pos(secondOut, "extra-data").x < pos(secondOut, "trainer").x, "New dataset should join main flow left of trainer");
  assert(pos(secondOut, "extra-note").x > pos(secondOut, "trainer").x, "New note should move into the right notes lane");
  assertPackedRows(
    secondOut,
    [...obsIds, "extra-obs-a", "extra-obs-b"],
    "Observable rows after adding nodes and re-layout",
  );
  assert(pos(secondOut, "extra-viz-a").y > pos(secondOut, "extra-obs-a").y, "New viz nodes should stay below observables");
  assertNoOverlaps(secondOut);
}

{
  const nodes = [
    node("a", "tensor_add", 0, 0),
    node("b", "tensor_concat", 0, 0),
    node("note", "comment", 0, 0),
    node("child", "linear_layer", 1, 2, { parentId: "group" }),
    node("hidden", "mlp_model", 3, 4, { hidden: true }),
  ];
  const edges = [edge("a", "b"), edge("b", "a")];
  const result = layoutResearchGraphNodes(nodes, edges);
  assert(result.changed, "Expected cyclic fallback graph to move visible top-level nodes");
  const out = result.nodes;
  assert(Number.isFinite(pos(out, "a").x) && Number.isFinite(pos(out, "b").y), "Cycle fallback positions must be finite");
  assert(pos(out, "child").x === 1 && pos(out, "child").y === 2, "Parented child position should be preserved");
  assert(pos(out, "hidden").x === 3 && pos(out, "hidden").y === 4, "Hidden node position should be preserved");
  assert(pos(out, "note").x > pos(out, "a").x, "Notes should move into a right-side lane");
}

console.log("OK: graph auto-layout verification passed.");
