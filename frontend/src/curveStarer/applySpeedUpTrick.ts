import type { Edge, Node } from "@xyflow/react";
import { defaultWeightRegData } from "../components/nodes/weightRegDefaults";
import { appendResearchNode } from "../graph/nodeInstanceTitle";
import type { SpeedUpTrickProposal } from "./speedUpTrickTypes";

function cloneGraph(nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  return {
    nodes: nodes.map((n) => ({ ...n, data: { ...(n.data as object) } })),
    edges: edges.map((e) => ({ ...e })),
  };
}

function findTrainer(nodes: Node[]): Node | null {
  const trainers = nodes.filter((n) => n.type === "trainer" || n.type === "crl_trainer");
  return trainers.length === 1 ? trainers[0]! : trainers[0] ?? null;
}

function nextAuxId(prefix: string, nodes: Node[]): string {
  let i = 0;
  while (nodes.some((n) => n.id === `${prefix}-${i}`)) i += 1;
  return `${prefix}-${i}`;
}

/** Trick tests should not stack on canvas L1/L2 reg nodes — strength must come from the trick only. */
function stripExistingWeightRegNodes(nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  const drop = new Set(
    nodes.filter((n) => n.type === "l2_reg" || n.type === "l1_reg").map((n) => n.id),
  );
  if (drop.size === 0) return { nodes, edges };
  return {
    nodes: nodes.filter((n) => !drop.has(n.id)),
    edges: edges.filter((e) => !drop.has(e.source) && !drop.has(e.target)),
  };
}

export function applySpeedUpTrick(
  nodes: Node[],
  edges: Edge[],
  proposal: SpeedUpTrickProposal,
  trainerId?: string,
): { nodes: Node[]; edges: Edge[] } {
  const cloned = cloneGraph(nodes, edges);
  let outNodes = cloned.nodes.filter((n) => String(n.type) !== "graph_assist_failure_overlay");
  let outEdges = cloned.edges;
  const trainer = trainerId
    ? outNodes.find((n) => n.id === trainerId) ?? findTrainer(outNodes)
    : findTrainer(outNodes);
  if (!trainer) return { nodes: outNodes, edges: outEdges };

  const tid = trainer.id;
  const pos = { x: trainer.position.x - 180, y: trainer.position.y + 120 };

  if (proposal.trickKind === "grad_clip_shell") {
    outNodes = outNodes.map((n) =>
      n.id === tid
        ? {
            ...n,
            data: {
              ...(n.data as object),
              gradClipMaxNorm: proposal.params.shellRadius,
            },
          }
        : n,
    );
    return { nodes: outNodes, edges: outEdges };
  }

  if (proposal.trickKind === "l2_reg_shell") {
    ({ nodes: outNodes, edges: outEdges } = stripExistingWeightRegNodes(outNodes, outEdges));
    const id = nextAuxId("l2_reg-trick", outNodes);
    const node = appendResearchNode(
      outNodes,
      "l2_reg",
      pos,
      { ...defaultWeightRegData(), lossScale: proposal.params.lossScale ?? 1 } as Record<
        string,
        unknown
      >,
      id,
    );
    outNodes = [...outNodes, node];
    outEdges = [
      ...outEdges,
      {
        id: `e-${id}-${tid}-loss`,
        source: id,
        target: tid,
        sourceHandle: "loss",
        targetHandle: "loss",
        type: "research_default",
      },
    ];
    return { nodes: outNodes, edges: outEdges };
  }

  if (proposal.trickKind === "l1_reg_shell") {
    ({ nodes: outNodes, edges: outEdges } = stripExistingWeightRegNodes(outNodes, outEdges));
    const id = nextAuxId("l1_reg-trick", outNodes);
    const node = appendResearchNode(
      outNodes,
      "l1_reg",
      pos,
      { ...defaultWeightRegData(), lossScale: proposal.params.lossScale ?? 1 } as Record<
        string,
        unknown
      >,
      id,
    );
    outNodes = [...outNodes, node];
    outEdges = [
      ...outEdges,
      {
        id: `e-${id}-${tid}-loss`,
        source: id,
        target: tid,
        sourceHandle: "loss",
        targetHandle: "loss",
        type: "research_default",
      },
    ];
    return { nodes: outNodes, edges: outEdges };
  }

  if (proposal.trickKind === "l2_projection_shell") {
    const id = nextAuxId("l2_projection-trick", outNodes);
    const node = appendResearchNode(
      outNodes,
      "l2_projection",
      pos,
      { targetNorm: proposal.params.shellRadius } as Record<string, unknown>,
      id,
    );
    outNodes = [...outNodes, node];
    outEdges = [
      ...outEdges,
      {
        id: `e-${id}-${tid}-loss`,
        source: id,
        target: tid,
        sourceHandle: "loss",
        targetHandle: "loss",
        type: "research_default",
      },
    ];
    return { nodes: outNodes, edges: outEdges };
  }

  return { nodes: outNodes, edges: outEdges };
}
