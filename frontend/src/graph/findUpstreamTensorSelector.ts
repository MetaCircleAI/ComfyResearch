import type { Edge, Node } from "@xyflow/react";

function targetHandlesForUpstreamWalk(n: Node): Set<string> | null {
  const t = String(n.type);
  if (
    t === "pca" ||
    t === "svd" ||
    t === "statistics" ||
    t === "effective_rank" ||
    t === "series_endpoint_gap" ||
    t === "smoothing_curve" ||
    t === "derivative_curve" ||
    t === "tensor_constant" ||
    t === "tensor_linspace"
  ) {
    return new Set(["tensor"]);
  }
  if (t === "dimension_permutator") return new Set(["tensor_in"]);
  if (t === "tensor_slicing" || t === "elementwise_transform") return new Set(["tensor"]);
  if (
    t === "statistics2" ||
    t === "tensor_add" ||
    t === "tensor_stack" ||
    t === "tensor_concat" ||
    t === "basic_calculator"
  ) {
    const hs = new Set<string>(["tensor"]);
    for (let i = 1; i <= 32; i += 1) hs.add(`tensor_${i}`);
    return hs;
  }
  if (t === "tensor_viz_0d" || t === "tensor_viz_general" || t === "tensor_viz_1d" || t === "tensor_viz_2d") {
    return new Set(["tensor"]);
  }
  return null;
}

/**
 * Walk the tensor input chain from a tensor viz and return the first `tensor_selector` upstream, if any.
 */
export function findUpstreamTensorSelectorFromTensorViz(
  nodes: Node[],
  edges: Edge[],
  tensorVizId: string,
): Node | null {
  return findTensorSelectorFeedingTensorViz(nodes, edges, tensorVizId)?.selector ?? null;
}

export type TensorSelectorFeedFromViz = { selector: Node; selectorOutputHandle: string };

/**
 * Walk upstream from a tensor viz and return the first `tensor_selector` plus the output handle used
 * toward the viz (e.g. `tensor_1` vs `tensor_2`).
 */
export function findTensorSelectorFeedingTensorViz(
  nodes: Node[],
  edges: Edge[],
  tensorVizId: string,
): TensorSelectorFeedFromViz | null {
  const queue: Array<{ id: string; allowed: Set<string> }> = [
    { id: tensorVizId, allowed: new Set(["tensor", ""]) },
  ];
  const seen = new Set<string>();

  while (queue.length) {
    const { id: curId, allowed } = queue.shift()!;
    if (seen.has(curId)) continue;
    seen.add(curId);

    for (const e of edges) {
      if (e.target !== curId) continue;
      const th = e.targetHandle ?? "";
      if (!allowed.has(th)) continue;

      const src = nodes.find((n) => n.id === e.source);
      if (!src) continue;
      if (src.type === "tensor_selector") {
        return { selector: src, selectorOutputHandle: e.sourceHandle ?? "" };
      }

      const next = targetHandlesForUpstreamWalk(src);
      if (next) queue.push({ id: src.id, allowed: next });
    }
  }
  return null;
}
