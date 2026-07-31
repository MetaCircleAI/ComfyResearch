import type { Edge, Node } from "@xyflow/react";
import {
  coerceSweepParamsNumeric,
  mergeSweepParamRecords,
  parseSweepParamsFromSummary,
} from "./sweepParamExtract";

export type UpstreamSweepMetadata = {
  summary: string;
  params: Record<string, string>;
  paramsNumeric: Record<string, number>;
};

function readNodeSweepFields(n: Node): { summary: string; params: Record<string, string> | null } {
  const d = (n.data ?? {}) as { lastSweepSummary?: string; lastSweepParams?: Record<string, string> };
  const summary = typeof d.lastSweepSummary === "string" ? d.lastSweepSummary.trim() : "";
  const rawParams = d.lastSweepParams;
  const params =
    rawParams && typeof rawParams === "object" && !Array.isArray(rawParams)
      ? Object.fromEntries(
          Object.entries(rawParams).filter(
            ([k, v]) => typeof k === "string" && k.trim() && typeof v === "string",
          ),
        )
      : null;
  return { summary, params };
}

function finalizeSweepMetadata(
  summary: string,
  params: Record<string, string>,
): UpstreamSweepMetadata {
  return {
    summary,
    params,
    paramsNumeric: coerceSweepParamsNumeric(params),
  };
}

/**
 * Walk upstream from `startId` (BFS) and return the first non-empty `lastSweepSummary`
 * from a training visualization node; otherwise the first from any upstream node.
 */
export function findUpstreamSweepSummary(nodes: Node[], edges: Edge[], startId: string): string {
  return findUpstreamSweepMetadata(nodes, edges, startId).summary;
}

/**
 * Prefer structured ``lastSweepParams`` from upstream viz; fall back to parsing ``lastSweepSummary``.
 */
export function findUpstreamSweepMetadata(
  nodes: Node[],
  edges: Edge[],
  startId: string,
): UpstreamSweepMetadata {
  const visited = new Set<string>();
  const queue: string[] = [startId];
  let fallbackSummary = "";
  let fallbackParams: Record<string, string> | null = null;

  while (queue.length) {
    const tid = queue.shift()!;
    for (const e of edges) {
      if (e.target !== tid) continue;
      const src = e.source;
      if (visited.has(src)) continue;
      visited.add(src);
      const n = nodes.find((x) => x.id === src);
      if (!n) continue;
      const { summary, params } = readNodeSweepFields(n);
      if (summary || params) {
        const parsed = summary ? parseSweepParamsFromSummary(summary) : {};
        const merged = mergeSweepParamRecords(params ?? {}, parsed);
        if (n.type === "training_visualization") {
          return finalizeSweepMetadata(summary, merged);
        }
        if (!fallbackSummary && !fallbackParams) {
          fallbackSummary = summary;
          fallbackParams = Object.keys(merged).length ? merged : null;
        }
      }
      queue.push(src);
    }
  }

  if (fallbackSummary || fallbackParams) {
    const parsed = fallbackSummary ? parseSweepParamsFromSummary(fallbackSummary) : {};
    return finalizeSweepMetadata(fallbackSummary, mergeSweepParamRecords(fallbackParams ?? {}, parsed));
  }
  return { summary: "", params: {}, paramsNumeric: {} };
}

/** @deprecated Use {@link findUpstreamSweepMetadata} */
export function findUpstreamSweepParams(nodes: Node[], edges: Edge[], startId: string): Record<string, string> {
  return findUpstreamSweepMetadata(nodes, edges, startId).params;
}
