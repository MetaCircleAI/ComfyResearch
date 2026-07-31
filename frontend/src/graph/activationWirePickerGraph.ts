import type { Edge, Node } from "@xyflow/react";
import type { MlpModelNodeData } from "../components/nodes/mlpModelDefaults";
import { defaultMlpModelData } from "../components/nodes/mlpModelDefaults";
import { defaultInstanceTitleBase, INSTANCE_TITLE_KEY, readInstanceTitle } from "./nodeInstanceTitle";
import {
  COMBINED_SUBGRAPH_IO_EDGE_TYPE,
  LAYER_STRIP_SOURCE_HANDLE,
  LAYER_STRIP_TARGET_HANDLE,
} from "./layerStripHandles";
import { reconcileMlpLowExpansion } from "./mlpLowLevelExpansion";
import { readGraphNodeLoopCount } from "./nodeLoopCount";
import type { ActivationWireResolvedModel } from "./resolveActivationWireModel";

const SEQUENTIAL_TYPES = new Set([
  "linear_layer",
  "activation_layer",
  "layer_norm_layer",
  "rms_norm_layer",
  "embedding_layer",
  "unembedding_layer",
  "absolute_pos_embed_layer",
  "rotary_embed_layer",
  "local_mixing_layer",
]);

/** Direct children of ``combined_model`` that participate in its inner tensor chain (atomics + nested wrappers). */
const COMBINED_CHAIN_MEMBER_TYPES = new Set([...SEQUENTIAL_TYPES, "combined_model"]);

export type ActivationPickerWireEdge = {
  edgeId: string;
  sourceId: string;
  targetId: string;
  /** Output of ``nn.Sequential[afterModuleIndex]`` matches tensor on this wire. */
  afterModuleIndex: number;
};

export type ActivationWirePickerBuild =
  | {
      ok: true;
      /** React Flow nodes (flat layout, optional custom type). */
      flowNodes: Node[];
      flowEdges: Edge[];
      wires: ActivationPickerWireEdge[];
      /** Ordered chain node ids (front → back), same as Python ``collect_atomic_layer_chain_front_to_back``. */
      chainIds: string[];
      /** Matches canvas ``loopCount`` on the model shell (combined / MLP / parent of chain). */
      loopDisplay: { count: number } | null;
      /** When true, the loop pill is drawn inside the picker flow (combined-model frame), not modal chrome. */
      loopBadgeInFlow?: boolean;
    }
  | { ok: false; message: string };

function isTensorChainEdge(e: Edge, idSet: Set<string>): boolean {
  if (!idSet.has(e.source) || !idSet.has(e.target)) return false;
  const sh = (e.sourceHandle ?? "").trim();
  const th = (e.targetHandle ?? "").trim();
  const shOk = sh === LAYER_STRIP_SOURCE_HANDLE || sh === "tensor" || sh === "tensor_out" || sh === "model" || sh === "";
  const thOk = th === LAYER_STRIP_TARGET_HANDLE || th === "tensor" || th === "tensor_in" || th === "in" || th === "";
  return shOk && thOk;
}

/** Topological order of a single tensor chain (matches trainer / Python atomic walk when unique). */
export function orderSequentialChainIds(memberIds: Set<string>, edges: Edge[]): string[] | null {
  const indeg = new Map<string, number>();
  for (const id of memberIds) indeg.set(id, 0);
  for (const e of edges) {
    if (!isTensorChainEdge(e, memberIds)) continue;
    indeg.set(e.target, (indeg.get(e.target) ?? 0) + 1);
  }
  const heads = [...memberIds].filter((id) => (indeg.get(id) ?? 0) === 0);
  heads.sort();
  if (heads.length !== 1) return null;
  const out: string[] = [];
  const seen = new Set<string>();
  let cur: string | null = heads[0] ?? null;
  while (cur != null) {
    if (seen.has(cur)) return null;
    seen.add(cur);
    out.push(cur);
    const outs = edges
      .filter((e) => e.source === cur && memberIds.has(e.target) && isTensorChainEdge(e, memberIds))
      .map((e) => e.target);
    if (outs.length === 0) {
      cur = null;
    } else if (outs.length > 1) {
      return null;
    } else {
      cur = outs[0] ?? null;
    }
  }
  if (out.length !== memberIds.size) return null;
  return out;
}

function shortLabelForAtomic(n: Node): string {
  const d = (n.data ?? {}) as Record<string, unknown>;
  const title = typeof d[INSTANCE_TITLE_KEY] === "string" ? (d[INSTANCE_TITLE_KEY] as string).trim() : "";
  if (title) return title.slice(0, 28);
  const t = String(n.type).replace(/_layer$/, "").replace(/_/g, " ");
  return t.slice(0, 24) || n.id.slice(0, 8);
}

/** Picker node size — keep in sync with ``ActivationPickerBlockNode`` / CSS. */
const PICKER_NODE_W = 96;
const PICKER_NODE_H = 30;
const PICKER_COL_STEP = 124;

/** Padding and chrome inside the combined-model picker frame (keep in sync with CSS). */
const APW_FRAME_PAD = 16;
const APW_FRAME_TITLE_H = 26;
const APW_FRAME_LOOP_H = 28;
const APW_FRAME_GAP = 8;
const APW_COMBINED_FRAME_ID = "__apw_combined_frame";

function chainEdgesAndWires(chainIds: string[]): { flowEdges: Edge[]; wires: ActivationPickerWireEdge[] } {
  const flowEdges: Edge[] = [];
  const wires: ActivationPickerWireEdge[] = [];
  for (let i = 0; i < chainIds.length - 1; i++) {
    const sourceId = chainIds[i]!;
    const targetId = chainIds[i + 1]!;
    const edgeId = `apw_${sourceId}__${targetId}`;
    flowEdges.push({
      id: edgeId,
      source: sourceId,
      target: targetId,
      sourceHandle: "out",
      targetHandle: "in",
      type: "research_default",
      selectable: true,
      focusable: true,
    });
    wires.push({ edgeId, sourceId, targetId, afterModuleIndex: i });
  }
  return { flowEdges, wires };
}

/** Horizontally and vertically centers the chain on the flow origin so fitView centers on the canvas. */
function buildFlowFromChain(chainIds: string[], nmap: Map<string, Node>): ActivationWirePickerBuild {
  const n = chainIds.length;
  const innerWidth = (n - 1) * PICKER_COL_STEP + PICKER_NODE_W;
  const startX = -innerWidth / 2;
  const startY = -PICKER_NODE_H / 2;
  const flowNodes: Node[] = chainIds.map((id, i) => {
    const raw = nmap.get(id);
    return {
      id,
      type: "activationPickerBlock",
      position: { x: startX + i * PICKER_COL_STEP, y: startY },
      data: { shortLabel: raw ? shortLabelForAtomic(raw) : id },
      width: PICKER_NODE_W,
      height: PICKER_NODE_H,
      style: { width: PICKER_NODE_W, height: PICKER_NODE_H, zIndex: 1 },
    };
  });
  const { flowEdges, wires } = chainEdgesAndWires(chainIds);
  return { ok: true, flowNodes, flowEdges, wires, chainIds, loopDisplay: null, loopBadgeInFlow: false };
}

/**
 * Combined model: wrapped frame with instance title, loop pill above the title row, chain centered inside.
 * Loop pill is embedded in the flow (``loopBadgeInFlow``) so modal chrome does not host it.
 */
function buildCombinedModelFramedFlow(
  chainIds: string[],
  nmap: Map<string, Node>,
  shellNode: Node,
  loopRaw: number | null,
): ActivationWirePickerBuild {
  const n = chainIds.length;
  const innerWidth = (n - 1) * PICKER_COL_STEP + PICKER_NODE_W;
  const loopN = loopRaw != null && Number.isFinite(loopRaw) ? Math.floor(loopRaw) : null;
  const showLoop = loopN != null && loopN >= 2;
  const topStripe =
    APW_FRAME_PAD +
    (showLoop ? APW_FRAME_LOOP_H + APW_FRAME_GAP : 0) +
    APW_FRAME_TITLE_H +
    APW_FRAME_GAP +
    PICKER_NODE_H +
    APW_FRAME_PAD;
  const frameW = innerWidth + 2 * APW_FRAME_PAD;
  const frameH = topStripe;
  const frameLeft = -frameW / 2;
  const frameTop = -frameH / 2;
  /** Child positions are relative to the frame parent (see ``extent: \"parent\"``). */
  const relY =
    APW_FRAME_PAD +
    (showLoop ? APW_FRAME_LOOP_H + APW_FRAME_GAP : 0) +
    APW_FRAME_TITLE_H +
    APW_FRAME_GAP;
  const title = readInstanceTitle(shellNode.data, defaultInstanceTitleBase("combined_model"));

  const frameNode: Node = {
    id: APW_COMBINED_FRAME_ID,
    type: "activationPickerFrame",
    position: { x: frameLeft, y: frameTop },
    data: { title, loopCount: showLoop ? loopN! : 0 },
    width: frameW,
    height: frameH,
    style: {
      width: frameW,
      height: frameH,
      zIndex: 0,
    },
    selectable: true,
    draggable: true,
    focusable: false,
  };

  const blockNodes: Node[] = chainIds.map((id, i) => {
    const raw = nmap.get(id);
    return {
      id,
      type: "activationPickerBlock",
      parentId: APW_COMBINED_FRAME_ID,
      extent: "parent" as const,
      position: { x: APW_FRAME_PAD + i * PICKER_COL_STEP, y: relY },
      data: { shortLabel: raw ? shortLabelForAtomic(raw) : id },
      width: PICKER_NODE_W,
      height: PICKER_NODE_H,
      style: { width: PICKER_NODE_W, height: PICKER_NODE_H },
    };
  });

  const { flowEdges, wires } = chainEdgesAndWires(chainIds);
  const firstId = chainIds[0]!;
  const lastId = chainIds[n - 1]!;
  /** Same handle ids as ``CombinedModelIoStrip`` / main canvas subgraph I/O (not activation-pick targets). */
  flowEdges.push(
    {
      id: `apw_cmb_in__${firstId}`,
      source: APW_COMBINED_FRAME_ID,
      target: firstId,
      sourceHandle: "tensor_boundary",
      targetHandle: "in",
      type: COMBINED_SUBGRAPH_IO_EDGE_TYPE,
      selectable: false,
      focusable: false,
      data: { pickerPickableEdge: false },
    },
    {
      id: `apw_cmb_ret__${lastId}`,
      source: lastId,
      target: APW_COMBINED_FRAME_ID,
      sourceHandle: "out",
      targetHandle: "tensor_return",
      type: COMBINED_SUBGRAPH_IO_EDGE_TYPE,
      selectable: false,
      focusable: false,
      data: { pickerPickableEdge: false },
    },
  );
  const base: Extract<ActivationWirePickerBuild, { ok: true }> = {
    ok: true,
    flowNodes: [frameNode, ...blockNodes],
    flowEdges,
    wires,
    chainIds,
    loopDisplay: null,
    loopBadgeInFlow: showLoop,
  };
  return attachLoopDisplay(base, loopRaw);
}

function loopCountForShell(resolved: ActivationWireResolvedModel, nmap: Map<string, Node>): number | null {
  if (resolved.kind === "mlp" || resolved.kind === "combined_model") {
    return readGraphNodeLoopCount(resolved.node.data);
  }
  if (resolved.kind === "sequential_tip") {
    const pid = resolved.tipNode.parentId;
    if (!pid) return null;
    const parent = nmap.get(pid);
    if (!parent) return null;
    if (parent.type === "combined_model" || parent.type === "mlp_model") {
      return readGraphNodeLoopCount(parent.data);
    }
  }
  return null;
}

function attachLoopDisplay(
  result: Extract<ActivationWirePickerBuild, { ok: true }>,
  loopRaw: number | null,
): Extract<ActivationWirePickerBuild, { ok: true }> {
  const n = loopRaw != null && Number.isFinite(loopRaw) ? Math.floor(loopRaw) : null;
  const loopDisplay = n != null && n >= 2 ? { count: n } : null;
  const loopBadgeInFlow = result.loopBadgeInFlow ?? false;
  return { ...result, loopDisplay, loopBadgeInFlow };
}

/**
 * Ordered atomic layer ids under a combined-model shell, recursively expanding nested ``combined_model`` children.
 * Matches backend ``collect_flat_atomic_chain_under_combined``.
 */
function collectFlatAtomicChainIds(shellId: string, allNodes: Node[], allEdges: Edge[]): string[] | null {
  const nmap = new Map(allNodes.map((n) => [n.id, n]));
  const memberIds = new Set(
    allNodes
      .filter((n) => n.parentId === shellId && COMBINED_CHAIN_MEMBER_TYPES.has(String(n.type)))
      .map((n) => n.id),
  );
  if (memberIds.size === 0) return null;
  const innerEdges = allEdges.filter((e) => isTensorChainEdge(e, memberIds));
  const ordered = orderSequentialChainIds(memberIds, innerEdges);
  if (!ordered) return null;
  const flat: string[] = [];
  for (const id of ordered) {
    const raw = nmap.get(id);
    if (!raw) return null;
    const t = String(raw.type);
    if (SEQUENTIAL_TYPES.has(t)) {
      flat.push(id);
    } else if (t === "combined_model") {
      const sub = collectFlatAtomicChainIds(id, allNodes, allEdges);
      if (!sub || sub.length === 0) return null;
      flat.push(...sub);
    } else {
      return null;
    }
  }
  return flat;
}

function collectChainFromTip(
  tipId: string,
  nmap: Map<string, Node>,
  edges: Edge[],
): { chainRev: string[] } | null {
  const rev: string[] = [];
  let cur: string | null = tipId;
  const seen = new Set<string>();
  while (cur != null) {
    if (seen.has(cur)) return null;
    seen.add(cur);
    const node = nmap.get(cur);
    if (!node || !SEQUENTIAL_TYPES.has(String(node.type))) return null;
    rev.push(cur);
    const preds: string[] = [];
    for (const e of edges) {
      if (e.target !== cur) continue;
      const th = (e.targetHandle ?? "").trim();
      if (!(th === LAYER_STRIP_TARGET_HANDLE || th === "tensor" || th === "tensor_in" || th === "in" || th === "")) {
        continue;
      }
      const sh = (e.sourceHandle ?? "").trim();
      if (!(sh === LAYER_STRIP_SOURCE_HANDLE || sh === "tensor" || sh === "tensor_out" || sh === "model" || sh === "")) {
        continue;
      }
      const src = nmap.get(e.source);
      if (src && SEQUENTIAL_TYPES.has(String(src.type))) preds.push(src.id);
    }
    if (preds.length > 1) return null;
    const nextId = preds[0] ?? null;
    cur = nextId;
  }
  rev.reverse();
  return { chainRev: rev };
}

export function buildActivationWirePickerGraph(
  resolved: ActivationWireResolvedModel,
  allNodes: Node[],
  allEdges: Edge[],
): ActivationWirePickerBuild {
  const nmap = new Map(allNodes.map((n) => [n.id, n]));

  if (resolved.kind === "mlp") {
    const defs = defaultMlpModelData();
    const data: MlpModelNodeData = { ...defs, ...resolved.data };
    const shell: Node = {
      ...resolved.node,
      parentId: undefined,
      extent: undefined,
      position: { x: 0, y: 0 },
    };
    const { nodes: expanded, edges: expEdges } = reconcileMlpLowExpansion([shell], [], shell.id, data);
    const memberIds = new Set(
      expanded.filter((n) => n.parentId === shell.id && SEQUENTIAL_TYPES.has(String(n.type))).map((n) => n.id),
    );
    const innerEdges = expEdges.filter((e) => isTensorChainEdge(e, memberIds));
    const ordered = orderSequentialChainIds(memberIds, innerEdges);
    if (!ordered || ordered.length < 2) {
      return { ok: false, message: "Could not derive a linear layer chain inside the MLP (check depth)." };
    }
    const subMap = new Map(expanded.map((n) => [n.id, n]));
    const flow = buildFlowFromChain(ordered, subMap);
    if (!flow.ok) return flow;
    return attachLoopDisplay({ ...flow, loopBadgeInFlow: false }, loopCountForShell(resolved, nmap));
  }

  if (resolved.kind === "combined_model") {
    const cid = resolved.node.id;
    const ordered = collectFlatAtomicChainIds(cid, allNodes, allEdges);
    if (!ordered || ordered.length < 2) {
      return {
        ok: false,
        message: "Combined model has no inner atomic tensor chain (add linear / activation / … layers inside).",
      };
    }
    const loopRaw = loopCountForShell(resolved, nmap);
    return buildCombinedModelFramedFlow(ordered, nmap, resolved.node, loopRaw);
  }

  if (resolved.kind === "sequential_tip") {
    const collected = collectChainFromTip(resolved.tipNode.id, nmap, allEdges);
    if (!collected) {
      return { ok: false, message: "Could not walk the atomic layer chain from the model connection." };
    }
    const { chainRev } = collected;
    if (chainRev.length < 2) {
      return { ok: false, message: "Atomic model chain needs at least two layers to pick intermediate wires." };
    }
    const flow = buildFlowFromChain(chainRev, nmap);
    if (!flow.ok) return flow;
    return attachLoopDisplay({ ...flow, loopBadgeInFlow: false }, loopCountForShell(resolved, nmap));
  }

  return { ok: false, message: "Wire picking is only available for MLP, combined model, or atomic layer chains." };
}

export function sanitizeTensorKey(raw: string): string {
  const t = raw
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
  const base = t.slice(0, 80);
  return base || "activation";
}
