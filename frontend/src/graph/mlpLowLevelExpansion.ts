import type { Edge, Node, NodeChange } from "@xyflow/react";
import type { ActivationLayerNodeData } from "../components/nodes/activationLayerDefaults";
import { defaultActivationLayerData } from "../components/nodes/activationLayerDefaults";
import type { LinearLayerNodeData } from "../components/nodes/linearLayerDefaults";
import { defaultLinearLayerData } from "../components/nodes/linearLayerDefaults";
import type { MlpModelNodeData } from "../components/nodes/mlpModelDefaults";
import { defaultMlpModelData } from "../components/nodes/mlpModelDefaults";
import { intChoices } from "../components/nodes/multiValueUtils";
import { INSTANCE_TITLE_KEY } from "./nodeInstanceTitle";
import {
  COMBINED_MODEL_RETURN_TARGET_HANDLE,
  COMBINED_SUBGRAPH_IO_EDGE_TYPE,
  LAYER_STRIP_SOURCE_HANDLE,
  LAYER_STRIP_TARGET_HANDLE,
} from "./layerStripHandles";
import { readNodeCanvasIoMode } from "./nodeCanvasIoMode";
import { readNodeCanvasLevelMode } from "./nodeCanvasLevelMode";
import { sortNodesParentBeforeChildren } from "./sortNodesParentBeforeChildren";

const SUFFIX_LIN = "__mlp_low_lin_";
const SUFFIX_ACT = "__mlp_low_act_";

/** Match ``ResearchCanvas`` combined-model chrome so children clear header + I/O row. */
const COMBINED_COMBINE_PAD = 18;
const COMBINED_COMBINE_HEAD = 136;
const COMBINED_MODEL_CHILD_MIN_Y = COMBINED_COMBINE_PAD + COMBINED_COMBINE_HEAD;
const COMBINED_MODEL_CHILD_MIN_X = 16;
const COMBINED_SHELL_EDGE_PAD_X = 12;
const COMBINED_SHELL_EDGE_PAD_Y = 10;
const INNER_FALLBACK_W = 256;
const INNER_FALLBACK_H = 256;

const ATOMIC_LAYER_TYPES = new Set([
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

function parseCssPixel(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  if (typeof value !== "string") return null;
  const m = /^(\d+(?:\.\d+)?)px\s*$/i.exec(value.trim());
  if (!m) return null;
  const n = Number(m[1]);
  return Number.isFinite(n) && n > 0 ? n : null;
}

function readNodeOuterSizeForReflow(n: Node): { width: number; height: number } {
  const m = n.measured;
  const mw = typeof m?.width === "number" && m.width > 0 ? m.width : null;
  const mh = typeof m?.height === "number" && m.height > 0 ? m.height : null;
  if (mw != null && mh != null) return { width: mw, height: mh };

  const nw = typeof n.width === "number" && n.width > 0 ? n.width : null;
  const nh = typeof n.height === "number" && n.height > 0 ? n.height : null;
  if (nw != null && nh != null) return { width: nw, height: nh };

  const iw = typeof n.initialWidth === "number" && n.initialWidth > 0 ? n.initialWidth : null;
  const ih = typeof n.initialHeight === "number" && n.initialHeight > 0 ? n.initialHeight : null;
  if (iw != null && ih != null) return { width: iw, height: ih };

  const st = (n.style ?? {}) as Record<string, unknown>;
  const sw = parseCssPixel(st.width) ?? parseCssPixel(st.minWidth);
  const sh = parseCssPixel(st.height) ?? parseCssPixel(st.minHeight);
  if (sw != null && sh != null) return { width: sw, height: sh };
  if (sw != null) return { width: sw, height: sh ?? INNER_FALLBACK_H };
  if (sh != null) return { width: INNER_FALLBACK_W, height: sh };

  const t = String(n.type);
  /** Slightly wide vs legacy 252 so MLP low-level shell refit matches real linear/activation chrome. */
  if (ATOMIC_LAYER_TYPES.has(t)) return { width: 292, height: 248 };
  return { width: INNER_FALLBACK_W, height: INNER_FALLBACK_H };
}

function addToNodeFrameSize(
  style: Record<string, unknown> | undefined,
  dw: number,
  dh: number,
): Record<string, unknown> | undefined {
  if ((dw === 0 && dh === 0) || !style || typeof style !== "object") return style;
  const out = { ...style };
  const addDim = (key: "width" | "height", delta: number) => {
    if (delta === 0) return;
    const v = out[key];
    if (typeof v === "number" && Number.isFinite(v)) {
      out[key] = Math.max(0, v + delta);
    } else if (typeof v === "string") {
      const m = /^(\d+(?:\.\d+)?)px\s*$/.exec(v);
      if (m) out[key] = `${Math.round(Number(m[1]) + delta)}px`;
    }
  };
  addDim("width", dw);
  addDim("height", dh);
  return out;
}

function isMlpModelLowShellNode(n: Node): boolean {
  return (
    String(n.type) === "mlp_model" && readNodeCanvasLevelMode((n.data ?? {}) as Record<string, unknown>) === "low"
  );
}

function migrateMlpLowChildChromeInset(nodes: Node[]): Node[] {
  const mlpLowShellIds = new Set(nodes.filter((n) => isMlpModelLowShellNode(n)).map((n) => n.id));
  if (mlpLowShellIds.size === 0) return nodes;

  const childrenByParent = new Map<string, Node[]>();
  for (const n of nodes) {
    if (!n.parentId) continue;
    const pid = String(n.parentId);
    if (!mlpLowShellIds.has(pid)) continue;
    if (!childrenByParent.has(pid)) childrenByParent.set(pid, []);
    childrenByParent.get(pid)!.push(n);
  }

  const parentDeltas = new Map<string, { dx: number; dy: number }>();
  for (const [pid, kids] of childrenByParent) {
    let minX = Infinity;
    let minY = Infinity;
    for (const k of kids) {
      minX = Math.min(minX, k.position.x);
      minY = Math.min(minY, k.position.y);
    }
    if (!Number.isFinite(minX) || !Number.isFinite(minY)) continue;
    const dx = Math.max(0, COMBINED_MODEL_CHILD_MIN_X - minX);
    const dy = Math.max(0, COMBINED_MODEL_CHILD_MIN_Y - minY);
    if (dx === 0 && dy === 0) continue;
    parentDeltas.set(pid, { dx, dy });
  }
  if (parentDeltas.size === 0) return nodes;

  return nodes.map((n) => {
    const del = parentDeltas.get(n.id);
    if (n.type === "mlp_model" && isMlpModelLowShellNode(n) && del && (del.dx || del.dy)) {
      const data = (n.data ?? {}) as Record<string, unknown>;
      const ef = data.__expandedFrame as { width?: number; height?: number } | undefined;
      let nextData = data;
      if (ef && (typeof ef.width === "number" || typeof ef.height === "number")) {
        nextData = {
          ...data,
          __expandedFrame: {
            width: (typeof ef.width === "number" ? ef.width : 0) + del.dx,
            height: (typeof ef.height === "number" ? ef.height : 0) + del.dy,
          },
        };
      }
      return {
        ...n,
        position: { x: n.position.x - del.dx, y: n.position.y - del.dy },
        style: addToNodeFrameSize((n.style ?? {}) as Record<string, unknown>, del.dx, del.dy),
        data: nextData,
      };
    }
    const cdel = n.parentId ? parentDeltas.get(String(n.parentId)) : undefined;
    if (cdel && (cdel.dx || cdel.dy)) {
      return {
        ...n,
        position: { x: n.position.x + cdel.dx, y: n.position.y + cdel.dy },
      };
    }
    return n;
  });
}

function refitMlpLowShells(nodes: Node[]): Node[] {
  const mlpLowIds = new Set(nodes.filter((n) => isMlpModelLowShellNode(n)).map((n) => n.id));
  if (mlpLowIds.size === 0) return nodes;

  const shellExtents = (pid: string) => {
    let maxR = COMBINED_MODEL_CHILD_MIN_X;
    let maxB = COMBINED_MODEL_CHILD_MIN_Y;
    for (const c of nodes) {
      if (String(c.parentId) !== pid) continue;
      const { width, height } = readNodeOuterSizeForReflow(c);
      maxR = Math.max(maxR, c.position.x + width);
      maxB = Math.max(maxB, c.position.y + height);
    }
    return {
      needW: Math.max(236, maxR + COMBINED_SHELL_EDGE_PAD_X),
      needH: Math.max(COMBINED_MODEL_CHILD_MIN_Y + 20, maxB + COMBINED_SHELL_EDGE_PAD_Y),
    };
  };

  return nodes.map((n) => {
    if (String(n.type) !== "mlp_model" || !mlpLowIds.has(n.id)) return n;
    const hasKids = nodes.some((c) => c.parentId === n.id);
    if (!hasKids) return n;
    const { needW, needH } = shellExtents(n.id);
    const st = (n.style ?? {}) as Record<string, unknown>;
    const d = (n.data ?? {}) as Record<string, unknown>;
    const ef = d.__expandedFrame as { width?: number; height?: number } | undefined;
    const curW = typeof st.width === "number" ? st.width : ef?.width;
    const curH = typeof st.height === "number" ? st.height : ef?.height;
    if (curW === needW && curH === needH) return n;
    return {
      ...n,
      style: { ...st, width: needW, height: needH },
      data: {
        ...d,
        __expandedFrame: { width: needW, height: needH },
      },
    };
  });
}

function finalizeMlpLowExpansionNodes(nodes: Node[]): Node[] {
  let out = sortNodesParentBeforeChildren(nodes);
  out = migrateMlpLowChildChromeInset(out);
  out = refitMlpLowShells(out);
  return out;
}

export function mlpLowLinearNodeId(mlpId: string, index: number): string {
  return `${mlpId}${SUFFIX_LIN}${index}`;
}

export function mlpLowActivationNodeId(mlpId: string, index: number): string {
  return `${mlpId}${SUFFIX_ACT}${index}`;
}

export function isMlpLowExpansionNodeId(nodeId: string, mlpId: string): boolean {
  return nodeId.startsWith(`${mlpId}${SUFFIX_LIN}`) || nodeId.startsWith(`${mlpId}${SUFFIX_ACT}`);
}

function parseMlpLowLinearId(nodeId: string): { mlpId: string; index: number } | null {
  const i = nodeId.indexOf(SUFFIX_LIN);
  if (i < 0) return null;
  const rest = nodeId.slice(i + SUFFIX_LIN.length);
  const index = Number.parseInt(rest, 10);
  if (!Number.isFinite(index) || index < 0) return null;
  return { mlpId: nodeId.slice(0, i), index };
}

function parseMlpLowActivationId(nodeId: string): { mlpId: string; index: number } | null {
  const i = nodeId.indexOf(SUFFIX_ACT);
  if (i < 0) return null;
  const rest = nodeId.slice(i + SUFFIX_ACT.length);
  const index = Number.parseInt(rest, 10);
  if (!Number.isFinite(index) || index < 0) return null;
  return { mlpId: nodeId.slice(0, i), index };
}

function isMlpLowInternalEdge(e: Edge, mlpId: string): boolean {
  const src =
    e.source.startsWith(`${mlpId}${SUFFIX_LIN}`) || e.source.startsWith(`${mlpId}${SUFFIX_ACT}`);
  const tgt =
    e.target.startsWith(`${mlpId}${SUFFIX_LIN}`) || e.target.startsWith(`${mlpId}${SUFFIX_ACT}`);
  return src && tgt;
}

/** Auto ``tensor_boundary`` / ``tensor_return`` wiring between the MLP shell and the inner chain. */
function isMlpLowShellSubgraphIoEdge(e: Edge, mlpId: string): boolean {
  if (String(e.type) !== COMBINED_SUBGRAPH_IO_EDGE_TYPE) return false;
  const sh = (e.sourceHandle ?? "").trim();
  const th = (e.targetHandle ?? "").trim();
  if (e.source === mlpId && sh === "tensor_boundary") return true;
  if (e.target === mlpId && th === COMBINED_MODEL_RETURN_TARGET_HANDLE) return true;
  return false;
}

function mlpLowShellSubgraphIoEdges(mlpId: string, depth: number): Edge[] {
  const nLin = depth + 1;
  if (nLin < 1) return [];
  const firstId = mlpLowLinearNodeId(mlpId, 0);
  const lastId = mlpLowLinearNodeId(mlpId, nLin - 1);
  return [
    {
      id: `${mlpId}__mlp_low_shell_in`,
      type: COMBINED_SUBGRAPH_IO_EDGE_TYPE,
      source: mlpId,
      target: firstId,
      sourceHandle: "tensor_boundary",
      targetHandle: LAYER_STRIP_TARGET_HANDLE,
    },
    {
      id: `${mlpId}__mlp_low_shell_out`,
      type: COMBINED_SUBGRAPH_IO_EDGE_TYPE,
      source: lastId,
      target: mlpId,
      sourceHandle: LAYER_STRIP_SOURCE_HANDLE,
      targetHandle: COMBINED_MODEL_RETURN_TARGET_HANDLE,
    },
  ];
}

function mlpExpansionFingerprint(nodes: Node[], edges: Edge[], mlpId: string): string {
  const expNodes = nodes
    .filter((n) => n.id === mlpId || isMlpLowExpansionNodeId(n.id, mlpId))
    .map((n) => ({
      id: n.id,
      type: n.type,
      x: n.position.x,
      y: n.position.y,
      parentId: n.parentId ?? null,
      style: n.style ?? null,
      data: n.data ?? null,
    }));
  expNodes.sort((a, b) => a.id.localeCompare(b.id));
  const expEdgeIds = edges
    .filter((e) => isMlpLowInternalEdge(e, mlpId) || isMlpLowShellSubgraphIoEdge(e, mlpId))
    .map((e) => e.id)
    .sort();
  return JSON.stringify({ expNodes, expEdgeIds });
}

function internalEdgesForChain(mlpId: string, depth: number): Edge[] {
  const out: Edge[] = [];
  for (let i = 0; i < depth; i++) {
    out.push({
      id: `${mlpId}__mlp_low_e_lin${i}_act${i}`,
      type: "research_default",
      source: mlpLowLinearNodeId(mlpId, i),
      sourceHandle: LAYER_STRIP_SOURCE_HANDLE,
      target: mlpLowActivationNodeId(mlpId, i),
      targetHandle: LAYER_STRIP_TARGET_HANDLE,
    });
    out.push({
      id: `${mlpId}__mlp_low_e_act${i}_lin${i + 1}`,
      type: "research_default",
      source: mlpLowActivationNodeId(mlpId, i),
      sourceHandle: LAYER_STRIP_SOURCE_HANDLE,
      target: mlpLowLinearNodeId(mlpId, i + 1),
      targetHandle: LAYER_STRIP_TARGET_HANDLE,
    });
  }
  return out;
}

/**
 * Horizontal distance between successive inner nodes' **left** edges.
 * Must cover real card width (header + fields + spec footer) plus space so wires read clearly.
 */
const MLP_LOW_COLUMN_STEP_X = 340;
/** Reserved width from the last inner node's left edge to the shell's right inner pad. */
const MLP_LOW_LAST_COLUMN_CONTENT_W = 312;

function expansionChildPosition(
  col: number,
  existing: Map<string, Node>,
  id: string,
): { x: number; y: number } {
  const prev = existing.get(id);
  const y =
    prev?.parentId && Number.isFinite(prev.position?.y) ? prev.position.y : COMBINED_MODEL_CHILD_MIN_Y;
  return {
    x: COMBINED_MODEL_CHILD_MIN_X + col * MLP_LOW_COLUMN_STEP_X,
    y,
  };
}

function clearMlpLowShellSizing(n: Node): Node {
  if (n.type !== "mlp_model") return n;
  const st = (n.style ?? {}) as Record<string, unknown>;
  const hasStyleSize = st.width != null || st.height != null;
  const d = (n.data ?? {}) as Record<string, unknown>;
  const hasFrame = d.__expandedFrame != null;
  if (!hasStyleSize && !hasFrame) return n;
  const nextSt = { ...st };
  delete nextSt.width;
  delete nextSt.height;
  const nextData = { ...d };
  delete nextData.__expandedFrame;
  return {
    ...n,
    style: Object.keys(nextSt).length ? nextSt : undefined,
    data: nextData,
  };
}

/** Remove MLP-owned low expansion nodes and any edges touching them; reset MLP shell sizing. */
export function removeMlpLowExpansionFromGraph(mlpId: string, nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  const hasExpansionNodes = nodes.some((n) => isMlpLowExpansionNodeId(n.id, mlpId));
  const mlpNode = nodes.find((n) => n.id === mlpId);
  let nextNodes = nodes;
  if (hasExpansionNodes) {
    nextNodes = nodes
      .filter((n) => !isMlpLowExpansionNodeId(n.id, mlpId))
      .map((n) => (n.id === mlpId ? clearMlpLowShellSizing(n) : n));
  } else if (mlpNode) {
    const cleared = clearMlpLowShellSizing(mlpNode);
    if (cleared !== mlpNode) {
      nextNodes = nodes.map((n) => (n.id === mlpId ? cleared : n));
    }
  }
  const hasExpansionEdges = edges.some(
    (e) => isMlpLowExpansionNodeId(e.source, mlpId) || isMlpLowExpansionNodeId(e.target, mlpId),
  );
  const nextEdges = hasExpansionEdges
    ? edges.filter(
        (e) => !isMlpLowExpansionNodeId(e.source, mlpId) && !isMlpLowExpansionNodeId(e.target, mlpId),
      )
    : edges;
  if (nextNodes === nodes && nextEdges === edges) {
    return { nodes, edges };
  }
  return { nodes: nextNodes, edges: nextEdges };
}

/**
 * Reconcile low-level MLP: inner ``linear_layer`` / ``activation_layer`` as **children** of the MLP node
 * (same pattern as ``combined_model``), wired ``tensor_out`` → ``tensor_in``. Training still uses the MLP node.
 */
export function reconcileMlpLowExpansion(
  nodes: Node[],
  edges: Edge[],
  mlpId: string,
  d: MlpModelNodeData,
): { nodes: Node[]; edges: Edge[] } {
  const defs = defaultMlpModelData();
  const full = { ...defs, ...d };
  const shellIo = readNodeCanvasIoMode(full as Record<string, unknown>);
  const depth = intChoices(full.depth, 2)[0];
  const nLin = depth + 1;

  const existing = new Map(nodes.map((n) => [n.id, n]));
  const addNodes: Node[] = [];
  let col = 0;

  for (let li = 0; li < nLin; li++) {
    const lid = mlpLowLinearNodeId(mlpId, li);
    const prevLin = (existing.get(lid)?.data ?? {}) as Partial<LinearLayerNodeData>;
    const linData: LinearLayerNodeData & Record<string, unknown> = {
      ...defaultLinearLayerData(),
      ...prevLin,
      bias: 1,
      seed: full.seed,
      ioMode: "input-output",
      levelMode: "high",
      [INSTANCE_TITLE_KEY]: `Linear ${li}`,
    };
    if (li === 0) {
      linData.inFeatures = full.inputDim;
      linData.outFeatures = nLin > 1 ? full.width : full.outputDim;
    } else if (li === nLin - 1) {
      linData.inFeatures = full.width;
      linData.outFeatures = full.outputDim;
    } else {
      linData.inFeatures = full.width;
      linData.outFeatures = full.width;
    }

    addNodes.push({
      id: lid,
      type: "linear_layer",
      parentId: mlpId,
      extent: "parent",
      position: expansionChildPosition(col, existing, lid),
      data: linData,
    });
    col++;

    if (li < nLin - 1) {
      const aid = mlpLowActivationNodeId(mlpId, li);
      const prevAct = (existing.get(aid)?.data ?? {}) as Partial<ActivationLayerNodeData>;
      const actData: ActivationLayerNodeData & Record<string, unknown> = {
        ...defaultActivationLayerData(),
        ...prevAct,
        activation: full.activation,
        ioMode: "input-output",
        levelMode: "high",
        [INSTANCE_TITLE_KEY]: `Activation ${li}`,
      };
      addNodes.push({
        id: aid,
        type: "activation_layer",
        parentId: mlpId,
        extent: "parent",
        position: expansionChildPosition(col, existing, aid),
        data: actData,
      });
      col++;
    }
  }

  const withoutExpansion = nodes.filter((n) => !isMlpLowExpansionNodeId(n.id, mlpId));
  const mlpBase = withoutExpansion.find((n) => n.id === mlpId);
  const prelimW =
    COMBINED_MODEL_CHILD_MIN_X +
    (col > 0 ? (col - 1) * MLP_LOW_COLUMN_STEP_X : 0) +
    MLP_LOW_LAST_COLUMN_CONTENT_W +
    COMBINED_SHELL_EDGE_PAD_X;
  const prelimH = COMBINED_MODEL_CHILD_MIN_Y + INNER_FALLBACK_H + COMBINED_SHELL_EDGE_PAD_Y;
  const mergedMlp =
    mlpBase && mlpBase.type === "mlp_model"
      ? {
          ...mlpBase,
          style: {
            ...((mlpBase.style ?? {}) as Record<string, unknown>),
            width: Math.max(236, prelimW),
            height: Math.max(COMBINED_MODEL_CHILD_MIN_Y + 20, prelimH),
          },
          data: {
            ...((mlpBase.data ?? {}) as Record<string, unknown>),
            __expandedFrame: { width: Math.max(236, prelimW), height: Math.max(COMBINED_MODEL_CHILD_MIN_Y + 20, prelimH) },
          },
        }
      : mlpBase;

  const others = withoutExpansion.map((n) => (n.id === mlpId && mergedMlp ? mergedMlp : n));
  let nextNodes = finalizeMlpLowExpansionNodes([...others, ...addNodes]);

  const idSet = new Set(nextNodes.map((n) => n.id));
  let nextEdges = edges.filter((e) => idSet.has(e.source) && idSet.has(e.target));
  nextEdges = nextEdges.filter((e) => !isMlpLowInternalEdge(e, mlpId));
  nextEdges = nextEdges.filter((e) => !isMlpLowShellSubgraphIoEdge(e, mlpId));
  nextEdges = [...nextEdges, ...internalEdgesForChain(mlpId, depth)];
  if (shellIo === "input-output") {
    nextEdges = [...nextEdges, ...mlpLowShellSubgraphIoEdges(mlpId, depth)];
  }

  if (mlpExpansionFingerprint(nodes, edges, mlpId) === mlpExpansionFingerprint(nextNodes, nextEdges, mlpId)) {
    return { nodes, edges };
  }
  return { nodes: nextNodes, edges: nextEdges };
}

function mlpPatchFromLinearRow(
  mlp: MlpModelNodeData,
  index: number,
  row: LinearLayerNodeData,
): Partial<MlpModelNodeData> {
  const defs = defaultMlpModelData();
  const full = { ...defs, ...mlp };
  const depth = intChoices(full.depth, 2)[0];
  const nLin = depth + 1;
  const patch: Partial<MlpModelNodeData> = {};

  if (index === 0 && nLin === 1) {
    patch.inputDim = row.inFeatures;
    patch.outputDim = row.outFeatures;
  } else if (index === 0) {
    patch.inputDim = row.inFeatures;
    patch.width = row.outFeatures;
  } else if (index === nLin - 1) {
    patch.outputDim = row.outFeatures;
    patch.width = row.inFeatures;
  } else {
    patch.width = row.outFeatures;
  }
  return patch;
}

/** After a linear expansion node updates, push compatible fields onto the owning MLP. */
export function applyMlpOwnerPatchForLinearExpansion(
  nodes: Node[],
  linearNodeId: string,
  nextLinearData: LinearLayerNodeData,
): Node[] {
  const parsed = parseMlpLowLinearId(linearNodeId);
  if (!parsed) return nodes;
  const { mlpId, index } = parsed;
  const mlpNode = nodes.find((n) => n.id === mlpId && n.type === "mlp_model");
  if (!mlpNode) return nodes;

  const mlpData = { ...defaultMlpModelData(), ...(mlpNode.data as MlpModelNodeData) };
  const patch = mlpPatchFromLinearRow(mlpData, index, nextLinearData);

  return nodes.map((n) => {
    if (n.id !== mlpId) return n;
    return { ...n, data: { ...(n.data as Record<string, unknown>), ...patch } };
  });
}

/** After an activation expansion node updates, mirror activation onto the owning MLP. */
export function applyMlpOwnerPatchForActivationExpansion(
  nodes: Node[],
  activationNodeId: string,
  nextActData: ActivationLayerNodeData,
): Node[] {
  const parsed = parseMlpLowActivationId(activationNodeId);
  if (!parsed) return nodes;
  const { mlpId } = parsed;
  const mlpNode = nodes.find((n) => n.id === mlpId && n.type === "mlp_model");
  if (!mlpNode) return nodes;

  return nodes.map((n) => {
    if (n.id !== mlpId) return n;
    return {
      ...n,
      data: { ...(n.data as Record<string, unknown>), activation: nextActData.activation },
    };
  });
}

/** When an MLP node is removed, also remove its owned low-expansion subgraph. */
export function augmentNodeRemovesWithMlpLowExpansion(changes: NodeChange[], nodes: Node[]): NodeChange[] {
  const extra: NodeChange[] = [];
  const removing = new Set<string>();
  for (const ch of changes) {
    if (ch.type === "remove") removing.add(ch.id);
  }
  for (const ch of changes) {
    if (ch.type !== "remove") continue;
    const n = nodes.find((x) => x.id === ch.id);
    if (n?.type !== "mlp_model") continue;
    for (const x of nodes) {
      if (removing.has(x.id)) continue;
      if (isMlpLowExpansionNodeId(x.id, ch.id) && (x.parentId === ch.id || x.parentId == null || x.parentId === "")) {
        removing.add(x.id);
        extra.push({ type: "remove", id: x.id });
      }
    }
  }
  return extra.length ? [...changes, ...extra] : changes;
}
