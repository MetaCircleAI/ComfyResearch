import type { Edge, Node, NodeChange } from "@xyflow/react";
import { defaultAttentionOnlyModelData, type AttentionOnlyModelNodeData } from "../components/nodes/attentionOnlyModelDefaults";
import { defaultLinearLayerData, type LinearLayerNodeData } from "../components/nodes/linearLayerDefaults";
import { intChoices } from "../components/nodes/multiValueUtils";

function attentionLowIsCausal(full: AttentionOnlyModelNodeData): boolean {
  const v = full.causalAttention;
  const s = String(Array.isArray(v) ? v[0] : v)
    .trim()
    .toLowerCase();
  return s !== "no" && s !== "false" && s !== "0";
}
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

const SUFFIX_LINEAR_IN = "__attn_low_linear_in";
const SUFFIX_SPLITTER = "__attn_low_split";
const SUFFIX_RESHAPE_0 = "__attn_low_reshape_0";
const SUFFIX_RESHAPE_1 = "__attn_low_reshape_1";
const SUFFIX_RESHAPE_2 = "__attn_low_reshape_2";
const SUFFIX_EINSUM_0 = "__attn_low_einsum_0";
const SUFFIX_CAUSAL_MASK = "__attn_low_causal_mask";
const SUFFIX_SOFTMAX = "__attn_low_softmax";
const SUFFIX_EINSUM_1 = "__attn_low_einsum_1";
const SUFFIX_RESHAPE_3 = "__attn_low_reshape_3";
const SUFFIX_LINEAR_OUT = "__attn_low_linear_out";

const COMBINED_COMBINE_PAD = 18;
const COMBINED_COMBINE_HEAD = 136;
const COMBINED_MODEL_CHILD_MIN_Y = COMBINED_COMBINE_PAD + COMBINED_COMBINE_HEAD;
const COMBINED_MODEL_CHILD_MIN_X = 16;
const COMBINED_SHELL_EDGE_PAD_X = 12;
const COMBINED_SHELL_EDGE_PAD_Y = 10;
const ATTN_LOW_COLUMN_STEP_X = 312;
const ATTN_LOW_LAST_COLUMN_CONTENT_W = 292;
const INNER_FALLBACK_W = 292;
const INNER_FALLBACK_H = 240;
const ATTN_LOW_NODE_TYPES = new Set([
  "linear_layer",
  "tensor_splitter",
  "reshape",
  "flatten",
  "einsum",
  "softmax",
  "causal_mask",
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
  if (ATTN_LOW_NODE_TYPES.has(String(n.type))) return { width: INNER_FALLBACK_W, height: INNER_FALLBACK_H };
  return { width: INNER_FALLBACK_W, height: INNER_FALLBACK_H };
}

function addToNodeFrameSize(style: Record<string, unknown> | undefined, dw: number, dh: number) {
  if ((dw === 0 && dh === 0) || !style || typeof style !== "object") return style;
  const out = { ...style };
  const addDim = (key: "width" | "height", delta: number) => {
    if (delta === 0) return;
    const v = out[key];
    if (typeof v === "number" && Number.isFinite(v)) out[key] = Math.max(0, v + delta);
    else if (typeof v === "string") {
      const m = /^(\d+(?:\.\d+)?)px\s*$/.exec(v);
      if (m) out[key] = `${Math.round(Number(m[1]) + delta)}px`;
    }
  };
  addDim("width", dw);
  addDim("height", dh);
  return out;
}

function isAttentionModelLowShellNode(n: Node): boolean {
  return String(n.type) === "attention_only_model" && readNodeCanvasLevelMode((n.data ?? {}) as Record<string, unknown>) === "low";
}

function refitAttentionLowShells(nodes: Node[]): Node[] {
  const attnLowIds = new Set(nodes.filter((n) => isAttentionModelLowShellNode(n)).map((n) => n.id));
  if (attnLowIds.size === 0) return nodes;
  const shellExtents = (pid: string) => {
    let maxR = COMBINED_MODEL_CHILD_MIN_X;
    let maxB = COMBINED_MODEL_CHILD_MIN_Y;
    for (const c of nodes) {
      if (String(c.parentId) !== pid) continue;
      const { width, height } = readNodeOuterSizeForReflow(c);
      maxR = Math.max(maxR, c.position.x + width);
      maxB = Math.max(maxB, c.position.y + height);
    }
    return { needW: Math.max(236, maxR + COMBINED_SHELL_EDGE_PAD_X), needH: Math.max(COMBINED_MODEL_CHILD_MIN_Y + 20, maxB + COMBINED_SHELL_EDGE_PAD_Y) };
  };
  return nodes.map((n) => {
    if (String(n.type) !== "attention_only_model" || !attnLowIds.has(n.id)) return n;
    if (!nodes.some((c) => c.parentId === n.id)) return n;
    const { needW, needH } = shellExtents(n.id);
    return { ...n, style: { ...((n.style ?? {}) as Record<string, unknown>), width: needW, height: needH }, data: { ...((n.data ?? {}) as Record<string, unknown>), __expandedFrame: { width: needW, height: needH } } };
  });
}

function migrateAttentionLowChildChromeInset(nodes: Node[]): Node[] {
  const attnLowShellIds = new Set(nodes.filter((n) => isAttentionModelLowShellNode(n)).map((n) => n.id));
  if (attnLowShellIds.size === 0) return nodes;
  const childrenByParent = new Map<string, Node[]>();
  for (const n of nodes) {
    if (!n.parentId) continue;
    const pid = String(n.parentId);
    if (!attnLowShellIds.has(pid)) continue;
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
    const dx = Math.max(0, COMBINED_MODEL_CHILD_MIN_X - minX);
    const dy = Math.max(0, COMBINED_MODEL_CHILD_MIN_Y - minY);
    if (Number.isFinite(dx) && Number.isFinite(dy) && (dx || dy)) parentDeltas.set(pid, { dx, dy });
  }
  if (parentDeltas.size === 0) return nodes;
  return nodes.map((n) => {
    const del = parentDeltas.get(n.id);
    if (n.type === "attention_only_model" && isAttentionModelLowShellNode(n) && del && (del.dx || del.dy)) {
      const data = (n.data ?? {}) as Record<string, unknown>;
      const ef = data.__expandedFrame as { width?: number; height?: number } | undefined;
      return {
        ...n,
        position: { x: n.position.x - del.dx, y: n.position.y - del.dy },
        style: addToNodeFrameSize((n.style ?? {}) as Record<string, unknown>, del.dx, del.dy),
        data: ef ? { ...data, __expandedFrame: { width: (typeof ef.width === "number" ? ef.width : 0) + del.dx, height: (typeof ef.height === "number" ? ef.height : 0) + del.dy } } : data,
      };
    }
    const cdel = n.parentId ? parentDeltas.get(String(n.parentId)) : undefined;
    if (cdel && (cdel.dx || cdel.dy)) return { ...n, position: { x: n.position.x + cdel.dx, y: n.position.y + cdel.dy } };
    return n;
  });
}

function finalizeAttentionLowExpansionNodes(nodes: Node[]): Node[] {
  return refitAttentionLowShells(migrateAttentionLowChildChromeInset(sortNodesParentBeforeChildren(nodes)));
}

function attentionLowNodeId(attnId: string, suffix: string): string {
  return `${attnId}${suffix}`;
}

export function isAttentionLowExpansionNodeId(nodeId: string, attnId: string): boolean {
  return nodeId.startsWith(`${attnId}__attn_low_`);
}

function isAttentionLowInternalEdge(e: Edge, attnId: string): boolean {
  const pref = `${attnId}__attn_low_`;
  return e.source.startsWith(pref) && e.target.startsWith(pref);
}

function isAttentionLowShellSubgraphIoEdge(e: Edge, attnId: string): boolean {
  if (String(e.type) !== COMBINED_SUBGRAPH_IO_EDGE_TYPE) return false;
  const sh = (e.sourceHandle ?? "").trim();
  const th = (e.targetHandle ?? "").trim();
  return (e.source === attnId && sh === "tensor_boundary") || (e.target === attnId && th === COMBINED_MODEL_RETURN_TARGET_HANDLE);
}

function clearAttentionLowShellSizing(n: Node): Node {
  if (n.type !== "attention_only_model") return n;
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

export function removeAttentionLowExpansionFromGraph(attnId: string, nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  const hasExpansionNodes = nodes.some((n) => isAttentionLowExpansionNodeId(n.id, attnId));
  const attnNode = nodes.find((n) => n.id === attnId);
  let nextNodes = nodes;
  if (hasExpansionNodes) {
    nextNodes = nodes
      .filter((n) => !isAttentionLowExpansionNodeId(n.id, attnId))
      .map((n) => (n.id === attnId ? clearAttentionLowShellSizing(n) : n));
  } else if (attnNode) {
    const cleared = clearAttentionLowShellSizing(attnNode);
    if (cleared !== attnNode) {
      nextNodes = nodes.map((n) => (n.id === attnId ? cleared : n));
    }
  }
  const hasExpansionEdges = edges.some(
    (e) => isAttentionLowExpansionNodeId(e.source, attnId) || isAttentionLowExpansionNodeId(e.target, attnId),
  );
  const nextEdges = hasExpansionEdges
    ? edges.filter(
        (e) => !isAttentionLowExpansionNodeId(e.source, attnId) && !isAttentionLowExpansionNodeId(e.target, attnId),
      )
    : edges;
  if (nextNodes === nodes && nextEdges === edges) {
    return { nodes, edges };
  }
  return { nodes: nextNodes, edges: nextEdges };
}

function childPos(col: number, row: number) {
  const rowStep = 92;
  const centerY = COMBINED_MODEL_CHILD_MIN_Y + rowStep;
  return { x: COMBINED_MODEL_CHILD_MIN_X + col * ATTN_LOW_COLUMN_STEP_X, y: centerY + row * rowStep };
}

function buildInternalEdges(attnId: string, causal: boolean): Edge[] {
  const linearIn = attentionLowNodeId(attnId, SUFFIX_LINEAR_IN);
  const split = attentionLowNodeId(attnId, SUFFIX_SPLITTER);
  const qReshape = attentionLowNodeId(attnId, SUFFIX_RESHAPE_0);
  const kReshape = attentionLowNodeId(attnId, SUFFIX_RESHAPE_1);
  const vReshape = attentionLowNodeId(attnId, SUFFIX_RESHAPE_2);
  const qk = attentionLowNodeId(attnId, SUFFIX_EINSUM_0);
  const causalMask = attentionLowNodeId(attnId, SUFFIX_CAUSAL_MASK);
  const softmax = attentionLowNodeId(attnId, SUFFIX_SOFTMAX);
  const av = attentionLowNodeId(attnId, SUFFIX_EINSUM_1);
  const outReshape = attentionLowNodeId(attnId, SUFFIX_RESHAPE_3);
  const linearOut = attentionLowNodeId(attnId, SUFFIX_LINEAR_OUT);
  const base: Edge[] = [
    { id: `${attnId}__attn_low_linear_in`, type: "research_default", source: linearIn, sourceHandle: LAYER_STRIP_SOURCE_HANDLE, target: split, targetHandle: LAYER_STRIP_TARGET_HANDLE },
    { id: `${attnId}__attn_low_split_0`, type: "research_default", source: split, sourceHandle: "tensor_0", target: qReshape, targetHandle: LAYER_STRIP_TARGET_HANDLE },
    { id: `${attnId}__attn_low_split_1`, type: "research_default", source: split, sourceHandle: "tensor_1", target: kReshape, targetHandle: LAYER_STRIP_TARGET_HANDLE },
    { id: `${attnId}__attn_low_split_2`, type: "research_default", source: split, sourceHandle: "tensor_2", target: vReshape, targetHandle: LAYER_STRIP_TARGET_HANDLE },
    { id: `${attnId}__attn_low_qk_0`, type: "research_default", source: qReshape, sourceHandle: LAYER_STRIP_SOURCE_HANDLE, target: qk, targetHandle: "tensor_1" },
    { id: `${attnId}__attn_low_qk_1`, type: "research_default", source: kReshape, sourceHandle: LAYER_STRIP_SOURCE_HANDLE, target: qk, targetHandle: "tensor_2" },
  ];
  const afterQk: Edge[] = causal
    ? [
        {
          id: `${attnId}__attn_low_causal_mask_in`,
          type: "research_default",
          source: qk,
          sourceHandle: LAYER_STRIP_SOURCE_HANDLE,
          target: causalMask,
          targetHandle: LAYER_STRIP_TARGET_HANDLE,
        },
        {
          id: `${attnId}__attn_low_softmax_in`,
          type: "research_default",
          source: causalMask,
          sourceHandle: LAYER_STRIP_SOURCE_HANDLE,
          target: softmax,
          targetHandle: LAYER_STRIP_TARGET_HANDLE,
        },
      ]
    : [
        {
          id: `${attnId}__attn_low_softmax_in`,
          type: "research_default",
          source: qk,
          sourceHandle: LAYER_STRIP_SOURCE_HANDLE,
          target: softmax,
          targetHandle: LAYER_STRIP_TARGET_HANDLE,
        },
      ];
  const tail: Edge[] = [
    { id: `${attnId}__attn_low_av_0`, type: "research_default", source: softmax, sourceHandle: LAYER_STRIP_SOURCE_HANDLE, target: av, targetHandle: "tensor_1" },
    { id: `${attnId}__attn_low_av_1`, type: "research_default", source: vReshape, sourceHandle: LAYER_STRIP_SOURCE_HANDLE, target: av, targetHandle: "tensor_2" },
    { id: `${attnId}__attn_low_out`, type: "research_default", source: av, sourceHandle: LAYER_STRIP_SOURCE_HANDLE, target: outReshape, targetHandle: LAYER_STRIP_TARGET_HANDLE },
    { id: `${attnId}__attn_low_linear_out`, type: "research_default", source: outReshape, sourceHandle: LAYER_STRIP_SOURCE_HANDLE, target: linearOut, targetHandle: LAYER_STRIP_TARGET_HANDLE },
  ];
  return [...base, ...afterQk, ...tail];
}

function attentionLowShellSubgraphIoEdges(attnId: string): Edge[] {
  const linearIn = attentionLowNodeId(attnId, SUFFIX_LINEAR_IN);
  const linearOut = attentionLowNodeId(attnId, SUFFIX_LINEAR_OUT);
  return [
    { id: `${attnId}__attn_low_shell_in`, type: COMBINED_SUBGRAPH_IO_EDGE_TYPE, source: attnId, target: linearIn, sourceHandle: "tensor_boundary", targetHandle: LAYER_STRIP_TARGET_HANDLE },
    { id: `${attnId}__attn_low_shell_out`, type: COMBINED_SUBGRAPH_IO_EDGE_TYPE, source: linearOut, target: attnId, sourceHandle: LAYER_STRIP_SOURCE_HANDLE, targetHandle: COMBINED_MODEL_RETURN_TARGET_HANDLE },
  ];
}

export function reconcileAttentionLowExpansion(nodes: Node[], edges: Edge[], attnId: string, d: AttentionOnlyModelNodeData): { nodes: Node[]; edges: Edge[] } {
  const full = { ...defaultAttentionOnlyModelData(), ...d };
  const shellIo = readNodeCanvasIoMode(full as Record<string, unknown>);
  const causal = attentionLowIsCausal(full);
  const embedDim = intChoices(full.embedDim, 2)[0];
  const requestedHeads = intChoices(full.numHeads, 1)[0];
  const nHeads = Math.max(1, Math.min(embedDim, requestedHeads));
  const headDim = Math.max(1, Math.floor(embedDim / nHeads));
  const attentionWidth = nHeads * headDim;

  const linearInData: LinearLayerNodeData & Record<string, unknown> = {
    ...defaultLinearLayerData(),
    inFeatures: embedDim,
    outFeatures: 3 * attentionWidth,
    bias: 1,
    seed: full.seed,
    ioMode: "input-output",
    levelMode: "high",
    [INSTANCE_TITLE_KEY]: "Linear",
  };
  const linearOutData: LinearLayerNodeData & Record<string, unknown> = {
    ...defaultLinearLayerData(),
    inFeatures: attentionWidth,
    outFeatures: embedDim,
    bias: 1,
    seed: full.seed,
    ioMode: "input-output",
    levelMode: "high",
    [INSTANCE_TITLE_KEY]: "Linear",
  };

  const cSoft = causal ? 5 : 4;
  const cAv = causal ? 6 : 5;
  const cOutR = causal ? 7 : 6;
  const cLinOut = causal ? 8 : 7;

  const addNodes: Node[] = [
    { id: attentionLowNodeId(attnId, SUFFIX_LINEAR_IN), type: "linear_layer", parentId: attnId, extent: "parent", position: childPos(0, 0), data: linearInData },
    { id: attentionLowNodeId(attnId, SUFFIX_SPLITTER), type: "tensor_splitter", parentId: attnId, extent: "parent", position: childPos(1, 0), data: { ioMode: "input-output", levelMode: "high", splitDimension: -1, numParts: 3, [INSTANCE_TITLE_KEY]: "Tensor splitter" } },
    { id: attentionLowNodeId(attnId, SUFFIX_RESHAPE_0), type: "reshape", parentId: attnId, extent: "parent", position: childPos(2, -1), data: { ioMode: "input-output", levelMode: "high", reshapeRule: `b t d -> b ${nHeads} t ${headDim}`, shapeHint: `[batch, ${nHeads}, tokens, ${headDim}]`, [INSTANCE_TITLE_KEY]: "Reshape" } },
    { id: attentionLowNodeId(attnId, SUFFIX_RESHAPE_1), type: "reshape", parentId: attnId, extent: "parent", position: childPos(2, 0), data: { ioMode: "input-output", levelMode: "high", reshapeRule: `b t d -> b ${nHeads} t ${headDim}`, shapeHint: `[batch, ${nHeads}, tokens, ${headDim}]`, [INSTANCE_TITLE_KEY]: "Reshape" } },
    { id: attentionLowNodeId(attnId, SUFFIX_RESHAPE_2), type: "reshape", parentId: attnId, extent: "parent", position: childPos(2, 1), data: { ioMode: "input-output", levelMode: "high", reshapeRule: `b t d -> b ${nHeads} t ${headDim}`, shapeHint: `[batch, ${nHeads}, tokens, ${headDim}]`, [INSTANCE_TITLE_KEY]: "Reshape" } },
    { id: attentionLowNodeId(attnId, SUFFIX_EINSUM_0), type: "einsum", parentId: attnId, extent: "parent", position: childPos(3, -1), data: { ioMode: "input-output", levelMode: "high", equation: "b h t d, b h s d -> b h t s", [INSTANCE_TITLE_KEY]: "Einsum" } },
    ...(causal
      ? [
          {
            id: attentionLowNodeId(attnId, SUFFIX_CAUSAL_MASK),
            type: "causal_mask" as const,
            parentId: attnId,
            extent: "parent" as const,
            position: childPos(4, -1),
            data: { ioMode: "input-output" as const, levelMode: "high" as const, diagonalOffset: 1, [INSTANCE_TITLE_KEY]: "Causal mask" },
          },
        ]
      : []),
    { id: attentionLowNodeId(attnId, SUFFIX_SOFTMAX), type: "softmax", parentId: attnId, extent: "parent", position: childPos(cSoft, -1), data: { ioMode: "input-output", levelMode: "high", dimension: -1, [INSTANCE_TITLE_KEY]: "Softmax" } },
    { id: attentionLowNodeId(attnId, SUFFIX_EINSUM_1), type: "einsum", parentId: attnId, extent: "parent", position: childPos(cAv, 0), data: { ioMode: "input-output", levelMode: "high", equation: "b h t s, b h s d -> b h t d", [INSTANCE_TITLE_KEY]: "Einsum" } },
    { id: attentionLowNodeId(attnId, SUFFIX_RESHAPE_3), type: "reshape", parentId: attnId, extent: "parent", position: childPos(cOutR, 0), data: { ioMode: "input-output", levelMode: "high", reshapeRule: `b ${nHeads} t ${headDim} -> b t ${attentionWidth}`, shapeHint: `[batch, tokens, ${attentionWidth}]`, [INSTANCE_TITLE_KEY]: "Reshape" } },
    { id: attentionLowNodeId(attnId, SUFFIX_LINEAR_OUT), type: "linear_layer", parentId: attnId, extent: "parent", position: childPos(cLinOut, 0), data: linearOutData },
  ];

  const withoutExpansion = nodes.filter((n) => !isAttentionLowExpansionNodeId(n.id, attnId));
  const shell = withoutExpansion.find((n) => n.id === attnId);
  const lastCol = causal ? 8 : 7;
  const prelimW = COMBINED_MODEL_CHILD_MIN_X + lastCol * ATTN_LOW_COLUMN_STEP_X + ATTN_LOW_LAST_COLUMN_CONTENT_W + COMBINED_SHELL_EDGE_PAD_X;
  const prelimH = COMBINED_MODEL_CHILD_MIN_Y + INNER_FALLBACK_H + COMBINED_SHELL_EDGE_PAD_Y;
  const mergedShell = shell && shell.type === "attention_only_model" ? { ...shell, style: { ...((shell.style ?? {}) as Record<string, unknown>), width: Math.max(236, prelimW), height: Math.max(COMBINED_MODEL_CHILD_MIN_Y + 20, prelimH) }, data: { ...((shell.data ?? {}) as Record<string, unknown>), __expandedFrame: { width: Math.max(236, prelimW), height: Math.max(COMBINED_MODEL_CHILD_MIN_Y + 20, prelimH) } } } : shell;
  const others = withoutExpansion.map((n) => (n.id === attnId && mergedShell ? mergedShell : n));
  const nextNodes = finalizeAttentionLowExpansionNodes([...others, ...addNodes]);

  const idSet = new Set(nextNodes.map((n) => n.id));
  let nextEdges = edges.filter((e) => idSet.has(e.source) && idSet.has(e.target));
  nextEdges = nextEdges.filter((e) => !isAttentionLowInternalEdge(e, attnId));
  nextEdges = nextEdges.filter((e) => !isAttentionLowShellSubgraphIoEdge(e, attnId));
  nextEdges = [...nextEdges, ...buildInternalEdges(attnId, causal)];
  if (shellIo === "input-output") nextEdges = [...nextEdges, ...attentionLowShellSubgraphIoEdges(attnId)];
  return { nodes: nextNodes, edges: nextEdges };
}

export function augmentNodeRemovesWithAttentionLowExpansion(changes: NodeChange[], nodes: Node[]): NodeChange[] {
  const extra: NodeChange[] = [];
  const removing = new Set<string>();
  for (const ch of changes) if (ch.type === "remove") removing.add(ch.id);
  for (const ch of changes) {
    if (ch.type !== "remove") continue;
    const n = nodes.find((x) => x.id === ch.id);
    if (n?.type !== "attention_only_model") continue;
    for (const x of nodes) {
      if (removing.has(x.id)) continue;
      if (isAttentionLowExpansionNodeId(x.id, ch.id) && (x.parentId === ch.id || x.parentId == null || x.parentId === "")) {
        removing.add(x.id);
        extra.push({ type: "remove", id: x.id });
      }
    }
  }
  return extra.length ? [...changes, ...extra] : changes;
}
