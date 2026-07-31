import type { Edge, Node } from "@xyflow/react";
import { SHAPE_ATOMIC_LAYER_TYPES, SHAPE_FULL_MODEL_TYPES } from "./canvasShapeSupport";
import type { FakeTensorDtype, FakeTensorNodeData } from "../components/nodes/fakeTensorDefaults";
import type { TensorConstantNodeData } from "../components/nodes/tensorConstantDefaults";
import type { TensorLinspaceNodeData } from "../components/nodes/tensorLinspaceDefaults";
import type { LinearLayerNodeData } from "../components/nodes/linearLayerDefaults";
import type { EmbeddingLayerNodeData } from "../components/nodes/embeddingLayerDefaults";
import type { UnembeddingLayerNodeData } from "../components/nodes/unembeddingLayerDefaults";
import type { LayerNormLayerNodeData } from "../components/nodes/layerNormLayerDefaults";
import type { AbsolutePosEmbedLayerNodeData } from "../components/nodes/absolutePosEmbedLayerDefaults";
import type { RotaryEmbedLayerNodeData } from "../components/nodes/rotaryEmbedLayerDefaults";
import type { DimensionPermutatorNodeData } from "../components/nodes/dimensionPermutatorDefaults";
import type { StatisticsNodeData } from "../components/nodes/statisticsDefaults";
import type { Statistics2NodeData } from "../components/nodes/statistics2Defaults";
import { clampBasicCalculatorInputCount } from "../components/nodes/basicCalculatorDefaults";
import type { PcaNodeData } from "../components/nodes/pcaDefaults";
import type { SvdNodeData } from "../components/nodes/svdDefaults";
import type { TensorSlicingNodeData } from "../components/nodes/tensorSlicingDefaults";
import type { FlattenNodeData } from "../components/nodes/flattenDefaults";
import { readFlattenExceptDim } from "../components/nodes/flattenDefaults";
import { intChoices } from "../components/nodes/multiValueUtils";
import { readNodeCanvasIoMode } from "./nodeCanvasIoMode";
import {
  COMBINED_MODEL_RETURN_TARGET_HANDLE,
  LAYER_STRIP_SOURCE_HANDLE,
  LAYER_STRIP_TARGET_HANDLE,
} from "./layerStripHandles";
import { normalizePermutation } from "./tensorPermute";
import { broadcastShapesOnly } from "./tensorBroadcastAdd";
import { inferBinaryOutputShapeSafe, inferSingleOutputShapeFromShape } from "./einsumCustom";
import { inferTensorSliceShape, normalizeSlices } from "./tensorSlice";

export type TensorSig = { shape: number[]; dtype: FakeTensorDtype };

const ATOMIC_LAYER_TYPES = SHAPE_ATOMIC_LAYER_TYPES;

const FULL_MODEL_IO_TYPES = SHAPE_FULL_MODEL_TYPES;

const TENSOR_VIZ_TYPES = new Set([
  "tensor_viz_general",
  "tensor_viz_1d",
  "tensor_viz_2d",
]);

function shapeFmt(shape: number[]): string {
  return `[${shape.join(", ")}]`;
}

function sigEqual(a: TensorSig | undefined, b: TensorSig | undefined): boolean {
  if (!a || !b) return false;
  if (a.dtype !== b.dtype) return false;
  if (a.shape.length !== b.shape.length) return false;
  return a.shape.every((v, i) => v === b.shape[i]!);
}

function normTargetHandle(nodeType: string | undefined | null, h: string | null | undefined): string {
  const v = (h ?? "").trim();
  if (v) return v;
  if (nodeType && ATOMIC_LAYER_TYPES.has(nodeType)) return LAYER_STRIP_TARGET_HANDLE;
  return "tensor";
}

function normSourceHandle(_node: Node | undefined, h: string | null | undefined): string {
  const v = (h ?? "").trim();
  return v || "tensor";
}

function outKey(nodeId: string, sourceHandle: string): string {
  return `${nodeId}\x1f${sourceHandle}`;
}

function readFakeTensorSig(n: Node): TensorSig | null {
  if (n.type !== "fake_tensor") return null;
  const d = (n.data ?? {}) as Partial<FakeTensorNodeData>;
  const sh = d.shape;
  if (!Array.isArray(sh) || sh.length === 0) return null;
  const shape = sh.map((x) => Number(x));
  if (shape.some((x) => !Number.isFinite(x) || !Number.isInteger(x) || x < 1)) return null;
  const dtype: FakeTensorDtype = d.dtype === "long" ? "long" : "float";
  return { shape, dtype };
}

function readTensorConstantSig(n: Node): TensorSig | null {
  if (n.type !== "tensor_constant") return null;
  const d = (n.data ?? {}) as Partial<TensorConstantNodeData>;
  const ot = d.outputTensor;
  if (ot?.shape?.length) {
    const shape = ot.shape.map((x) => Number(x));
    if (shape.every((x) => Number.isFinite(x) && Number.isInteger(x) && x >= 1)) {
      return { shape, dtype: "float" };
    }
  }
  const sh = d.shape;
  if (Array.isArray(sh) && sh.length) {
    const shape = sh.map((x) => Number(x));
    if (shape.every((x) => Number.isFinite(x) && Number.isInteger(x) && x >= 1)) {
      return { shape, dtype: "float" };
    }
  }
  return null;
}

function readTensorLinspaceSig(n: Node): TensorSig | null {
  if (n.type !== "tensor_linspace") return null;
  const d = (n.data ?? {}) as Partial<TensorLinspaceNodeData>;
  const ot = d.outputTensor;
  if (ot?.shape?.length === 1) {
    const shape = ot.shape.map((x) => Number(x));
    if (shape.every((x) => Number.isFinite(x) && Number.isInteger(x) && x >= 1)) {
      return { shape, dtype: "float" };
    }
  }
  const rawNum = d.numPoints;
  const num = Array.isArray(rawNum) ? Number(rawNum[0]) : Number(rawNum);
  if (Number.isInteger(num) && num >= 1) {
    return { shape: [num], dtype: "float" };
  }
  return null;
}

function gatherIncoming(
  n: Node,
  edges: Edge[],
  outMap: Map<string, TensorSig>,
  nodesById: Map<string, Node>,
): { byTarget: Map<string, TensorSig[]>; conflict: boolean } {
  const byTarget = new Map<string, TensorSig[]>();
  for (const e of edges) {
    if (e.target !== n.id) continue;
    const srcNode = nodesById.get(e.source);
    const sh = normSourceHandle(srcNode, e.sourceHandle);
    const sig = outMap.get(outKey(e.source, sh));
    if (!sig) continue;
    const th = normTargetHandle(n.type, e.targetHandle);
    const arr = byTarget.get(th) ?? [];
    arr.push(sig);
    byTarget.set(th, arr);
  }
  let conflict = false;
  for (const [, sigs] of byTarget) {
    if (sigs.length < 2) continue;
    const s0 = sigs[0]!;
    for (let i = 1; i < sigs.length; i++) {
      if (!sigEqual(s0, sigs[i]!)) {
        conflict = true;
        break;
      }
    }
  }
  return { byTarget, conflict };
}

function firstSig(byTarget: Map<string, TensorSig[]>, th: string): TensorSig | undefined {
  return byTarget.get(th)?.[0];
}

function linearLikeOut(
  d: { inFeatures: unknown; outFeatures: unknown },
  inSig: TensorSig,
  outHandle: string,
): { outs: Map<string, TensorSig>; err: boolean } {
  const outs = new Map<string, TensorSig>();
  const inF = intChoices(d.inFeatures, 1)[0]!;
  const outF = intChoices(d.outFeatures, 1)[0]!;
  const sh = inSig.shape;
  if (sh.length < 1) return { outs, err: true };
  if (sh[sh.length - 1]! !== inF) return { outs, err: true };
  const next = [...sh.slice(0, -1), outF];
  outs.set(outHandle, { shape: next, dtype: inSig.dtype });
  return { outs, err: false };
}

function embeddingOut(d: EmbeddingLayerNodeData, inSig: TensorSig, outHandle: string): { outs: Map<string, TensorSig>; err: boolean } {
  const outs = new Map<string, TensorSig>();
  const emb = intChoices(d.embeddingDim, 1)[0]!;
  const nIdx = intChoices(d.numIndexColumns, 1)[0]!;
  const sh = inSig.shape;
  if (sh.length < nIdx) return { outs, err: true };
  const next = [...sh.slice(0, sh.length - nIdx), emb];
  outs.set(outHandle, { shape: next, dtype: inSig.dtype });
  return { outs, err: false };
}

function unembeddingOut(d: UnembeddingLayerNodeData, inSig: TensorSig, outHandle: string): { outs: Map<string, TensorSig>; err: boolean } {
  const outs = new Map<string, TensorSig>();
  const inF = intChoices(d.inFeatures, 1)[0]!;
  const outF = intChoices(d.outFeatures, 1)[0]!;
  const sh = inSig.shape;
  if (sh.length < 1 || sh[sh.length - 1]! !== inF) return { outs, err: true };
  const next = [...sh.slice(0, -1), outF];
  outs.set(outHandle, { shape: next, dtype: inSig.dtype });
  return { outs, err: false };
}

function layerNormOut(d: LayerNormLayerNodeData, inSig: TensorSig, outHandle: string): { outs: Map<string, TensorSig>; err: boolean } {
  const outs = new Map<string, TensorSig>();
  const nrm = intChoices(d.normalizedShape, 1)[0]!;
  const sh = inSig.shape;
  if (sh.length < 1 || sh[sh.length - 1]! !== nrm) return { outs, err: true };
  outs.set(outHandle, { ...inSig });
  return { outs, err: false };
}

function absolutePosEmbedOut(
  d: AbsolutePosEmbedLayerNodeData,
  inSig: TensorSig,
  outHandle: string,
): { outs: Map<string, TensorSig>; err: boolean } {
  const outs = new Map<string, TensorSig>();
  const ed = intChoices(d.embeddingDim, 1)[0]!;
  const mxl = intChoices(d.maxSeqLen, 1)[0]!;
  const sh = inSig.shape;
  if (sh.length < 2 || sh[sh.length - 1]! !== ed) return { outs, err: true };
  if (sh.length >= 3) {
    const t = sh[sh.length - 2]!;
    if (t > mxl) return { outs, err: true };
  }
  outs.set(outHandle, { shape: [...sh], dtype: "float" });
  return { outs, err: false };
}

function rotaryEmbedOut(
  d: RotaryEmbedLayerNodeData,
  inSig: TensorSig,
  outHandle: string,
): { outs: Map<string, TensorSig>; err: boolean } {
  const outs = new Map<string, TensorSig>();
  const rd = intChoices(d.rotaryDim, 1)[0]!;
  const sh = inSig.shape;
  if (sh.length < 2 || sh[sh.length - 1]! !== rd) return { outs, err: true };
  if (rd % 2 !== 0) return { outs, err: true };
  outs.set(outHandle, { shape: [...sh], dtype: "float" });
  return { outs, err: false };
}

function permutatorOut(d: DimensionPermutatorNodeData, inSig: TensorSig, outHandle: string): { outs: Map<string, TensorSig>; err: boolean } {
  const outs = new Map<string, TensorSig>();
  const rank = inSig.shape.length;
  const axes = normalizePermutation(Array.isArray(d.axes) ? d.axes : [], rank);
  if (rank > 0 && axes.length === rank) {
    try {
      const next = axes.map((i) => inSig.shape[i]!);
      outs.set(outHandle, { shape: next, dtype: inSig.dtype });
      return { outs, err: false };
    } catch {
      return { outs, err: true };
    }
  }
  outs.set(outHandle, { ...inSig });
  return { outs, err: false };
}

function inferFlattenOutputShape(shape: number[], exceptDim: number | null): number[] | null {
  const rank = shape.length;
  if (rank < 1) return null;
  if (exceptDim == null) {
    let p = 1;
    for (const s of shape) p *= s;
    return [p];
  }
  let ax = exceptDim;
  if (ax < 0) ax = rank + ax;
  if (!Number.isInteger(ax) || ax < 0 || ax >= rank) return null;
  let prod = 1;
  for (let i = 0; i < rank; i += 1) {
    if (i !== ax) prod *= shape[i]!;
  }
  return [shape[ax]!, prod];
}

function statisticsOut(d: StatisticsNodeData, inSig: TensorSig, outHandle: string): { outs: Map<string, TensorSig>; err: boolean } {
  const outs = new Map<string, TensorSig>();
  const expr = (d.einsumSubscripts ?? "").trim().replace(/\s+/g, "");
  if (!expr) {
    outs.set(outHandle, { ...inSig });
    return { outs, err: false };
  }
  try {
    const next = inferSingleOutputShapeFromShape(inSig.shape, expr);
    outs.set(outHandle, { shape: next, dtype: inSig.dtype });
    return { outs, err: false };
  } catch {
    outs.set(outHandle, { ...inSig });
    return { outs, err: false };
  }
}

function pcaOuts(d: PcaNodeData, inSig: TensorSig): Map<string, TensorSig> {
  const outs = new Map<string, TensorSig>();
  const k = Math.max(1, Math.floor(Number.isFinite(d.nComponents) ? Number(d.nComponents) : 2));
  const sh = inSig.shape;
  if (sh.length < 1) return outs;
  const F = sh[sh.length - 1]!;
  const kEff = Math.min(k, F);
  const prefix = sh.slice(0, -1);
  outs.set("transformed_tensor", { shape: [...prefix, kEff], dtype: inSig.dtype });
  outs.set("principal_components", { shape: [F, kEff], dtype: inSig.dtype });
  outs.set("explained_variance_ratio", { shape: [kEff], dtype: inSig.dtype });
  return outs;
}

function svdOuts(_d: SvdNodeData, inSig: TensorSig): Map<string, TensorSig> {
  const outs = new Map<string, TensorSig>();
  const sh = inSig.shape;
  if (sh.length < 1) return outs;
  const F = sh[sh.length - 1]!;
  const prefix = sh.slice(0, -1);
  let m = 1;
  for (const p of prefix) m *= Math.max(1, Math.floor(p));
  const kEff = Math.min(Math.max(1, m), F);
  outs.set("u", { shape: [...prefix, kEff], dtype: inSig.dtype });
  outs.set("s", { shape: [kEff], dtype: inSig.dtype });
  outs.set("v", { shape: [kEff, F], dtype: inSig.dtype });
  return outs;
}

function computeNodeOutputs(n: Node, byTarget: Map<string, TensorSig[]>): { outs: Map<string, TensorSig>; err: boolean } {
  const outs = new Map<string, TensorSig>();
  const t = String(n.type);
  const d = (n.data ?? {}) as Record<string, unknown>;

  if (t === "fake_tensor") {
    const s = readFakeTensorSig(n);
    if (s) outs.set("tensor", s);
    return { outs, err: false };
  }

  if (t === "tensor_constant") {
    const s = readTensorConstantSig(n);
    if (s) outs.set("tensor", s);
    return { outs, err: false };
  }

  if (t === "tensor_linspace") {
    const s = readTensorLinspaceSig(n);
    if (s) outs.set("tensor", s);
    return { outs, err: false };
  }

  if (t === "tensor_add") {
    const a = firstSig(byTarget, "tensor_1");
    const b = firstSig(byTarget, "tensor_2");
    if (!a || !b) return { outs, err: false };
    try {
      const shape = broadcastShapesOnly(a.shape, b.shape);
      outs.set("tensor", { shape, dtype: a.dtype });
      return { outs, err: false };
    } catch {
      return { outs, err: true };
    }
  }

  if (t === "basic_calculator") {
    const ic = clampBasicCalculatorInputCount((d as { inputCount?: unknown }).inputCount);
    let dtype: FakeTensorDtype | null = null;
    for (let i = 1; i <= ic; i++) {
      const inn = firstSig(byTarget, `tensor_${i}`);
      if (!inn) return { outs, err: false };
      const prod = inn.shape.reduce((a, b) => a * b, 1);
      if (prod !== 1) return { outs, err: true };
      dtype = dtype ?? inn.dtype;
    }
    outs.set("tensor", { shape: [], dtype: dtype ?? "float32" });
    return { outs, err: false };
  }

  if (t === "tensor_stack") {
    const ins: TensorSig[] = [];
    for (const [k, vals] of byTarget.entries()) {
      if (!k.startsWith("tensor_")) continue;
      if (vals[0]) ins.push(vals[0]);
    }
    if (ins.length === 0) return { outs, err: false };
    const key = (s: TensorSig) => `${s.dtype}:${s.shape.join("x")}`;
    if (new Set(ins.map(key)).size > 1) return { outs, err: true };
    outs.set("tensor", { shape: [ins.length, ...ins[0]!.shape], dtype: ins[0]!.dtype });
    return { outs, err: false };
  }

  if (t === "tensor_concat") {
    const ins: TensorSig[] = [];
    for (const [k, vals] of byTarget.entries()) {
      if (!k.startsWith("tensor_")) continue;
      if (vals[0]) ins.push(vals[0]);
    }
    if (ins.length === 0) return { outs, err: false };
    const rank = ins[0]!.shape.length;
    const rawDim = Number((d as { concatDimension?: unknown }).concatDimension ?? 0);
    const dim = rawDim < 0 ? rank + Math.floor(rawDim) : Math.floor(rawDim);
    if (!Number.isFinite(dim) || dim < 0 || dim >= rank) return { outs, err: true };
    const base = ins[0]!;
    let sum = base.shape[dim]!;
    for (let i = 1; i < ins.length; i += 1) {
      const s = ins[i]!;
      if (s.shape.length !== rank) return { outs, err: true };
      for (let ax = 0; ax < rank; ax += 1) {
        if (ax === dim) continue;
        if (s.shape[ax] !== base.shape[ax]) return { outs, err: true };
      }
      sum += s.shape[dim]!;
    }
    const outShape = [...base.shape];
    outShape[dim] = sum;
    outs.set("tensor", { shape: outShape, dtype: base.dtype });
    return { outs, err: false };
  }

  if (t === "statistics2") {
    const a = firstSig(byTarget, "tensor_1");
    const b = firstSig(byTarget, "tensor_2");
    if (!a || !b) return { outs, err: false };
    const expr = ((d as Statistics2NodeData).einsumSubscripts ?? "").trim().replace(/\s+/g, "");
    if (!expr) {
      if (!sigEqual(a, b)) return { outs, err: true };
      outs.set("tensor", { ...a });
      return { outs, err: false };
    }
    const outShape = inferBinaryOutputShapeSafe(a.shape, b.shape, expr);
    if (!outShape) return { outs, err: true };
    outs.set("tensor", { shape: outShape, dtype: a.dtype });
    return { outs, err: false };
  }

  if (t === "effective_rank") {
    const inn = firstSig(byTarget, "tensor");
    if (!inn) return { outs, err: false };
    outs.set("tensor", { shape: [], dtype: inn.dtype });
    return { outs, err: false };
  }

  if (t === "series_endpoint_gap") {
    const inn = firstSig(byTarget, "tensor");
    if (!inn) return { outs, err: false };
    if (inn.shape.length !== 1) return { outs, err: true };
    outs.set("tensor", { shape: [], dtype: inn.dtype });
    return { outs, err: false };
  }

  if (t === "smoothing_curve" || t === "derivative_curve") {
    const inn = firstSig(byTarget, "tensor");
    if (!inn) return { outs, err: false };
    if (inn.shape.length !== 1) return { outs, err: true };
    outs.set("tensor", { ...inn });
    return { outs, err: false };
  }

  if (t === "dimension_permutator") {
    const inn = firstSig(byTarget, "tensor_in");
    if (!inn) return { outs, err: false };
    const r = permutatorOut(d as DimensionPermutatorNodeData, inn, "tensor_out");
    return r;
  }

  if (t === "tensor_slicing") {
    const inn = firstSig(byTarget, "tensor");
    if (!inn) return { outs, err: false };
    const next = inferTensorSliceShape(inn.shape, normalizeSlices((d as TensorSlicingNodeData).slices ?? []));
    if (!next) return { outs, err: true };
    outs.set("tensor", { shape: next, dtype: inn.dtype });
    return { outs, err: false };
  }

  if (t === "elementwise_transform") {
    const inn = firstSig(byTarget, "tensor");
    if (!inn) return { outs, err: false };
    outs.set("tensor", { ...inn });
    return { outs, err: false };
  }

  if (t === "flatten") {
    const io = readNodeCanvasIoMode(d);
    const outH = io === "model" ? "tensor" : LAYER_STRIP_SOURCE_HANDLE;
    if (io === "model") {
      return { outs, err: false };
    }
    const inn = firstSig(byTarget, LAYER_STRIP_TARGET_HANDLE) ?? firstSig(byTarget, "tensor");
    if (!inn) return { outs, err: false };
    const ex = readFlattenExceptDim((d as FlattenNodeData).exceptDim);
    const next = inferFlattenOutputShape(inn.shape, ex);
    if (!next) return { outs, err: true };
    outs.set(outH, { shape: next, dtype: inn.dtype });
    return { outs, err: false };
  }

  if (t === "statistics") {
    const inn = firstSig(byTarget, "tensor");
    if (!inn) return { outs, err: false };
    return statisticsOut(d as StatisticsNodeData, inn, "tensor");
  }

  if (t === "pca") {
    const inn = firstSig(byTarget, "tensor");
    if (!inn) return { outs, err: false };
    return { outs: pcaOuts(d as PcaNodeData, inn), err: false };
  }

  if (t === "svd") {
    const inn = firstSig(byTarget, "tensor");
    if (!inn) return { outs, err: false };
    return { outs: svdOuts(d as SvdNodeData, inn), err: false };
  }

  if (t === "combined_model" && readNodeCanvasIoMode(d) === "input-output") {
    const tin = firstSig(byTarget, LAYER_STRIP_TARGET_HANDLE) ?? firstSig(byTarget, "tensor");
    const tret = firstSig(byTarget, COMBINED_MODEL_RETURN_TARGET_HANDLE);
    if (tin) outs.set("tensor_boundary", tin);
    const stripOut = tret ?? tin;
    if (stripOut) outs.set(LAYER_STRIP_SOURCE_HANDLE, stripOut);
    return { outs, err: false };
  }

  if (ATOMIC_LAYER_TYPES.has(t)) {
    const io = readNodeCanvasIoMode(d);
    const outH = io === "model" ? "tensor" : LAYER_STRIP_SOURCE_HANDLE;
    const inH = io === "model" ? null : LAYER_STRIP_TARGET_HANDLE;
    if (io === "model") {
      /* No tensor input on canvas in model mode. */
      return { outs, err: false };
    }
    const inn = firstSig(byTarget, inH!) ?? firstSig(byTarget, "tensor");
    if (!inn) return { outs, err: false };
    if (t === "linear_layer") {
      const r = linearLikeOut(d as LinearLayerNodeData, inn, outH);
      return r;
    }
    if (t === "activation_layer") {
      outs.set(outH, { ...inn });
      return { outs, err: false };
    }
    if (t === "layer_norm_layer") {
      return layerNormOut(d as LayerNormLayerNodeData, inn, outH);
    }
    if (t === "rms_norm_layer") {
      return layerNormOut(d as LayerNormLayerNodeData, inn, outH);
    }
    if (t === "embedding_layer") {
      return embeddingOut(d as EmbeddingLayerNodeData, inn, outH);
    }
    if (t === "unembedding_layer") {
      return unembeddingOut(d as UnembeddingLayerNodeData, inn, outH);
    }
    if (t === "absolute_pos_embed_layer") {
      return absolutePosEmbedOut(d as AbsolutePosEmbedLayerNodeData, inn, outH);
    }
    if (t === "rotary_embed_layer") {
      return rotaryEmbedOut(d as RotaryEmbedLayerNodeData, inn, outH);
    }
    if (t === "local_mixing_layer") {
      outs.set(outH, { ...inn });
      return { outs, err: false };
    }
  }

  if (FULL_MODEL_IO_TYPES.has(t) && readNodeCanvasIoMode(d) === "input-output") {
    const inn = firstSig(byTarget, LAYER_STRIP_TARGET_HANDLE) ?? firstSig(byTarget, "tensor");
    if (inn) outs.set(LAYER_STRIP_SOURCE_HANDLE, { ...inn });
    return { outs, err: false };
  }

  if (TENSOR_VIZ_TYPES.has(t)) {
    const inn = firstSig(byTarget, "tensor");
    if (inn) outs.set("out_tensor", { ...inn });
    return { outs, err: false };
  }

  return { outs, err: false };
}

function collectOutgoingSourceHandles(n: Node, edges: Edge[]): Set<string> {
  const hs = new Set<string>();
  for (const e of edges) {
    if (e.source !== n.id) continue;
    hs.add(normSourceHandle(n, e.sourceHandle));
  }
  if (n.type === "fake_tensor" || n.type === "tensor_constant" || n.type === "tensor_linspace") hs.add("tensor");
  if (
    n.type === "tensor_add" ||
    n.type === "basic_calculator" ||
    n.type === "tensor_stack" ||
    n.type === "tensor_concat" ||
    n.type === "effective_rank" ||
    n.type === "series_endpoint_gap" ||
    n.type === "smoothing_curve" ||
    n.type === "derivative_curve" ||
    n.type === "statistics"
  ) {
    hs.add("tensor");
  }
  if (n.type === "statistics2") hs.add("tensor");
  if (n.type === "dimension_permutator") hs.add("tensor_out");
  if (n.type === "tensor_slicing") hs.add("tensor");
  if (n.type === "elementwise_transform") hs.add("tensor");
  if (n.type === "flatten") {
    const io = readNodeCanvasIoMode((n.data ?? {}) as Record<string, unknown>);
    if (io === "input-output") hs.add(LAYER_STRIP_SOURCE_HANDLE);
    else hs.add("tensor");
  }
  if (ATOMIC_LAYER_TYPES.has(String(n.type)) || FULL_MODEL_IO_TYPES.has(String(n.type))) {
    const io = readNodeCanvasIoMode((n.data ?? {}) as Record<string, unknown>);
    if (io === "input-output") hs.add(LAYER_STRIP_SOURCE_HANDLE);
    else hs.add("tensor");
  }
  if (n.type === "combined_model" && readNodeCanvasIoMode((n.data ?? {}) as Record<string, unknown>) === "input-output") {
    hs.add("tensor_boundary");
    hs.add(LAYER_STRIP_SOURCE_HANDLE);
  }
  if (TENSOR_VIZ_TYPES.has(String(n.type))) hs.add("out_tensor");
  if (n.type === "pca") {
    hs.add("transformed_tensor");
    hs.add("principal_components");
    hs.add("explained_variance_ratio");
  }
  if (n.type === "svd") {
    hs.add("u");
    hs.add("s");
    hs.add("v");
  }
  return hs;
}

function genericPassthrough(
  n: Node,
  byTarget: Map<string, TensorSig[]>,
  edges: Edge[],
): { outs: Map<string, TensorSig>; err: boolean } {
  const outs = new Map<string, TensorSig>();
  const sigs: TensorSig[] = [];
  for (const [, arr] of byTarget) {
    if (arr[0]) sigs.push(arr[0]!);
  }
  if (sigs.length === 0) return { outs, err: false };
  const key = (s: TensorSig) => `${s.dtype}:${s.shape.join("x")}`;
  if (new Set(sigs.map(key)).size > 1) return { outs, err: true };
  const sig = sigs[0]!;
  for (const h of collectOutgoingSourceHandles(n, edges)) {
    outs.set(h, { ...sig });
  }
  return { outs, err: false };
}

function skipGenericWhenEmpty(t: string): boolean {
  if (
    t === "tensor_add" ||
    t === "basic_calculator" ||
    t === "tensor_stack" ||
    t === "tensor_concat" ||
    t === "statistics2" ||
    t === "dimension_permutator" ||
    t === "tensor_slicing" ||
    t === "elementwise_transform" ||
    t === "flatten" ||
    t === "statistics" ||
    t === "pca" ||
    t === "svd" ||
    t === "effective_rank" ||
    t === "series_endpoint_gap" ||
    t === "smoothing_curve" ||
    t === "derivative_curve" ||
    t === "fake_tensor" ||
    t === "tensor_constant" ||
    t === "tensor_linspace" ||
    t === "combined_model"
  ) {
    return true;
  }
  return ATOMIC_LAYER_TYPES.has(t) || FULL_MODEL_IO_TYPES.has(t) || TENSOR_VIZ_TYPES.has(t);
}

function mergeCompute(
  n: Node,
  byTarget: Map<string, TensorSig[]>,
  edges: Edge[],
): { outs: Map<string, TensorSig>; err: boolean } {
  const primary = computeNodeOutputs(n, byTarget);
  if (primary.err) return primary;
  if (primary.outs.size > 0) return primary;
  const t = String(n.type);
  if (skipGenericWhenEmpty(t)) return primary;
  return genericPassthrough(n, byTarget, edges);
}

/**
 * Static shape propagation from a `fake_tensor` source. Dtypes are tracked but mismatches do not error
 * (only shape conflicts and known incompatible ops).
 */
export function runFakeTensorShapeCheck(fakeNodeId: string, nodes: Node[], edges: Edge[]): { errorNodeIds: string[]; edgeIdToLabel: Record<string, string> } {
  const fake = nodes.find((x) => x.id === fakeNodeId && x.type === "fake_tensor");
  const start = fake ? readFakeTensorSig(fake) : null;
  const errorSet = new Set<string>();
  const outMap = new Map<string, TensorSig>();
  const nodesById = new Map(nodes.map((n) => [n.id, n]));

  if (!fake || !start) {
    if (fake) errorSet.add(fakeNodeId);
    return { errorNodeIds: [...errorSet], edgeIdToLabel: {} };
  }

  outMap.set(outKey(fakeNodeId, "tensor"), start);

  const maxIter = Math.max(80, nodes.length + edges.length) * 6;
  for (let it = 0; it < maxIter; it++) {
    let changed = false;
    const next = new Map(outMap);

    for (const n of nodes) {
      const { byTarget, conflict } = gatherIncoming(n, edges, next, nodesById);
      if (conflict) {
        errorSet.add(n.id);
        continue;
      }

      const { outs, err } = mergeCompute(n, byTarget, edges);
      if (err) errorSet.add(n.id);

      for (const [handle, sig] of outs) {
        const k = outKey(n.id, handle);
        if (!sigEqual(next.get(k), sig)) {
          next.set(k, sig);
          changed = true;
        }
      }
    }

    if (!changed) break;
    for (const [k, v] of next) outMap.set(k, v);
  }

  const edgeIdToLabel: Record<string, string> = {};
  for (const e of edges) {
    const srcNode = nodesById.get(e.source);
    const sh = normSourceHandle(srcNode, e.sourceHandle);
    const sig = outMap.get(outKey(e.source, sh));
    if (sig) edgeIdToLabel[e.id] = shapeFmt(sig.shape);
  }

  return { errorNodeIds: [...errorSet], edgeIdToLabel };
}
