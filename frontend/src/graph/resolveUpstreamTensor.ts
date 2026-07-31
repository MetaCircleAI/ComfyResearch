import type { Edge, Node } from "@xyflow/react";
import { SHAPE_ATOMIC_LAYER_TYPES, SHAPE_FULL_MODEL_TYPES } from "./canvasShapeSupport";
import { getObservableVizVariant, observableVizAllowsTensorVizChain } from "./observableVizVariant";
import { readActivationManifest, readActivationRunId } from "./activationNodeData";
import type { ActivationNodeData, CollectedActivationTensor } from "../components/nodes/activationDefaults";
import type { ObservableVizUserNodeData } from "../components/nodes/observableVizUserDefaults";
import type { ObservableVizWeightL1NodeData } from "../components/nodes/observableVizWeightL1Defaults";
import type { ObservableVizWeightL2NodeData } from "../components/nodes/observableVizWeightL2Defaults";
import type { ObservableVizReluNonlinearNodeData } from "../components/nodes/observableVizReluNonlinearDefaults";
import type { PcaNodeData } from "../components/nodes/pcaDefaults";
import type { SvdNodeData } from "../components/nodes/svdDefaults";
import type { StatisticsNodeData } from "../components/nodes/statisticsDefaults";
import type { Statistics2NodeData } from "../components/nodes/statistics2Defaults";
import type { PredictionNodeData } from "../components/nodes/predictionDefaults";
import type { TensorAddNodeData } from "../components/nodes/tensorAddDefaults";
import type { BasicCalculatorNodeData } from "../components/nodes/basicCalculatorDefaults";
import type { TensorConstantNodeData } from "../components/nodes/tensorConstantDefaults";
import type { TensorLinspaceNodeData } from "../components/nodes/tensorLinspaceDefaults";
import type { TrainerNodeData } from "../components/nodes/trainerDefaults";
import type { TrainingVisualizationNodeData } from "../components/nodes/trainingVisualizationDefaults";
import type { EffectiveRankNodeData } from "../components/nodes/effectiveRankDefaults";
import type { SeriesEndpointGapNodeData } from "../components/nodes/seriesEndpointGapDefaults";
import type { SmoothingCurveNodeData } from "../components/nodes/smoothingCurveDefaults";
import type { DerivativeCurveNodeData } from "../components/nodes/derivativeCurveDefaults";
import type { ModelWeightTensorsNodeData } from "../components/nodes/modelWeightTensorsDefaults";
import { TOY_LANGUAGE_DATASET_KINDS } from "../components/nodes/toyLanguageDatasetDefaults";
import {
  tensorSelectorOutputIndexFromSourceHandle,
  type ActivationTensorCache,
  type TensorSelectorNodeData,
} from "../components/nodes/tensorSelectorDefaults";
import type { DimensionPermutatorNodeData } from "../components/nodes/dimensionPermutatorDefaults";
import type { TensorSlicingNodeData } from "../components/nodes/tensorSlicingDefaults";
import type { ElementwiseTransformNodeData } from "../components/nodes/elementwiseTransformDefaults";
import type { LinearLayerNodeData } from "../components/nodes/linearLayerDefaults";
import type { LayerNormLayerNodeData } from "../components/nodes/layerNormLayerDefaults";
import type { MlpModelNodeData } from "../components/nodes/mlpModelDefaults";
import type { KanModelNodeData } from "../components/nodes/kanModelDefaults";
import type { ResidualLnModelNodeData } from "../components/nodes/residualLnModelDefaults";
import type { AttentionOnlyModelNodeData } from "../components/nodes/attentionOnlyModelDefaults";
import type { AlternativeArchTokenLmNodeData } from "../components/nodes/alternativeArchModelDefaults";
import type { DiffusionScoreModelNodeData } from "../components/nodes/diffusionScoreModelDefaults";
import type { NumericTransformerModelNodeData } from "../components/nodes/numericTransformerModelDefaults";
import type { NumericHyenaModelNodeData } from "../components/nodes/numericHyenaModelDefaults";
import type { TransformerMultiTokenModelNodeData } from "../components/nodes/transformerMultiTokenModelDefaults";
import type { TransformerTokenModelNodeData } from "../components/nodes/transformerTokenModelDefaults";
import type { EmbeddingLayerNodeData } from "../components/nodes/embeddingLayerDefaults";
import type { UnembeddingLayerNodeData } from "../components/nodes/unembeddingLayerDefaults";
import type { AbsolutePosEmbedLayerNodeData } from "../components/nodes/absolutePosEmbedLayerDefaults";
import type { RotaryEmbedLayerNodeData } from "../components/nodes/rotaryEmbedLayerDefaults";
import type { LocalMixingLayerNodeData } from "../components/nodes/localMixingLayerDefaults";
import { parseCoordsFlat, type ProteinStructureDisplayerNodeData } from "../components/nodes/proteinStructureVizDefaults";
import { readFlattenExceptDim } from "../components/nodes/flattenDefaults";
import { intChoices } from "../components/nodes/multiValueUtils";
import { normalizePermutation, permuteRowMajor } from "./tensorPermute";
import { applyTensorSlicing, normalizeSlices } from "./tensorSlice";
import { tableVizTensorListChoices } from "./tableVizRegressor";
import { readNodeCanvasIoMode } from "./nodeCanvasIoMode";
import {
  COMBINED_MODEL_RETURN_TARGET_HANDLE,
  LAYER_STRIP_SOURCE_HANDLE,
  LAYER_STRIP_TARGET_HANDLE,
} from "./layerStripHandles";

export type FlowEdge = {
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
};

export type FlowNodeBare = {
  id: string;
  type: string;
  data?: unknown;
};

export type ResolvedOk = {
  kind: "ok";
  rank: number;
  shape: number[];
  values: number[];
  sourceSummary: string;
  /** When set (e.g. toy dataset word inspect), tensor reader shows this JSON instead of reshaped values. */
  textPreview?: string;
};

export type ResolvedNone = { kind: "none"; detail: string };

export type LazyTensorOp =
  | { kind: "slice"; slices: TensorSlicingNodeData["slices"] }
  | { kind: "permute"; axes: number[] }
  | { kind: "flatten"; exceptDim: number | null };

/** Server-side activation run; call {@link fetchActivationTensorAsOk} or `useHydratedResolved`. */
export type ResolvedLazyActivation = {
  kind: "lazy_activation";
  runId: string;
  repId: string;
  shape: number[];
  sourceSummary: string;
  ops?: LazyTensorOp[];
};

/** Server-side dataset materialization for dataset node train/test input/output tensors. */
export type ResolvedLazyDataset = {
  kind: "lazy_dataset";
  datasetNodeId: string;
  datasetNodeType: string;
  datasetData: unknown;
  graphNodes?: FlowNodeBare[];
  graphEdges?: FlowEdge[];
  split: "train" | "test";
  tensorKey: "input" | "output";
  sourceSummary: string;
  ops?: LazyTensorOp[];
};

export type Resolved = ResolvedOk | ResolvedNone | ResolvedLazyActivation | ResolvedLazyDataset;

function lazyOpsEqual(a: LazyTensorOp[] | undefined, b: LazyTensorOp[] | undefined): boolean {
  return JSON.stringify(a ?? []) === JSON.stringify(b ?? []);
}

/**
 * Reference-safe equality for `Resolved` values from {@link resolveUpstreamTensor}.
 * Use as the second argument to React Flow `useStore` so consumers like `useHydratedResolved`
 * do not see a “new” resolved object on unrelated store updates (which would cancel in-flight fetches).
 */
export function resolvedTensorEqual(a: Resolved, b: Resolved): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "none" && b.kind === "none") return a.detail === b.detail;
  if (a.kind === "lazy_activation" && b.kind === "lazy_activation") {
    return (
      a.runId === b.runId &&
      a.repId === b.repId &&
      a.sourceSummary === b.sourceSummary &&
      a.shape.length === b.shape.length &&
      a.shape.every((v, i) => v === b.shape[i]!) &&
      lazyOpsEqual(a.ops, b.ops)
    );
  }
  if (a.kind === "lazy_dataset" && b.kind === "lazy_dataset") {
    return (
      a.datasetNodeId === b.datasetNodeId &&
      a.datasetNodeType === b.datasetNodeType &&
      a.datasetData === b.datasetData &&
      a.graphNodes === b.graphNodes &&
      a.graphEdges === b.graphEdges &&
      a.split === b.split &&
      a.tensorKey === b.tensorKey &&
      a.sourceSummary === b.sourceSummary &&
      lazyOpsEqual(a.ops, b.ops)
    );
  }
  if (a.kind === "ok" && b.kind === "ok") {
    if (a.rank !== b.rank || a.sourceSummary !== b.sourceSummary) return false;
    if (a.shape.length !== b.shape.length) return false;
    for (let i = 0; i < a.shape.length; i++) {
      if (a.shape[i] !== b.shape[i]) return false;
    }
    if (a.values.length !== b.values.length) return false;
    const len = a.values.length;
    const fullCompare = a.rank === 0 || len <= 256;
    const n = fullCompare ? len : Math.min(8, len);
    for (let i = 0; i < n; i++) {
      if (a.values[i] !== b.values[i]) return false;
    }
    return true;
  }
  return false;
}

function appendLazyOp(inner: Resolved, op: LazyTensorOp): Resolved {
  if (inner.kind !== "lazy_activation" && inner.kind !== "lazy_dataset") return inner;
  const ops = [...(inner.ops ?? []), op];
  return { ...inner, ops };
}

export type TensorListChoice = { id: string; label: string };

const ATOMIC_LAYER_TYPES = SHAPE_ATOMIC_LAYER_TYPES;

const FULL_MODEL_CANVAS_TYPES = SHAPE_FULL_MODEL_TYPES;

function isDatasetTensorListSource(src: FlowNodeBare, sh: string): boolean {
  const isDatasetNode =
    src.type === "linear_dataset" ||
    src.type === "memorization_a_dataset" ||
    src.type === "memorization_b_dataset" ||
    src.type === "symbolic_func_dataset" ||
    src.type === "teacher_dataset" ||
    src.type === "token_prediction_dataset" ||
    src.type === "circle_random_walk_dataset" ||
    src.type === "circular_motion_dataset" ||
    src.type === "kepler_2d_dataset" ||
    src.type === "unigram_dataset" ||
    src.type === "bigram_low_rank_dataset" ||
    src.type === "in_context_associative_recall_dataset" ||
    src.type === "uniform_linear_motion_dataset" ||
    src.type === "diffusion_pde_dataset" ||
    src.type === "reaction_diffusion_dataset" ||
    src.type === "advection_dataset" ||
    src.type === "modular_addition_dataset" ||
    src.type === "dataset_mixer" ||
    src.type === "dataset_mixer_b" ||
    (TOY_LANGUAGE_DATASET_KINDS as readonly string[]).includes(String(src.type));
  if (!isDatasetNode) return false;
  return sh === "dataset" || sh === "train_dataset" || sh === "test_dataset";
}

function observableMetricLabel(nodes: FlowNodeBare[], observableNodeId: string): string {
  const [baseId, rankSuffix] = observableNodeId.split("::");
  const n = nodes.find((x) => x.id === baseId);
  if (!n) return observableNodeId;
  if (n.type === "kan_reg") return "KAN reg";
  if (n.type === "observable_weight_l2") return "Weight L2";
  if (n.type === "observable_weight_l1") return "Weight L1";
  if (n.type === "observable_capacity") return "Capacity";
  if (n.type === "observable_hessian_eigenvalues") {
    if (!rankSuffix) return "Hessian eigenvalues";
    const rankIdx = Number.parseInt(rankSuffix, 10);
    const d = (n.data ?? {}) as { order?: string };
    const order = d.order === "ascending" ? "ascending" : "descending";
    const orderWord = order === "ascending" ? "smallest" : "largest";
    if (Number.isFinite(rankIdx) && rankIdx >= 0) {
      return `Hessian eigenvalue ${rankIdx + 1} (${orderWord})`;
    }
    return "Hessian eigenvalues";
  }
  if (n.type === "observable_weight_product_sv") {
    if (!rankSuffix) return "Weight product SVs";
    const rankIdx = Number.parseInt(rankSuffix, 10);
    if (Number.isFinite(rankIdx) && rankIdx >= 0) {
      return `Weight product σ${rankIdx + 1}`;
    }
    return "Weight product SVs";
  }
  if (n.type === "observable_relu_nonlinear_count") return "ReLU nonlinear count";
  if (n.type === "observable_user") {
    const d = (n.data ?? {}) as { observableName?: string };
    return (d.observableName ?? "").trim() || "User observable";
  }
  return observableNodeId;
}

/**
 * Tensor list options for a given source handle (same as wiring into Tensor selector’s `tensor_list` input).
 */
export function tensorChoicesFromSourceHandle(
  nodes: FlowNodeBare[],
  src: FlowNodeBare,
  sourceHandle: string | null | undefined,
  edges?: FlowEdge[],
): TensorListChoice[] {
  const sh = sourceHandle ?? "";

  if (src.type === "activation" && sh === "tensor_list") {
    const data = (src.data ?? {}) as Partial<ActivationNodeData>;
    const picks = data.activationWirePicks ?? [];
    if (picks.length > 0) {
      return picks.map((p) => ({ id: p.tensorKey, label: p.label || p.tensorKey }));
    }
    const opts = data.representationOptions ?? [];
    const selected = new Set(data.selectedRepresentationIds ?? []);
    return opts.filter((o) => selected.has(o.id)).map((o) => ({ id: o.id, label: o.label }));
  }

  if (src.type === "model_weight_tensors" && sh === "tensor_list") {
    const data = (src.data ?? {}) as Partial<ModelWeightTensorsNodeData>;
    const w = data.weightTensorPayloads ?? {};
    return Object.keys(w)
      .sort()
      .map((id) => ({ id, label: id }));
  }

  if (src.type === "training_visualization" && sh === "out_tensor_list") {
    return [
      { id: "train_loss", label: "train loss" },
      { id: "test_loss", label: "test loss" },
    ];
  }

  if ((src.type === "observable_viz" || src.type === "observable_accuracy") && sh === "out_tensor") {
    const v = getObservableVizVariant(src as Node);
    if (v === "weight_l2") return [{ id: "weight_l2", label: "weight L2" }];
    if (v === "weight_l1") return [{ id: "weight_l1", label: "weight L1" }];
    if (v === "capacity") return [{ id: "capacity", label: "capacity" }];
    if (v === "accuracy") {
      return [
        { id: "train_accuracy", label: "train acc" },
        { id: "test_accuracy", label: "test acc" },
      ];
    }
    if (v === "relu_nonlinear") return [{ id: "relu_nonlinear_count", label: "ReLU nonlinear count" }];
    if (v === "user") {
      const d = (src.data ?? {}) as Partial<ObservableVizUserNodeData>;
      const name = (d.observableName ?? "user observable").trim() || "user observable";
      return [{ id: "user_observable", label: name }];
    }
    return [];
  }

  if ((src.type === "trainer" || src.type === "crl_trainer") && sh === "loss_results") {
    return [
      { id: "train_loss", label: "train loss" },
      { id: "test_loss", label: "test loss" },
    ];
  }

  if ((src.type === "trainer" || src.type === "crl_trainer") && sh === "observable_results") {
    const td = (src.data ?? {}) as Partial<TrainerNodeData>;
    const hist = td.observableMetricHistories ?? {};
    return Object.keys(hist)
      .sort()
      .map((obsId) => ({ id: obsId, label: observableMetricLabel(nodes, obsId) }));
  }

  if (src.type === "table_viz" && sh === "tensor" && edges) {
    return tableVizTensorListChoices(nodes as Node[], edges as Edge[], src.id);
  }

  if (isDatasetTensorListSource(src, sh)) {
    if (sh === "dataset") {
      return [
        { id: "train_input", label: "train · input" },
        { id: "train_output", label: "train · output" },
        { id: "test_input", label: "test · input" },
        { id: "test_output", label: "test · output" },
      ];
    }
    return [
      { id: "input", label: "input" },
      { id: "output", label: "output" },
    ];
  }

  return [];
}

/** Prefer train loss when present (e.g. trainer loss handle → tensor viz). */
export function defaultTensorKeyForMultiChoices(choices: TensorListChoice[]): string {
  const train = choices.find((c) => c.id === "train_loss");
  if (train) return train.id;
  const trainAcc = choices.find((c) => c.id === "train_accuracy");
  if (trainAcc) return trainAcc.id;
  const dsTrainIn = choices.find((c) => c.id === "train_input");
  if (dsTrainIn) return dsTrainIn.id;
  return choices[0]?.id ?? "";
}

/** Dropdown options for Tensor selector’s “tensor list” input, keyed by stable ids used in resolution. */
export function tensorChoicesForTensorsInput(
  nodes: FlowNodeBare[],
  edges: FlowEdge[],
  tensorSelectorId: string,
): TensorListChoice[] {
  const edge = edges.find((e) => e.target === tensorSelectorId && e.targetHandle === "tensor_list");
  if (!edge?.source) return [];
  const src = nodes.find((n) => n.id === edge.source);
  if (!src) return [];
  return tensorChoicesFromSourceHandle(nodes, src, edge.sourceHandle, edges);
}

function seriesOk(values: number[], sourceSummary: string): ResolvedOk {
  return {
    kind: "ok",
    rank: 1,
    shape: [values.length],
    values,
    sourceSummary,
  };
}

/**
 * Trainer stream may populate only `valueHistories` (rows per series) while `valueHistory` stays empty.
 * Passthrough to tensor viz / analysis uses the same primary row as the chart (first series, e.g. global).
 */
function observableVizScalarSeriesFromNodeData(d: {
  valueHistory?: unknown;
  valueHistories?: unknown;
}): number[] {
  if (Array.isArray(d.valueHistory)) {
    const single = d.valueHistory.map((x) => Number(x));
    if (single.length > 0 && single.some((x) => Number.isFinite(x))) {
      return single;
    }
  }
  const multi = d.valueHistories;
  if (!Array.isArray(multi) || multi.length === 0) return [];
  const row0 = multi[0];
  if (!Array.isArray(row0) || row0.length === 0) return [];
  const out = row0.map((x) => Number(x));
  return out.some((x) => Number.isFinite(x)) ? out : [];
}

function tensorElementCount(shape: number[]): number {
  return shape.reduce((acc, v) => acc * Math.max(0, Math.floor(v)), 1);
}

/** Placeholder flat buffer when shape changes and true activations are not materialized (e.g. linear / embedding). */
function placeholderValuesForShape(shape: number[]): number[] {
  const n = tensorElementCount(shape);
  if (!Number.isFinite(n) || n <= 0) return [];
  if (n > 50_000_000) return [];
  return new Array(n).fill(0);
}

function flattenOutputShapeRowMajor(shape: number[], exceptDim: number | null): number[] | null {
  const rank = shape.length;
  if (rank < 1) return null;
  if (exceptDim == null) {
    let p = 1;
    for (const s of shape) p *= Math.max(0, Math.floor(s));
    return [p];
  }
  let ax = exceptDim;
  if (ax < 0) ax = rank + ax;
  if (!Number.isInteger(ax) || ax < 0 || ax >= rank) return null;
  let prod = 1;
  for (let i = 0; i < rank; i += 1) {
    if (i !== ax) prod *= Math.max(0, Math.floor(shape[i]!));
  }
  return [Math.max(0, Math.floor(shape[ax]!)), prod];
}

/** Row-major contiguous flatten; only first- or last-axis “except” avoids reordering values. */
function applyFlattenToResolvedOk(inner: ResolvedOk, exceptDim: number | null): ResolvedOk | ResolvedNone {
  const outShape = flattenOutputShapeRowMajor(inner.shape, exceptDim);
  if (!outShape) return { kind: "none", detail: "Flatten: invalid shape or except dimension." };
  const exp = tensorElementCount(inner.shape);
  if (exp !== inner.values.length) {
    return { kind: "none", detail: "Flatten: value count does not match input shape." };
  }
  if (exceptDim == null) {
    return {
      ...inner,
      rank: outShape.length,
      shape: outShape,
      values: [...inner.values],
      sourceSummary: `${inner.sourceSummary} · flatten`,
    };
  }
  const rank = inner.shape.length;
  let ax = exceptDim;
  if (ax < 0) ax = rank + ax;
  if (ax !== 0 && ax !== rank - 1) {
    return {
      kind: "none",
      detail:
        "Flatten: keeping a middle axis needs reordering — use “except dimension” null (full flatten), first axis (0), or last axis (-1) for shape check.",
    };
  }
  return {
    ...inner,
    rank: outShape.length,
    shape: outShape,
    values: [...inner.values],
    sourceSummary: `${inner.sourceSummary} · flatten`,
  };
}

/**
 * Infer tensor shape at an atomic layer's output from its configured hyperparameters and the resolved input tensor.
 * Values may be zeros when the op changes shape (no forward pass here).
 */
function applyAtomicLayerIoOutput(inner: ResolvedOk, layerType: string, layerData: unknown): Resolved {
  const suffix = ` · ${layerType}`;
  const baseSummary = inner.sourceSummary;
  const sh = inner.shape;

  if (layerType === "activation_layer") {
    return { ...inner, sourceSummary: `${baseSummary}${suffix}` };
  }

  if (layerType === "linear_layer") {
    const d = layerData as Partial<LinearLayerNodeData>;
    const inF = intChoices(d.inFeatures, 1)[0]!;
    const outF = intChoices(d.outFeatures, 1)[0]!;
    if (sh.length < 1) {
      return { kind: "none", detail: "Linear layer: upstream tensor has rank 0." };
    }
    if (sh[sh.length - 1] !== inF) {
      return {
        kind: "none",
        detail: `Linear layer: last dimension is ${sh[sh.length - 1]} but in_features is ${inF} (upstream shape [${sh.join(", ")}]).`,
      };
    }
    const outShape = [...sh.slice(0, -1), outF];
    return {
      kind: "ok",
      rank: outShape.length,
      shape: outShape,
      values: placeholderValuesForShape(outShape),
      sourceSummary: `${baseSummary}${suffix}`,
    };
  }

  if (layerType === "layer_norm_layer") {
    const d = layerData as Partial<LayerNormLayerNodeData>;
    const nrm = intChoices(d.normalizedShape, 1)[0]!;
    if (sh.length < 1 || sh[sh.length - 1] !== nrm) {
      return {
        kind: "none",
        detail: `Layer norm: last dimension must equal normalized_shape (${nrm}); upstream is [${sh.join(", ")}].`,
      };
    }
    return { ...inner, sourceSummary: `${baseSummary}${suffix}` };
  }

  if (layerType === "embedding_layer") {
    const d = layerData as Partial<EmbeddingLayerNodeData>;
    const emb = intChoices(d.embeddingDim, 1)[0]!;
    const nIdx = intChoices(d.numIndexColumns, 1)[0]!;
    if (sh.length < nIdx) {
      return { kind: "none", detail: "Embedding layer: upstream rank is too small for num_index_columns." };
    }
    const outShape = [...sh.slice(0, sh.length - nIdx), emb];
    return {
      kind: "ok",
      rank: outShape.length,
      shape: outShape,
      values: placeholderValuesForShape(outShape),
      sourceSummary: `${baseSummary}${suffix}`,
    };
  }

  if (layerType === "unembedding_layer") {
    const d = layerData as Partial<UnembeddingLayerNodeData>;
    const inF = intChoices(d.inFeatures, 1)[0]!;
    const outF = intChoices(d.outFeatures, 1)[0]!;
    if (sh.length < 1 || sh[sh.length - 1] !== inF) {
      return {
        kind: "none",
        detail: `Unembedding: last dimension must equal in_features (${inF}); upstream is [${sh.join(", ")}].`,
      };
    }
    const outShape = [...sh.slice(0, -1), outF];
    return {
      kind: "ok",
      rank: outShape.length,
      shape: outShape,
      values: placeholderValuesForShape(outShape),
      sourceSummary: `${baseSummary}${suffix}`,
    };
  }

  if (layerType === "absolute_pos_embed_layer") {
    const d = layerData as Partial<AbsolutePosEmbedLayerNodeData>;
    const ed = intChoices(d.embeddingDim, 1)[0]!;
    const mxl = intChoices(d.maxSeqLen, 1)[0]!;
    if (sh.length < 2 || sh[sh.length - 1] !== ed) {
      return {
        kind: "none",
        detail: `Absolute position embedding: last dim must equal embedding_dim (${ed}); upstream is [${sh.join(", ")}].`,
      };
    }
    if (sh.length >= 3) {
      const seq = sh[sh.length - 2]!;
      if (seq > mxl) {
        return {
          kind: "none",
          detail: `Absolute position embedding: sequence length exceeds max_seq_len (${mxl}).`,
        };
      }
    }
    return { ...inner, sourceSummary: `${baseSummary}${suffix}` };
  }

  if (layerType === "rotary_embed_layer") {
    const d = layerData as Partial<RotaryEmbedLayerNodeData>;
    const rd = intChoices(d.rotaryDim, 1)[0]!;
    if (sh.length < 2 || sh[sh.length - 1] !== rd) {
      return {
        kind: "none",
        detail: `Rotary embedding: last dim must equal rotary_dim (${rd}); upstream is [${sh.join(", ")}].`,
      };
    }
    if (rd % 2 !== 0) {
      return { kind: "none", detail: "Rotary embedding: rotary_dim must be even." };
    }
    return { ...inner, sourceSummary: `${baseSummary}${suffix}` };
  }

  if (layerType === "local_mixing_layer") {
    const d = layerData as Partial<LocalMixingLayerNodeData>;
    const md = intChoices(d.modelDim, 64)[0]!;
    if (sh.length < 1 || sh[sh.length - 1] !== md) {
      return {
        kind: "none",
        detail: `Local mixing: last dim must equal model_dim (${md}); upstream is [${sh.join(", ")}].`,
      };
    }
    return { ...inner, sourceSummary: `${baseSummary}${suffix}` };
  }

  return { ...inner, sourceSummary: `${baseSummary}${suffix}` };
}

function resolveTensorFromListSource(
  nodes: FlowNodeBare[],
  edges: FlowEdge[],
  listEdge: FlowEdge,
  selectedKey: string,
): Resolved {
  const src = nodes.find((n) => n.id === listEdge.source);
  if (!src) return { kind: "none", detail: "Source node missing." };
  const sh = listEdge.sourceHandle ?? "";

  if (src.type === "activation" && sh === "tensor_list") {
    const actRaw = (src.data ?? {}) as Record<string, unknown>;
    const actData = actRaw as Partial<ActivationNodeData>;
    const runId = readActivationRunId(actRaw);
    const manifest = readActivationManifest(actRaw);
    if (runId && manifest) {
      const entry = manifest[selectedKey] as { shape?: unknown } | undefined;
      const shRaw = entry?.shape;
      if (Array.isArray(shRaw) && shRaw.length > 0) {
        const shape = shRaw.map((x) => Number(x));
        return {
          kind: "lazy_activation",
          runId,
          repId: selectedKey,
          shape,
          sourceSummary: `Tensor selector · ${selectedKey}`,
        };
      }
    }
    const collected = actData.collectedActivations;
    if (!collected || typeof collected !== "object") {
      return {
        kind: "none",
        detail:
          runId && !manifest?.[selectedKey]
            ? `No manifest entry for “${selectedKey}” — collect again with this representation checked, or pick another tensor.`
            : "Activation has no collected tensors yet (no server run id / manifest). Press Collect after wiring the model and dataset.",
      };
    }
    const legacy = collected[selectedKey] as CollectedActivationTensor | undefined;
    if (!legacy || !Array.isArray(legacy.shape) || !Array.isArray(legacy.values)) {
      return { kind: "none", detail: `No tensor data for key “${selectedKey}”.` };
    }
    const shape = legacy.shape.map((x) => Number(x));
    const values = legacy.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: `Tensor selector · ${selectedKey}`,
    };
  }

  if (src.type === "training_visualization" && sh === "out_tensor_list") {
    const d = (src.data ?? {}) as Partial<TrainingVisualizationNodeData>;
    if (selectedKey === "train_loss") {
      const values = (d.lossHistory ?? []).map(Number);
      if (values.length === 0) {
        return { kind: "none", detail: "No train loss series yet — run training." };
      }
      return seriesOk(values, "Training viz · train loss");
    }
    if (selectedKey === "test_loss") {
      const values = (d.testLossHistory ?? []).map(Number);
      if (values.length === 0) {
        return { kind: "none", detail: "No test loss series yet — run training." };
      }
      return seriesOk(values, "Training viz · test loss");
    }
    return { kind: "none", detail: "Pick train loss or test loss in the Tensor selector." };
  }

  if ((src.type === "observable_viz" || src.type === "observable_accuracy") && sh === "out_tensor") {
    const v = getObservableVizVariant(src as Node);
    if (v === "weight_l2") {
      if (selectedKey !== "weight_l2") {
        return { kind: "none", detail: "Select weight L2 for this panel." };
      }
      const d = (src.data ?? {}) as Partial<ObservableVizWeightL2NodeData>;
      const values = observableVizScalarSeriesFromNodeData(d);
      if (values.length === 0) {
        return { kind: "none", detail: "No weight L2 series yet — run training." };
      }
      return seriesOk(values, "Observable viz");
    }
    if (v === "weight_l1") {
      if (selectedKey !== "weight_l1") {
        return { kind: "none", detail: "Select weight L1 for this panel." };
      }
      const d = (src.data ?? {}) as Partial<ObservableVizWeightL1NodeData>;
      const values = observableVizScalarSeriesFromNodeData(d);
      if (values.length === 0) {
        return { kind: "none", detail: "No weight L1 series yet — run training." };
      }
      return seriesOk(values, "Observable viz");
    }
    if (v === "capacity") {
      if (selectedKey !== "capacity") {
        return { kind: "none", detail: "Select capacity for this panel." };
      }
      const d = (src.data ?? {}) as Partial<ObservableVizUserNodeData>;
      const values = observableVizScalarSeriesFromNodeData(d);
      if (values.length === 0) {
        return { kind: "none", detail: "No capacity series yet — run training." };
      }
      return seriesOk(values, "Observable viz");
    }
    if (v === "accuracy") {
      if (selectedKey !== "train_accuracy" && selectedKey !== "test_accuracy") {
        return { kind: "none", detail: "Pick train acc or test acc in the Tensor selector." };
      }
      const d = (src.data ?? {}) as Partial<ObservableVizUserNodeData>;
      const values =
        selectedKey === "test_accuracy"
          ? (d.testValueHistory ?? []).map(Number)
          : (d.valueHistory ?? []).map(Number);
      if (values.length === 0) {
        return {
          kind: "none",
          detail:
            selectedKey === "test_accuracy"
              ? "No test accuracy series yet — run training with a test set."
              : "No train accuracy series yet — run training.",
        };
      }
      const label = src.type === "observable_accuracy" ? "Accuracy viz" : "Observable viz";
      return seriesOk(
        values,
        `${label} · ${selectedKey === "test_accuracy" ? "test" : "train"} acc`,
      );
    }
    if (v === "relu_nonlinear") {
      if (selectedKey !== "relu_nonlinear_count") {
        return { kind: "none", detail: "Select ReLU nonlinear count for this panel." };
      }
      const d = (src.data ?? {}) as Partial<ObservableVizReluNonlinearNodeData>;
      const values = observableVizScalarSeriesFromNodeData(d);
      if (values.length === 0) {
        return { kind: "none", detail: "No nonlinear count series yet — run training." };
      }
      return seriesOk(values, "Observable viz");
    }
    if (v === "user") {
      if (selectedKey !== "user_observable") {
        return { kind: "none", detail: "Select the user observable series." };
      }
      const d = (src.data ?? {}) as Partial<ObservableVizUserNodeData>;
      const values = observableVizScalarSeriesFromNodeData(d);
      if (values.length === 0) {
        return { kind: "none", detail: "No user observable series yet — run training." };
      }
      return seriesOk(values, "Observable viz");
    }
    return { kind: "none", detail: "Unsupported observable viz variant for tensor list." };
  }

  if ((src.type === "trainer" || src.type === "crl_trainer") && sh === "loss_results") {
    const d = (src.data ?? {}) as Partial<TrainerNodeData>;
    if (selectedKey === "train_loss") {
      const values = (d.lossHistory ?? []).map(Number);
      if (values.length === 0) {
        return { kind: "none", detail: "No train loss history yet — run training." };
      }
      return seriesOk(values, "Trainer · train loss");
    }
    if (selectedKey === "test_loss") {
      const values = (d.testLossHistory ?? []).map(Number);
      if (values.length === 0) {
        return { kind: "none", detail: "No test loss history yet — run training." };
      }
      return seriesOk(values, "Trainer · test loss");
    }
    return { kind: "none", detail: "Pick train loss or test loss in the Tensor selector." };
  }

  if ((src.type === "trainer" || src.type === "crl_trainer") && sh === "observable_results") {
    const d = (src.data ?? {}) as Partial<TrainerNodeData>;
    const hist = d.observableMetricHistories ?? {};
    const arr = hist[selectedKey];
    if (!Array.isArray(arr) || arr.length === 0) {
      return {
        kind: "none",
        detail: `No data for observable “${observableMetricLabel(nodes, selectedKey)}” yet — run training.`,
      };
    }
    const values = arr.map(Number);
    return seriesOk(values, `Trainer · ${observableMetricLabel(nodes, selectedKey)}`);
  }

  if (src.type === "model_weight_tensors" && sh === "tensor_list") {
    const data = (src.data ?? {}) as Partial<ModelWeightTensorsNodeData>;
    const w = data.weightTensorPayloads ?? {};
    const raw = w[selectedKey];
    if (!raw || !Array.isArray(raw.shape) || !Array.isArray(raw.values)) {
      return {
        kind: "none",
        detail: `No weight data for “${selectedKey}” — run Collect on Model weight tensors.`,
      };
    }
    const shape = raw.shape.map((x) => Number(x));
    const values = raw.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: `Model weights · ${selectedKey}`,
    };
  }

  if (isDatasetTensorListSource(src, sh)) {
    let split: "train" | "test";
    let tensorKey: "input" | "output";
    if (sh === "dataset") {
      if (selectedKey === "test_input" || selectedKey === "test_output") {
        split = "test";
      } else if (selectedKey === "train_input" || selectedKey === "train_output") {
        split = "train";
      } else {
        split = "train";
      }
      if (selectedKey === "train_output" || selectedKey === "test_output" || selectedKey === "output") {
        tensorKey = "output";
      } else {
        tensorKey = "input";
      }
    } else {
      split = sh === "test_dataset" ? "test" : "train";
      tensorKey = selectedKey === "output" ? "output" : "input";
    }
    const includeGraph = src.type === "teacher_dataset";
    return {
      kind: "lazy_dataset",
      datasetNodeId: src.id,
      datasetNodeType: src.type,
      datasetData: src.data ?? {},
      graphNodes: includeGraph ? nodes : undefined,
      graphEdges: includeGraph ? edges : undefined,
      split,
      tensorKey,
      sourceSummary: `Dataset · ${split} ${tensorKey}`,
    };
  }

  return { kind: "none", detail: "Unsupported tensor list source for list picker." };
}

/** Selected tensor ids in upstream list order (matches Tensor selector checkboxes). */
export function orderedSelectedTensorKeysForPicker(
  tsData: Partial<TensorSelectorNodeData>,
  choices: TensorListChoice[],
): string[] {
  const idOrder = choices.map((c) => c.id);
  const idSet = new Set(idOrder);
  const raw = tsData.selectedTensorKeys;
  let keys: string[] = [];
  if (Array.isArray(raw)) {
    keys = raw.map((k) => String(k).trim()).filter((k) => k && idSet.has(k));
  } else {
    const one = String(tsData.selectedTensorKey ?? "").trim();
    if (one && idSet.has(one)) keys = [one];
  }
  const keySet = new Set(keys);
  return idOrder.filter((id) => keySet.has(id));
}

function resolveTensorFromListPicker(
  nodes: FlowNodeBare[],
  edges: FlowEdge[],
  pickerId: string,
  sourceHandle: string,
): Resolved {
  const listEdge = edges.find((e) => e.target === pickerId && e.targetHandle === "tensor_list");
  if (!listEdge?.source) {
    return {
      kind: "none",
      detail: "Connect the tensor list input (activation, model weights, viz panel, or trainer).",
    };
  }
  const picker = nodes.find((n) => n.id === pickerId);
  if (!picker || picker.type !== "tensor_selector") {
    return { kind: "none", detail: "Invalid list picker node." };
  }
  const tsData = (picker.data ?? {}) as Partial<TensorSelectorNodeData>;
  const choices = tensorChoicesForTensorsInput(nodes, edges, pickerId);
  const orderedKeys = orderedSelectedTensorKeysForPicker(tsData, choices);
  const idx = tensorSelectorOutputIndexFromSourceHandle(sourceHandle);
  const key = orderedKeys[idx] ?? orderedKeys[0] ?? "";
  if (!key) {
    return {
      kind: "none",
      detail: "Pick a tensor in the Tensor selector.",
    };
  }
  if (picker.type === "tensor_selector") {
    const caches = (tsData.activationTensorCaches ?? {}) as Record<string, ActivationTensorCache>;
    const legacy = tsData.activationTensorCache;
    const cache = caches[key] ?? (legacy && legacy.tensorKey === key ? legacy : undefined);
    if (cache && cache.tensorKey === key && cache.values.length > 0) {
      const act = nodes.find((n) => n.id === listEdge.source);
      if (act?.type === "activation") {
        const actRaw = (act.data ?? {}) as Record<string, unknown>;
        const runId = readActivationRunId(actRaw);
        if (runId && runId === cache.runId) {
          const shape = cache.shape.map((x) => Number(x));
          const rank = shape.length;
          return {
            kind: "ok",
            rank,
            shape,
            values: cache.values,
            sourceSummary: `Tensor selector · ${key}`,
          };
        }
      }
    }
  }
  return resolveTensorFromListSource(nodes, edges, listEdge, key);
}

/**
 * Minimal upstream tensor for inferring output shape when an atomic layer uses canvas **model** I/O (no wired tensor chain).
 */
function syntheticInnerForAtomicModelLayer(layerType: string, layerData: unknown): ResolvedOk | null {
  if (layerType === "linear_layer") {
    const d = layerData as Partial<LinearLayerNodeData>;
    const inF = intChoices(d.inFeatures, 1)[0]!;
    const shape = [inF];
    return {
      kind: "ok",
      rank: 1,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: `Assumed rank-1 input [${inF}] (layer model I/O — use input-output mode for a real upstream shape)`,
    };
  }
  if (layerType === "layer_norm_layer") {
    const d = layerData as Partial<LayerNormLayerNodeData>;
    const nrm = intChoices(d.normalizedShape, 1)[0]!;
    const shape = [nrm];
    return {
      kind: "ok",
      rank: 1,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: `Assumed trailing dim ${nrm} (LayerNorm model I/O)`,
    };
  }
  if (layerType === "rms_norm_layer") {
    const d = layerData as Partial<LayerNormLayerNodeData>;
    const nrm = intChoices(d.normalizedShape, 1)[0]!;
    const shape = [nrm];
    return {
      kind: "ok",
      rank: 1,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: `Assumed trailing dim ${nrm} (RMSNorm model I/O)`,
    };
  }
  if (layerType === "activation_layer") {
    const shape = [1];
    return {
      kind: "ok",
      rank: 1,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: "Assumed scalar input (activation model I/O)",
    };
  }
  return null;
}

/** Inferred forward output for full model shells in canvas **model** I/O (batch dimension assumed 1). */
function resolvedFullModelModelModeShell(sourceType: string, data: unknown): ResolvedOk | null {
  if (sourceType === "mlp_model" || sourceType === "gated_mlp_model" || sourceType === "moe_mlp_model") {
    const d = data as Partial<MlpModelNodeData>;
    const outF = intChoices(d.outputDim, 1)[0]!;
    const shape = [1, outF];
    return {
      kind: "ok",
      rank: 2,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary:
        sourceType === "gated_mlp_model"
          ? `Gated MLP model · forward output (batch assumed 1, ${outF} outputs)`
          : sourceType === "moe_mlp_model"
            ? `MoE MLP model · forward output (batch assumed 1, ${outF} outputs)`
            : `MLP model · forward output (batch assumed 1, ${outF} outputs)`,
    };
  }
  if (sourceType === "kan_model") {
    const d = data as Partial<KanModelNodeData>;
    const outF = intChoices(d.outputDim, 1)[0]!;
    const shape = [1, outF];
    return {
      kind: "ok",
      rank: 2,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: `KAN model · forward output (batch assumed 1, ${outF} outputs)`,
    };
  }
  if (sourceType === "residual_ln_model") {
    const d = data as Partial<ResidualLnModelNodeData>;
    const dim = intChoices(d.dim, 256)[0]!;
    const shape = [1, dim];
    return {
      kind: "ok",
      rank: 2,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: `Residual LN model · hidden dim ${dim} (batch assumed 1)`,
    };
  }
  if (
    sourceType === "attention_only_model" ||
    sourceType === "linear_attention_model" ||
    sourceType === "diagonal_ssm_token_model" ||
    sourceType === "rwkv_time_mix_token_model" ||
    sourceType === "hyena_like_conv_model" ||
    sourceType === "slot_attention_token_model"
  ) {
    const d = data as Partial<AttentionOnlyModelNodeData & AlternativeArchTokenLmNodeData>;
    const dim = intChoices(d.embedDim, 32)[0]!;
    const t = intChoices(d.contextLength, sourceType === "attention_only_model" ? 4 : 8)[0]!;
    const shape = [1, t, dim];
    return {
      kind: "ok",
      rank: 3,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: `${sourceType} · hidden sequence [batch, T, d] (batch=1, T=${t}, d=${dim})`,
    };
  }
  if (sourceType === "diffusion_score_model") {
    const d = data as Partial<DiffusionScoreModelNodeData>;
    const din = intChoices(d.inputDim, 8)[0]!;
    const shape = [1, din];
    return {
      kind: "ok",
      rank: 2,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: `Diffusion score MLP · noise prediction dim ${din} (batch=1)`,
    };
  }
  if (sourceType === "numeric_transformer_model") {
    const d = data as Partial<NumericTransformerModelNodeData>;
    const t = intChoices(d.contextLength, 2)[0]!;
    const dout = intChoices(d.outputDim, 1)[0]!;
    const shape = [1, t, dout];
    return {
      kind: "ok",
      rank: 3,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: `Numeric transformer model · forward output [batch, T, D_out] (batch=1, T=${t}, D_out=${dout})`,
    };
  }
  if (sourceType === "numeric_hyena_model") {
    const d = data as Partial<NumericHyenaModelNodeData>;
    const t = intChoices(d.contextLength, 8)[0]!;
    const dout = intChoices(d.outputDim, 2)[0]!;
    const shape = [1, t, dout];
    return {
      kind: "ok",
      rank: 3,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: `Numeric Hyena model · forward output [batch, T, D_out] (batch=1, T=${t}, D_out=${dout})`,
    };
  }
  if (sourceType === "transformer_token_model") {
    const d = data as Partial<TransformerTokenModelNodeData>;
    const v = intChoices(d.vocabSize, 100)[0]!;
    const shape = [1, v];
    return {
      kind: "ok",
      rank: 2,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: `Transformer (tokens) model · last-token logits (batch assumed 1, vocab ${v})`,
    };
  }
  if (sourceType === "transformer_multi_token_model") {
    const d = data as Partial<TransformerMultiTokenModelNodeData>;
    const v = intChoices(d.vocabSize, 100)[0]!;
    const k = intChoices(d.tokensPerPosition, 2)[0]!;
    const shape = [1, k, v];
    return {
      kind: "ok",
      rank: 3,
      shape,
      values: placeholderValuesForShape(shape),
      sourceSummary: `Transformer (multiple tokens) · last-timestep logits (batch=1, K=${k}, vocab ${v})`,
    };
  }
  return null;
}

/**
 * Resolve the numeric tensor feeding `consumerId` on its `targetHandle` input (e.g. Tensor viz "tensor").
 */
export function resolveUpstreamTensor(
  nodes: FlowNodeBare[],
  edges: FlowEdge[],
  consumerId: string,
  targetHandle = "tensor",
): Resolved {
  const incomingForNode = edges.filter((e) => e.target === consumerId);
  let inc =
    incomingForNode.find((e) => (e.targetHandle ?? "") === targetHandle) ??
    incomingForNode.find((e) => (e.targetHandle ?? "") === "") ??
    (incomingForNode.length === 1 ? incomingForNode[0] : undefined);
  if (!inc?.source) {
    return {
      kind: "none",
      detail:
        "Connect a tensor output (e.g. PCA / SVD outputs, Statistics, or Tensor selector).",
    };
  }
  return resolveTensorAtSourcePort(nodes, edges, inc.source, inc.sourceHandle);
}

/**
 * Resolve the tensor produced at ``sourceId``'s output handle (same cases as an incoming edge's source side).
 */
function resolveTensorAtSourcePort(
  nodes: FlowNodeBare[],
  edges: FlowEdge[],
  sourceId: string,
  sourceHandle: string | null | undefined,
): Resolved {
  const src = nodes.find((n) => n.id === sourceId);
  if (!src) return { kind: "none", detail: "Source node missing." };
  const sourceType = String(src.type ?? "");

  if (src.type === "effective_rank") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: "Connect from Effective rank output handle “tensor”." };
    }
    const ed = (src.data ?? {}) as Partial<EffectiveRankNodeData>;
    const t = ed.outputTensor;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: "Effective rank has no output yet — connect upstream and click Compute.",
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: "Effective rank · output",
    };
  }

  if (src.type === "series_endpoint_gap") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: "Connect from Series endpoint gap output handle “tensor”." };
    }
    const ed = (src.data ?? {}) as Partial<SeriesEndpointGapNodeData>;
    const t = ed.outputTensor;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: "Series endpoint gap has no output yet — connect upstream and click Compute.",
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: "Series endpoint gap · output",
    };
  }

  if (src.type === "smoothing_curve") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: "Connect from Smoothing curve output handle “tensor”." };
    }
    const sd = (src.data ?? {}) as Partial<SmoothingCurveNodeData>;
    const t = sd.outputTensor;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: "Smoothing curve has no output yet — connect upstream and click Compute.",
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: "Smoothing curve · output",
    };
  }

  if (src.type === "derivative_curve") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: "Connect from Derivative curve output handle “tensor”." };
    }
    const sd = (src.data ?? {}) as Partial<DerivativeCurveNodeData>;
    const t = sd.outputTensor;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: "Derivative curve has no output yet — connect upstream and click Compute.",
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: "Derivative curve · output",
    };
  }

  if (src.type === "statistics") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: "Connect from the Statistics output handle “tensor”." };
    }
    const sd = (src.data ?? {}) as Partial<StatisticsNodeData>;
    const t = sd.outputTensor;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: "Statistics has no output yet — connect upstream and click Compute.",
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: "Statistics · output",
    };
  }

  if (src.type === "statistics2") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: "Connect from the Statistics 2 output handle “tensor”." };
    }
    const sd = (src.data ?? {}) as Partial<Statistics2NodeData>;
    const t = sd.outputTensor;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: "Statistics 2 has no output yet — connect both inputs and click Compute.",
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: "Statistics 2 · output",
    };
  }

  if (src.type === "prediction") {
    const h = (sourceHandle ?? "").trim();
    if (h !== "train_pred" && h !== "test_pred") {
      return { kind: "none", detail: "Connect from Prediction output handle “train_pred” or “test_pred”." };
    }
    const pd = (src.data ?? {}) as Partial<PredictionNodeData>;
    const t = h === "train_pred" ? pd.trainPrediction : pd.testPrediction;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: `Prediction has no ${h === "train_pred" ? "train" : "test"} output yet — run Predict first.`,
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    return {
      kind: "ok",
      rank: shape.length,
      shape,
      values,
      sourceSummary: `Prediction · ${h === "train_pred" ? "train" : "test"} output`,
    };
  }

  if (src.type === "protein_structure_displayer") {
    const h = (sourceHandle ?? "").trim();
    if (h !== "structure") {
      return { kind: "none", detail: "Connect from Protein structure displayer output handle “structure”." };
    }
    const pd = (src.data ?? {}) as Partial<ProteinStructureDisplayerNodeData>;
    const coords = parseCoordsFlat(pd.resolvedCoordsFlat ?? pd.coordsFlat);
    if (!coords.length) {
      return {
        kind: "none",
        detail: "Protein structure displayer has no structure output yet.",
      };
    }
    const values = coords.flatMap((c) => [Number(c[0] ?? 0), Number(c[1] ?? 0), Number(c[2] ?? 0)]);
    return {
      kind: "ok",
      rank: 2,
      shape: [coords.length, 3],
      values,
      sourceSummary: "Protein structure displayer · structure",
    };
  }

  if (src.type === "tensor_add") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: "Connect from the Tensor add output handle “tensor”." };
    }
    const sd = (src.data ?? {}) as Partial<TensorAddNodeData>;
    const t = sd.outputTensor;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: "Tensor add has no output yet — connect both inputs and click Compute.",
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: "Tensor add · output",
    };
  }

  if (src.type === "basic_calculator") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: "Connect from the Basic calculator output handle “tensor”." };
    }
    const sd = (src.data ?? {}) as Partial<BasicCalculatorNodeData>;
    const t = sd.outputTensor;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: "Basic calculator has no output yet — wire scalar inputs and click Compute.",
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: "Basic calculator · output",
    };
  }

  if (src.type === "tensor_stack" || src.type === "tensor_concat") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: `Connect from the ${src.type === "tensor_stack" ? "Tensor stack" : "Tensor concat"} output handle “tensor”.` };
    }
    const rankedInputs = edges
      .filter((e) => e.target === sourceId && /^tensor_\d+$/.test(e.targetHandle ?? ""))
      .map((e) => ({
        edge: e,
        idx: Number.parseInt(String(e.targetHandle).slice("tensor_".length), 10),
      }))
      .filter((x) => Number.isFinite(x.idx))
      .sort((a, b) => a.idx - b.idx);
    const inputs: ResolvedOk[] = [];
    for (const { edge } of rankedInputs) {
      const inner = resolveTensorAtSourcePort(nodes, edges, edge.source, edge.sourceHandle);
      if (inner.kind === "ok") {
        inputs.push(inner);
        continue;
      }
      if (inner.kind === "lazy_activation") {
        return inner;
      }
    }
    if (inputs.length > 0) {
      const first = inputs[0]!;
      const sameShape = inputs.every(
        (x) => x.shape.length === first.shape.length && x.shape.every((v, i) => v === first.shape[i]),
      );
      if (!sameShape) {
        return {
          kind: "none",
          detail: `${src.type === "tensor_stack" ? "Tensor stack" : "Tensor concat"} requires compatible input tensor shapes.`,
        };
      }
      if (src.type === "tensor_stack") {
        const outShape = [inputs.length, ...first.shape];
        const outValues = inputs.flatMap((x) => x.values);
        return {
          kind: "ok",
          rank: outShape.length,
          shape: outShape,
          values: outValues,
          sourceSummary: "Tensor stack · output",
        };
      }

      const sd = (src.data ?? {}) as { concatDimension?: unknown };
      const rank = first.shape.length;
      const rawDim = Number(sd.concatDimension ?? 0);
      const dim = rawDim < 0 ? rank + Math.floor(rawDim) : Math.floor(rawDim);
      if (!Number.isFinite(dim) || dim < 0 || dim >= rank) {
        return { kind: "none", detail: "Tensor concat dimension is out of range for the input rank." };
      }

      let dimTotal = 0;
      for (const x of inputs) {
        for (let ax = 0; ax < rank; ax += 1) {
          if (ax === dim) continue;
          if (x.shape[ax] !== first.shape[ax]) {
            return { kind: "none", detail: "Tensor concat requires matching dimensions except on concat dimension." };
          }
        }
        dimTotal += x.shape[dim]!;
      }

      const outShape = [...first.shape];
      outShape[dim] = dimTotal;

      const outer = first.shape.slice(0, dim).reduce((acc, v) => acc * v, 1);
      const inner = first.shape.slice(dim + 1).reduce((acc, v) => acc * v, 1);
      const outValues: number[] = [];
      for (let o = 0; o < outer; o += 1) {
        for (const x of inputs) {
          const chunk = x.shape[dim]! * inner;
          const start = o * chunk;
          const end = start + chunk;
          outValues.push(...x.values.slice(start, end));
        }
      }

      if (outValues.length !== tensorElementCount(outShape)) {
        return {
          kind: "none",
          detail: "Tensor concat could not reconcile output values with the inferred output shape.",
        };
      }
      return {
        kind: "ok",
        rank: outShape.length,
        shape: outShape,
        values: outValues,
        sourceSummary: "Tensor concat · output",
      };
    }
    return {
      kind: "none",
      detail: `${src.type === "tensor_stack" ? "Tensor stack" : "Tensor concat"} has no valid input tensor connected yet.`,
    };
  }

  if (src.type === "tensor_constant") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: "Connect from the Tensor constant output handle “tensor”." };
    }
    const sd = (src.data ?? {}) as Partial<TensorConstantNodeData>;
    const t = sd.outputTensor;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: "Tensor constant has no data yet — set a valid shape and initialization (or init seed for random fills).",
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: "Tensor constant · output",
    };
  }

  if (src.type === "tensor_linspace") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: "Connect from the Tensor linspace output handle “tensor”." };
    }
    const sd = (src.data ?? {}) as Partial<TensorLinspaceNodeData>;
    const t = sd.outputTensor;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: "Tensor linspace has no data yet — set start, end, and number of points.",
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: "Tensor linspace · output",
    };
  }

  if (src.type === "elementwise_transform") {
    const h = sourceHandle ?? "";
    if (h !== "tensor") {
      return { kind: "none", detail: "Connect from the Elementwise transform output handle “tensor”." };
    }
    const sd = (src.data ?? {}) as Partial<ElementwiseTransformNodeData>;
    const t = sd.outputTensor;
    if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
      return {
        kind: "none",
        detail: "Elementwise transform has no output yet — connect an input tensor and click Compute.",
      };
    }
    const shape = t.shape.map((x) => Number(x));
    const values = t.values.map((x) => Number(x));
    const rank = shape.length;
    return {
      kind: "ok",
      rank,
      shape,
      values,
      sourceSummary: "Elementwise transform · output",
    };
  }

  if (src.type === "flatten") {
    const h = (sourceHandle ?? "").trim();
    const mode = readNodeCanvasIoMode((src.data ?? {}) as Record<string, unknown>);
    if (
      mode === "input-output" &&
      (h === LAYER_STRIP_SOURCE_HANDLE || h === "tensor_out" || h === "tensor")
    ) {
      const ex = readFlattenExceptDim((src.data as { exceptDim?: unknown } | undefined)?.exceptDim);
      const inner = resolveUpstreamTensor(nodes, edges, src.id, LAYER_STRIP_TARGET_HANDLE);
      if (inner.kind === "lazy_activation" || inner.kind === "lazy_dataset") {
        return appendLazyOp(inner, { kind: "flatten", exceptDim: ex });
      }
      if (inner.kind !== "ok") return inner;
      return applyFlattenToResolvedOk(inner, ex);
    }
    return {
      kind: "none",
      detail: "Flatten: use input-output mode and connect from the right tensor output handle.",
    };
  }

  if (ATOMIC_LAYER_TYPES.has(sourceType)) {
    const h = (sourceHandle ?? "").trim();
    const mode = readNodeCanvasIoMode((src.data ?? {}) as Record<string, unknown>);
    if (mode === "input-output" && (h === LAYER_STRIP_SOURCE_HANDLE || h === "tensor_out")) {
      const inner = resolveUpstreamTensor(nodes, edges, src.id, LAYER_STRIP_TARGET_HANDLE);
      if (inner.kind === "lazy_activation") return inner;
      if (inner.kind === "lazy_dataset") return inner;
      if (inner.kind !== "ok") return inner;
      return applyAtomicLayerIoOutput(inner, sourceType, src.data);
    }
    if (mode === "model" && h === "tensor") {
      const syn = syntheticInnerForAtomicModelLayer(sourceType, src.data);
      if (syn) {
        return applyAtomicLayerIoOutput(syn, sourceType, src.data);
      }
      return {
        kind: "none",
        detail:
          "This layer is in model mode; tensor shape cannot be inferred for this layer type. Switch to input-output mode and connect from tensor out.",
      };
    }
    return {
      kind: "none",
      detail: "Connect from this layer's right tensor output handle.",
    };
  }

  if (FULL_MODEL_CANVAS_TYPES.has(sourceType)) {
    const h = (sourceHandle ?? "").trim();
    const mode = readNodeCanvasIoMode((src.data ?? {}) as Record<string, unknown>);
    if (mode === "input-output" && (h === LAYER_STRIP_SOURCE_HANDLE || h === "tensor_out")) {
      const inner = resolveUpstreamTensor(nodes, edges, src.id, LAYER_STRIP_TARGET_HANDLE);
      if (inner.kind === "lazy_activation") return inner;
      if (inner.kind !== "ok") return inner;
      return {
        ...inner,
        sourceSummary: `${inner.sourceSummary} · ${sourceType}`,
      };
    }
    if (mode === "model" && (h === "tensor" || h === "model")) {
      const shell = resolvedFullModelModelModeShell(sourceType, src.data);
      if (shell) return shell;
      return {
        kind: "none",
        detail:
          "This model is in model mode; tensor shape is unavailable on the model output. Switch to input-output mode and connect from tensor out.",
      };
    }
    return {
      kind: "none",
      detail: "Connect from this model's right tensor output handle.",
    };
  }

  if (src.type === "combined_model") {
    const h = (sourceHandle ?? "").trim();
    const mode = readNodeCanvasIoMode((src.data ?? {}) as Record<string, unknown>);
    if (mode === "input-output" && (h === LAYER_STRIP_SOURCE_HANDLE || h === "tensor_out")) {
      const inner = resolveUpstreamTensor(nodes, edges, src.id, LAYER_STRIP_TARGET_HANDLE);
      if (inner.kind === "lazy_activation") return inner;
      if (inner.kind !== "ok") return inner;
      return {
        ...inner,
        sourceSummary: `${inner.sourceSummary} · combined model`,
      };
    }
    if (mode === "model" && (h === "tensor" || h === "model")) {
      const retEdge = edges.find(
        (e) => e.target === src.id && (e.targetHandle ?? "").trim() === COMBINED_MODEL_RETURN_TARGET_HANDLE,
      );
      if (!retEdge?.source) {
        return {
          kind: "none",
          detail:
            "Combined model (model shell) has no inner return wire — re-Combine the subgraph so its tail connects to the shell, or switch I/O to input-output.",
        };
      }
      const innerAt = resolveTensorAtSourcePort(nodes, edges, retEdge.source, retEdge.sourceHandle);
      if (innerAt.kind === "lazy_activation") return innerAt;
      if (innerAt.kind !== "ok") return innerAt;
      return {
        ...innerAt,
        sourceSummary: `${innerAt.sourceSummary} · combined model`,
      };
    }
    return {
      kind: "none",
      detail:
        mode === "model"
          ? "Connect from the combined model's right \"tensor\" / model output handle."
          : "Connect from the combined model tensor out handle.",
    };
  }

  if (src.type === "pca") {
    const h = sourceHandle ?? "";
    const pd = (src.data ?? {}) as Partial<PcaNodeData>;
    if (h === "explained_variance_ratio") {
      const arr = pd.explainedVarianceRatio;
      if (!Array.isArray(arr) || arr.length === 0) {
        return { kind: "none", detail: "No explained variance yet — run PCA first." };
      }
      const values = arr.map((x) => Number(x));
      return {
        kind: "ok",
        rank: 1,
        shape: [values.length],
        values,
        sourceSummary: "PCA · explained variance ratio",
      };
    }
    if (h === "transformed_tensor") {
      const t = pd.transformedTensor;
      if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
        return { kind: "none", detail: "No transformed tensor yet — run PCA first." };
      }
      const shape = t.shape.map((x) => Number(x));
      const values = t.values.map((x) => Number(x));
      const rank = shape.length;
      return {
        kind: "ok",
        rank,
        shape,
        values,
        sourceSummary: "PCA · transformed tensor",
      };
    }
    if (h === "principal_components") {
      const t = pd.principalComponents;
      if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
        return { kind: "none", detail: "No principal components yet — run PCA first." };
      }
      const shape = t.shape.map((x) => Number(x));
      const values = t.values.map((x) => Number(x));
      const rank = shape.length;
      return {
        kind: "ok",
        rank,
        shape,
        values,
        sourceSummary: "PCA · principal components",
      };
    }
    return {
      kind: "none",
      detail:
        "Connect from PCA “transformed tensor”, “principal components”, or “explained variance ratio”.",
    };
  }

  if (src.type === "svd") {
    const h = sourceHandle ?? "";
    const sd = (src.data ?? {}) as Partial<SvdNodeData>;
    if (h === "u") {
      const t = sd.uTensor;
      if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
        return { kind: "none", detail: "No U tensor yet — run SVD first." };
      }
      const shape = t.shape.map((x) => Number(x));
      const values = t.values.map((x) => Number(x));
      const rank = shape.length;
      return {
        kind: "ok",
        rank,
        shape,
        values,
        sourceSummary: "SVD · U",
      };
    }
    if (h === "s") {
      const t = sd.sTensor;
      if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
        return { kind: "none", detail: "No S (singular values) tensor yet — run SVD first." };
      }
      const shape = t.shape.map((x) => Number(x));
      const values = t.values.map((x) => Number(x));
      const rank = shape.length;
      return {
        kind: "ok",
        rank,
        shape,
        values,
        sourceSummary: "SVD · singular values",
      };
    }
    if (h === "v") {
      const t = sd.vTensor;
      if (!t || !Array.isArray(t.shape) || !Array.isArray(t.values) || t.values.length === 0) {
        return { kind: "none", detail: "No Vh tensor yet — run SVD first." };
      }
      const shape = t.shape.map((x) => Number(x));
      const values = t.values.map((x) => Number(x));
      const rank = shape.length;
      return {
        kind: "ok",
        rank,
        shape,
        values,
        sourceSummary: "SVD · Vh",
      };
    }
    return {
      kind: "none",
      detail: "Connect from SVD outputs U, S, or Vh.",
    };
  }

  const incHandle = sourceHandle ?? "";
  if (src.type === "training_visualization" && incHandle === "out_tensor_list") {
    const d = (src.data ?? {}) as Partial<TrainingVisualizationNodeData>;
    const values = (d.lossHistory ?? []).map(Number);
    if (values.length === 0) {
      return { kind: "none", detail: "No train loss series yet — run training." };
    }
    return seriesOk(values, "Training viz · train loss");
  }

  if ((src.type === "observable_viz" || src.type === "observable_accuracy") && incHandle === "out_tensor") {
    if (observableVizAllowsTensorVizChain(src as Node)) {
      const d = (src.data ?? {}) as Partial<
        ObservableVizWeightL2NodeData &
          ObservableVizWeightL1NodeData &
          ObservableVizReluNonlinearNodeData &
          ObservableVizUserNodeData
      >;
      const values = observableVizScalarSeriesFromNodeData(d);
      if (values.length === 0) {
        return {
          kind: "none",
          detail: "No series yet — connect the trainer and run training.",
        };
      }
      return seriesOk(values, "Observable viz");
    }
  }

  const tensorVizPassthrough = new Set([
    "tensor_viz_general",
    "tensor_viz_0d",
    "tensor_viz_1d",
    "tensor_viz_2d",
  ]);
  const sh = incHandle;
  if (src.type && tensorVizPassthrough.has(String(src.type)) && sh === "out_tensor") {
    return resolveUpstreamTensor(nodes, edges, src.id, "tensor");
  }

  if (src.type === "trainer" || src.type === "crl_trainer") {
    const h = sourceHandle ?? "";
    const td = (src.data ?? {}) as Partial<TrainerNodeData>;
    if (h === "loss_results") {
      const values = (td.lossHistory ?? []).map(Number);
      if (values.length === 0) {
        return { kind: "none", detail: "No train loss history yet — run training (direct loss handle uses train loss)." };
      }
      return seriesOk(values, "Trainer · train loss");
    }
    if (h === "observable_results") {
      const hist = td.observableMetricHistories ?? {};
      const keys = Object.keys(hist).sort();
      if (keys.length === 0) {
        return { kind: "none", detail: "No observable metrics yet — run training." };
      }
      const pick = keys[0]!;
      const values = (hist[pick] ?? []).map(Number);
      if (values.length === 0) {
        return { kind: "none", detail: "Observable series is empty — run training." };
      }
      return seriesOk(values, `Trainer · ${observableMetricLabel(nodes, pick)}`);
    }
    return { kind: "none", detail: "Connect from trainer “loss” or “observable” result handles." };
  }

  if (isDatasetTensorListSource(src, sourceHandle ?? "")) {
    const sh = sourceHandle ?? "";
    const selectedKey =
      sh === "dataset"
        ? "train_output"
        : sh === "test_dataset"
          ? "output"
          : "output";
    return resolveTensorFromListSource(nodes, edges, src, sh, selectedKey);
  }

  if (src.type === "tensor_selector") {
    return resolveTensorFromListPicker(nodes, edges, src.id, sourceHandle ?? "");
  }

  if (src.type === "dimension_permutator") {
    const h = sourceHandle ?? "";
    if (h !== "tensor_out" && h !== "") {
      return {
        kind: "none",
        detail: 'Connect from the Dimension permutator output handle (labeled “tensor”).',
      };
    }
    const rawAxes = (src.data ?? {}) as Partial<DimensionPermutatorNodeData>;
    const inner = resolveUpstreamTensor(nodes, edges, src.id, "tensor_in");
    if (inner.kind === "lazy_activation") {
      const rankHint = inner.shape.length;
      const axLazy = normalizePermutation(Array.isArray(rawAxes.axes) ? rawAxes.axes : undefined, rankHint);
      if (axLazy.length !== rankHint) {
        return { kind: "none", detail: "Invalid permutation config for this tensor rank." };
      }
      const expected = [...Array(rankHint).keys()];
      const sorted = [...axLazy].sort((a, b) => a - b);
      if (sorted.some((v, i) => v !== expected[i])) {
        return { kind: "none", detail: "Permutation axes must be a full reordering of dimensions." };
      }
      return appendLazyOp(inner, { kind: "permute", axes: axLazy });
    }
    if (inner.kind === "lazy_dataset") return inner;
    if (inner.kind !== "ok") return inner;
    const ax = normalizePermutation(Array.isArray(rawAxes.axes) ? rawAxes.axes : undefined, inner.rank);
    const { shape, values } = permuteRowMajor(inner.shape, inner.values, ax);
    const axStr = ax.join(",");
    return {
      kind: "ok",
      rank: shape.length,
      shape,
      values,
      sourceSummary: `${inner.sourceSummary} · permute(${axStr})`,
    };
  }

  if (src.type === "tensor_slicing") {
    const h = sourceHandle ?? "";
    if (h !== "tensor" && h !== "") {
      return {
        kind: "none",
        detail: 'Connect from the Tensor slicing output handle (labeled "tensor").',
      };
    }
    const raw = (src.data ?? {}) as Partial<TensorSlicingNodeData>;
    const slices = normalizeSlices(raw.slices ?? []);
    const inner = resolveUpstreamTensor(nodes, edges, src.id, "tensor");
    if (inner.kind === "lazy_activation" || inner.kind === "lazy_dataset") {
      return appendLazyOp(inner, { kind: "slice", slices });
    }
    if (inner.kind !== "ok") return inner;
    const out = applyTensorSlicing(inner.shape, inner.values, slices);
    if (!out) {
      return {
        kind: "none",
        detail: "Invalid slicing config: check dimensions and index ranges.",
      };
    }
    return {
      kind: "ok",
      rank: out.shape.length,
      shape: out.shape,
      values: out.values,
      sourceSummary: `${inner.sourceSummary} · slice`,
    };
  }

  return {
    kind: "none",
    detail:
      "Unsupported source — use PCA, SVD, Statistics, Tensor selector, Tensor add/stack/concat, Dimension permutator, Tensor slicing, Flatten, Effective rank, smoothing/derivative curve, or trainer outputs.",
  };
}
