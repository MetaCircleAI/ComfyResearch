import {
  applyEdgeChanges,
  applyNodeChanges,
  Background,
  BackgroundVariant,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  useUpdateNodeInternals,
  useViewport,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  type OnEdgesChange,
  type OnNodesChange,
  type Viewport,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type CSSProperties,
  type DragEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import { flushSync } from "react-dom";
import { ShapeCheckOverlayProvider } from "../context/shapeCheckOverlayContext";
import { AddNodeSearchModal } from "./AddNodeSearchModal";
import { isInteractiveNodePressTarget } from "./canvasNodeGesture";
import {
  getContinuousWheelZoom,
  wheelTargetAllowsCanvasZoom,
} from "./canvasWheelZoom";
import { DeletingBusyOverlay } from "./DeletingBusyOverlay";

import { DiscreteMultiSelect } from "./nodes/DiscreteMultiSelect";
import { FlowSurfaceProvider } from "../context/FlowSurfaceContext";
import { ResearchGraphProvider, type AddNodeOptions } from "../context/ResearchGraphContext";
import {
  COMBINED_MODEL_TEMPLATE_DND_MIME,
  DND_MIME,
  DND_TEXT_PLAIN,
  GRAPH_COMBINED_MODEL_LIBRARY_CHANGED,
  GRAPH_TEMPLATE_LIBRARY_CHANGED,
  USER_LINEAR_DATASET_DND_MIME,
  USER_OBSERVABLE_DND_MIME,
  USER_OBSERVABLES_CHANGED,
  USER_SYMBOLIC_FUNC_DATASET_DND_MIME,
} from "../dnd";
import { useBlurOpenSelectOnOutsidePointer } from "../hooks/useBlurOpenSelectOnOutsidePointer";
import { useDismissOnOutsidePointer } from "../hooks/useDismissOnOutsidePointer";
import { useWorkspaceApi } from "../hooks/useWorkspaceApi";
import type { GraphDocument, NodeKind } from "../types/graph";
import type { WorkspaceSnapshotDTO } from "../types/workspace";
import {
  defaultLinearDatasetData,
  defaultRandomNoiseDatasetData,
  defaultMemorizationADatasetData,
  defaultMemorizationBDatasetData,
} from "./nodes/linearDatasetDefaults";
import { defaultSymbolicFuncDatasetData } from "./nodes/symbolicFuncDatasetDefaults";
import { defaultMlpModelData } from "./nodes/mlpModelDefaults";
import { defaultGatedMlpModelData } from "./nodes/gatedMlpModelDefaults";
import { defaultMoeMlpModelData } from "./nodes/moeMlpModelDefaults";
import { defaultKanModelData } from "./nodes/kanModelDefaults";
import { defaultMlpTokenModelData } from "./nodes/mlpTokenModelDefaults";
import { defaultModularAdditionDatasetData } from "./nodes/modularAdditionDatasetDefaults";
import { defaultDatasetMixerData } from "./nodes/datasetMixerDefaults";
import { defaultDatasetMixerBData } from "./nodes/datasetMixerBDefaults";
import {
  defaultToyLanguageDatasetData,
  TOY_LANGUAGE_DATASET_KINDS,
  type ToyLanguageDatasetKind,
} from "./nodes/toyLanguageDatasetDefaults";
import {
  defaultVisionDatasetData,
  VISION_DATASET_KINDS,
  type VisionDatasetKind,
} from "./nodes/visionDatasetDefaults";
import {
  defaultPdeFieldDatasetData,
  PDE_FIELD_DATASET_KINDS,
  type PdeFieldDatasetKind,
} from "./nodes/pdeFieldDatasetDefaults";
import { defaultNumericTransformerModelData } from "./nodes/numericTransformerModelDefaults";
import { defaultNumericHyenaModelData } from "./nodes/numericHyenaModelDefaults";
import { defaultMppSpatiotemporalModelData } from "./nodes/mppSpatiotemporalModelDefaults";
import { defaultAfnoLiteSpatiotemporalModelData } from "./nodes/afnoLiteSpatiotemporalModelDefaults";
import { defaultTransformerMultiTokenModelData } from "./nodes/transformerMultiTokenModelDefaults";
import { defaultTransformerTokenModelData } from "./nodes/transformerTokenModelDefaults";
import { defaultAttentionOnlyModelData } from "./nodes/attentionOnlyModelDefaults";
import {
  migrateAttentionOnlyTensorSelectorData,
  migrateAttentionOnlyWeightTensorPayloads,
} from "../graph/attentionOnlyParameterMigration";
import {
  defaultAlternativeArchTokenLmData,
  type ArchLmKind,
} from "./nodes/alternativeArchModelDefaults";
import { defaultTokenPredictionDatasetData } from "./nodes/tokenPredictionDatasetDefaults";
import { defaultCircleRandomWalkDatasetData } from "./nodes/circleRandomWalkDatasetDefaults";
import { defaultCircularMotionDatasetData } from "./nodes/circularMotionDatasetDefaults";
import { defaultKepler2dDatasetData } from "./nodes/kepler2dDatasetDefaults";
import { defaultUnigramDatasetData } from "./nodes/unigramDatasetDefaults";
import { defaultBigramLowRankDatasetData } from "./nodes/bigramLowRankDatasetDefaults";
import { defaultRandomInputDistributionData } from "./nodes/randomInputDistributionDefaults";
import { defaultInputSamplerData } from "./nodes/inputSamplerDefaults";
import { defaultTeacherDatasetData } from "./nodes/teacherDatasetDefaults";
import { defaultInContextAssociativeRecallDatasetData } from "./nodes/inContextAssociativeRecallDatasetDefaults";
import { defaultUniformLinearMotionDatasetData } from "./nodes/uniformLinearMotionDatasetDefaults";
import { defaultCrlEnvConfigData } from "./nodes/crlEnvDefaults";
import type { AutoTuneComparisonResult, TrainerNodeData } from "./nodes/trainerDefaults";
import { CombinedSubgraphIoEdge } from "./edges/CombinedSubgraphIoEdge";
import { ResearchConnectionLine, ResearchDefaultEdge } from "./edges/ResearchDefaultEdge";
import { ConnectionSnapContext, type ConnectionSnapTarget } from "./edges/connectionSnapContext";
import {
  defaultTensorKeyForMultiChoices,
  tensorChoicesFromSourceHandle,
  type FlowNodeBare,
} from "../graph/resolveUpstreamTensor";
import { readNdjsonTrainStream, type TrainStreamProgress } from "../graph/readNdjsonTrainStream";
import {
  computeSelfDrivingAnchor,
  planRandomTrainerSubgraph,
  type PlannedRandomTrainer,
} from "../graph/selfDrivingGraph";
import {
  appendUserObservableNodesToTrainer,
  createRandomUserObservablesForModel,
  findModelNodeIdFromPlan,
  PHYSICS_OF_AI_OBSERVABLE_COUNT,
} from "../graph/physicsOfAiAgent";
import {
  abortProjectTraining,
  summarizeProjectTrainActivity,
  type ProjectTrainSummary,
} from "../graph/projectTrainActivity";
import {
  GRAPH_ASSIST_MATRIX_CELL_H,
  GRAPH_ASSIST_MATRIX_CELL_INSET_X,
  GRAPH_ASSIST_MATRIX_CELL_INSET_Y,
  GRAPH_ASSIST_MATRIX_CELL_W,
  GRAPH_ASSIST_SUBGRAPH_TOP_DY,
  graphAssistGridCellOrder,
} from "../graph/graphAssistGridLayout";
import { createSeededRng, parseGraphAssistSeed, parseGraphAssistSeeds } from "../graph/seededRandom";
import {
  formatTrainSeriesSweptLines,
  getSweptAxisIdSet,
  planTrainSeriesAssignments,
  serializeNodeForTrain,
} from "../graph/trainSeriesPlan";
import { applyAssignmentsAndResolveTrainingLength } from "../graph/trainingLengthResolve";
import { buildAutoTuneAxisSuggestions } from "../graph/autoTuneAxisSuggestions";
import { normalizeOptimizerLrScheduleEdgeTargets } from "../graph/normalizeOptimizerLrEdges";
import { flushCheckpointApplyTrainerVizAndHydrateTv0d } from "../graph/trainerTrainCompleteCommit";
import { mergeWorkspaceHydrateWithLocalProjects } from "../graph/workspaceHydrate";
import { patchTrainerHostUi } from "../graph/trainerHostUiPatch";
import { ensureTrainerAutoVizes } from "../graph/trainerAutoVizSpawn";
import {
  clipAbsolutePosition,
  computeCombinedModelBridgeRewire,
  expandCombineSelectionNodeIds,
  extractSubgraphByNodeIds,
  repairMissingCombinedModelReturnEdges,
} from "../graph/combineNodesUtils";
import {
  buildNodeDefinitionPython,
  buildTrainerRunnerPython,
  shouldOmitNotebookCell,
} from "../graph/nodeDefinitionCode";
import {
  appendResearchNode,
  cloneSubgraphForPaste,
  INSTANCE_TITLE_KEY,
  migrateCombinedModelInstanceTitles,
  withNewInstanceTitle,
  type ClipboardSubgraph,
} from "../graph/nodeInstanceTitle";
import { applyCanvasConnection, isValidCanvasConnection, planAutoConnectCanvas } from "../graph/connectionRules";
import { readNodeCanvasIoMode } from "../graph/nodeCanvasIoMode";
import {
  COMBINED_MODEL_RETURN_TARGET_HANDLE,
  COMBINED_SUBGRAPH_IO_EDGE_TYPE,
  LAYER_STRIP_SOURCE_HANDLE,
  LAYER_STRIP_TARGET_HANDLE,
} from "../graph/layerStripHandles";
import { augmentNodeRemovesWithMlpLowExpansion } from "../graph/mlpLowLevelExpansion";
import { augmentNodeRemovesWithAttentionLowExpansion } from "../graph/attentionLowLevelExpansion";
import { readNodeCanvasLevelMode } from "../graph/nodeCanvasLevelMode";
import { sortNodesParentBeforeChildren } from "../graph/sortNodesParentBeforeChildren";
import {
  applyGraphFileExportTier,
  type GraphFileExportTier,
} from "../graph/graphFileExportTier";
import { layoutResearchGraphNodes } from "../graph/graphAutoLayout";
import {
  captureFlowRendererToPngBlob,
  pngBlobToSinglePagePdfBlob,
  saveBlobWithUserLocation,
} from "../graph/canvasExport";
import { readGraphNodeLoopCount, readGraphNodeLoopShareParams } from "../graph/nodeLoopCount";
import {
  addSavedGraphEntry,
  fetchSavedGraphLibrary,
  isClassicPaperReproductionTemplate,
  migrateLegacyLocalStorageToServer,
  removeSavedGraphEntry,
  type SavedGraphKind,
  type SavedGraphEntry,
} from "../graph/savedGraphLibrary";
import { GraphCompareModal, type GraphCompareTarget } from "./GraphCompareModal";
import { CombineModelModal } from "./CombineModelModal";
import { NodeLoopModal } from "./NodeLoopModal";
import { RenameCombinedModelModal } from "./RenameCombinedModelModal";
import { PasteNodesChoiceModal } from "./PasteNodesChoiceModal";
import { ConfirmModal } from "./ConfirmModal";
import { TargetCurveModal } from "./TargetCurveModal";
import { AutoTuneConfigModal, type AutoTuneConfig } from "./AutoTuneConfigModal";
import { AutoTuneResultsModal } from "./AutoTuneResultsModal";
import { LibrarySaveModal, type LibrarySaveDraft } from "./LibrarySaveModal";
import { SavedGraphLibraryPanel } from "./SavedGraphLibraryPanel";
import type { RailPrimarySection } from "./railTypes";
import { defaultCombinedModelData } from "./nodes/combinedModelDefaults";
import { defaultObservableUserData } from "./nodes/observableUserDefaults";
import { GENERATED_NODE_SPECS } from "../generated/generatedNodeSpecs";
import { nodeRegistryDefaults } from "../graph/nodeRegistrySpec";
import { tableVizTensorConnectable, tableVizTensorListChoices } from "../graph/tableVizRegressor";
import { LeftNavRail } from "./LeftNavRail";
import { NodeInformationPanel } from "./NodeInformationPanel";
import { OPEN_NODE_INFORMATION_EVENT, type OpenNodeInformationDetail } from "./nodeInformationEvents";
import { NodesLibraryPanel } from "./NodesLibraryPanel";
import { OPEN_CURVE_STARER_EVENT } from "../curveStarer/observableCurvePayload";
import { CurveStarerModal } from "../curveStarer/CurveStarerModal";
import { suggestDefaultTarget } from "../curveStarer/suggestDefaultTarget";
import type { CurveStarerTargetConfig } from "../curveStarer/speedUpTrickTypes";
import {
  getCurveStarerCachedEntries,
  setCurveStarerCache,
} from "../curveStarer/curveStarerCache";
import {
  collectObservableTrainingCurves,
  LPD_MIN_CURVE_POINTS,
} from "../curveStarer/collectObservableCurves";
import {
  fetchLpdPredictBatch,
  type CurveStarerAnalyzedEntry,
  type CurveStarerRankBy,
} from "../curveStarer/lpdTypes";
import { ObservablePanel } from "../observables/ObservablePanel";
import { isObservableModelNodeType } from "../observables/modelNodeTypes";
import { researchNodeTypes } from "./nodeTypes";
import { migrateObservableVizNodeTypes } from "../graph/observableVizVariant";
import {
  beginLibraryNodeDrag,
  endLibraryNodeDrag,
  isOverNodesLibrary,
  markLibraryDragNode,
  updateLibraryNodeDragTarget,
} from "../graph/libraryNodeDrag";

type CodeNotebookCell = {
  id: string;
  nodeId: string;
  nodeType: string;
  source: string;
  libraryModule?: string;
  navigateParentCellId?: string;
};

/** FastAPI often returns `{ "detail": "…" }` or a validation array; unwrap for readable toolbar text. */
function formatComfyResearchApiErrorBody(raw: string, statusText: string): string {
  const t = raw.trim();
  if (!t) return statusText || "Request failed";
  try {
    const j = JSON.parse(t) as { detail?: unknown };
    const d = j.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            const o = item as { msg?: unknown; loc?: unknown };
            const loc = Array.isArray(o.loc) ? o.loc.join(".") : o.loc != null ? String(o.loc) : "";
            const msg = String(o.msg ?? "");
            return loc ? `${loc}: ${msg}` : msg;
          }
          return JSON.stringify(item);
        })
        .join("\n");
    }
    if (d != null) return typeof d === "object" ? JSON.stringify(d) : String(d);
  } catch {
    /* not JSON */
  }
  return t;
}

function toApiDocument(
  nodes: Node[],
  edges: Edge[],
  viewport: Viewport | null,
): GraphDocument {
  return {
    version: 1,
    nodes: nodes.map((n) => {
      const base: GraphDocument["nodes"][number] = {
        id: n.id,
        type: n.type as NodeKind,
        position: { x: n.position.x, y: n.position.y },
        data: (n.data as Record<string, unknown>) ?? {},
      };
      if (n.parentId != null && n.parentId !== "") base.parentId = String(n.parentId);
      if (n.extent === "parent") base.extent = "parent";
      if (n.hidden === true) base.hidden = true;
      if (n.style != null && typeof n.style === "object" && Object.keys(n.style).length > 0) {
        base.style = { ...(n.style as Record<string, unknown>) };
      }
      return base;
    }),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle ?? null,
      targetHandle: e.targetHandle ?? null,
    })),
    viewport,
  };
}

function fromApiDocument(doc: GraphDocument): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = doc.nodes.map((n) => {
    const nn: Node = {
      id: n.id,
      type: n.type,
      position: n.position,
      data: n.data ?? {},
    };
    if (n.parentId != null && n.parentId !== "") nn.parentId = String(n.parentId);
    if (n.extent === "parent") nn.extent = "parent";
    if (n.hidden === true) nn.hidden = true;
    if (n.style != null && typeof n.style === "object") {
      nn.style = { ...(n.style as Record<string, unknown>) } as Node["style"];
    }
    return nn;
  });
  const edges: Edge[] = doc.edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle ?? undefined,
    targetHandle: e.targetHandle ?? undefined,
  }));
  return { nodes, edges };
}

/** Legacy `sweep_viz` → `sweep_data_table` with renamed data keys. */
function migrateLegacySweepNodeTypes(nodes: Node[]): Node[] {
  return nodes.map((n) => {
    if (n.type !== "sweep_viz") return n;
  const d = (n.data ?? {}) as Record<string, unknown>;
  const next: Record<string, unknown> = { ...d };
  if (d.plotSelectedRowIds !== undefined) {
  next.selectedRowIds = d.plotSelectedRowIds;
  delete next.plotSelectedRowIds;
  }
  delete next.plotXParamKey;
  return { ...n, type: "sweep_data_table", data: next };
  });
}

/** Removed node kind: same UI/data as Tensor selector. */
function migrateWeightExtractorToTensorSelector(nodes: Node[]): Node[] {
  return nodes.map((n) => {
    if (String(n.type) !== "weight_extractor") return n;
    return { ...n, type: "tensor_selector" as NodeKind };
  });
}

/** User observable had duplicate handle ids (`observable` target + source); incoming edge now uses `observable_in`. */
function migrateObservableUserTargetHandle(nodes: Node[], edges: Edge[]): Edge[] {
  const idToType = new Map(nodes.map((n) => [n.id, String(n.type)]));
  return edges.map((e) => {
    const tt = e.target ? idToType.get(e.target) : undefined;
    if (tt === "observable_user" && (e.targetHandle ?? "") === "observable") {
      return { ...e, targetHandle: "observable_in" };
    }
    return e;
  });
}

/** ``observable_user`` output is ``observables``; repair legacy ``observable`` / empty source handles to trainer. */
function migrateObservableUserTrainerSourceHandle(nodes: Node[], edges: Edge[]): Edge[] {
  const idToType = new Map(nodes.map((n) => [n.id, String(n.type)]));
  return edges.map((e) => {
    const st = e.source ? idToType.get(e.source) : undefined;
    const tt = e.target ? idToType.get(e.target) : undefined;
    if (st !== "observable_user") return e;
    if (tt !== "trainer" && tt !== "crl_trainer") return e;
    if ((e.targetHandle ?? "") !== "observables") return e;
    const sh = e.sourceHandle ?? "";
    if (sh === "observables") return e;
    if (sh === "observable" || sh === "") return { ...e, sourceHandle: "observables" };
    return e;
  });
}

/** Inset between dashed shell and children when placing / bounding the combined wrapper. */
const COMBINED_COMBINE_PAD = 18;

/**
 * Vertical chrome reserved inside a combined-model shell for header + I/O row.
 * Children use parent-relative Y ≥ PAD + HEAD. XYFlow always paints child nodes above the parent
 * wrapper, so any overlap with the tensor strip steals handle hover / connection cursor.
 * Shell size is refit after layout; this stays large enough for the real header + tensor strip.
 * Keep slack for wrapped titles / dropdown rows so inner nodes are not placed under the strip.
 */
const COMBINED_COMBINE_HEAD = 136;
const COMBINED_MODEL_CHILD_MIN_Y = COMBINED_COMBINE_PAD + COMBINED_COMBINE_HEAD;
const COMBINED_MODEL_CHILD_MIN_X = 16;

/** Extra shell past the bottom-right of the last child (tight pack after refit). */
const COMBINED_SHELL_EDGE_PAD_X = 12;
const COMBINED_SHELL_EDGE_PAD_Y = 10;

/** When RF has not measured a node yet — keep modest so combined shells are not oversized. */
const COMBINED_INNER_NODE_FALLBACK_W = 256;
const COMBINED_INNER_NODE_FALLBACK_H = 256;

const COMBINED_ATOMIC_LAYER_TYPES = new Set([
  "linear_layer",
  "activation_layer",
  "layer_norm_layer",
  "rms_norm_layer",
  "embedding_layer",
  "unembedding_layer",
  "absolute_pos_embed_layer",
  "rotary_embed_layer",
  "local_mixing_layer",
  "afno_patch_embed_layer",
  "afno_spectral_mixer_layer",
  "afno_encoder_block_layer",
  "afno_patch_decode_layer",
  "tensor_splitter",
  "reshape",
  "flatten",
  "einsum",
  "softmax",
  "causal_mask",
  "tensor_slicing",
  "elementwise_transform",
]);

function parseCssPixel(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  if (typeof value !== "string") return null;
  const m = /^(\d+(?:\.\d+)?)px\s*$/i.exec(value.trim());
  if (!m) return null;
  const n = Number(m[1]);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** Best-effort outer size for layout (library paste / combine sizing before measurement). */
function readNodeOuterSize(n: Node): { width: number; height: number } {
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
  if (sw != null) return { width: sw, height: sh ?? COMBINED_INNER_NODE_FALLBACK_H };
  if (sh != null) return { width: COMBINED_INNER_NODE_FALLBACK_W, height: sh };

  const t = String(n.type);
  if (COMBINED_ATOMIC_LAYER_TYPES.has(t)) {
    return { width: 292, height: 248 };
  }
  return { width: COMBINED_INNER_NODE_FALLBACK_W, height: COMBINED_INNER_NODE_FALLBACK_H };
}

/**
 * Union of node boxes in flow coordinates for a subgraph that may be nested (``parentId`` chain).
 * Uses ``clipAbsolutePosition`` so child positions are not treated as flow-space.
 */
function subgraphBoundingRectInFlow(
  nodes: Node[],
  clipIds: Set<string>,
): { x: number; y: number; width: number; height: number } {
  if (nodes.length === 0) return { x: 0, y: 0, width: 0, height: 0 };
  const clipById = new Map(nodes.map((n) => [n.id, n]));
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const n of nodes) {
    const { x, y } = clipAbsolutePosition(n, clipById, clipIds);
    const { width: w, height: h } = readNodeOuterSize(n);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x + w);
    maxY = Math.max(maxY, y + h);
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) return { x: 0, y: 0, width: 0, height: 0 };
  return { x: minX, y: minY, width: Math.max(0, maxX - minX), height: Math.max(0, maxY - minY) };
}

function isMlpModelLowShell(n: Node): boolean {
  return (
    String(n.type) === "mlp_model" && readNodeCanvasLevelMode((n.data ?? {}) as Record<string, unknown>) === "low"
  );
}

function isAttentionModelLowShell(n: Node): boolean {
  return (
    String(n.type) === "attention_only_model" &&
    readNodeCanvasLevelMode((n.data ?? {}) as Record<string, unknown>) === "low"
  );
}

/** Set each ``combined_model`` or low-level ``mlp_model`` shell to tightly wrap its children. */
function refitCombinedModelShellsToChildren(nodes: Node[]): Node[] {
  const combinedIds = new Set(nodes.filter((n) => String(n.type) === "combined_model").map((n) => n.id));
  const mlpLowIds = new Set(nodes.filter((n) => isMlpModelLowShell(n)).map((n) => n.id));
  const attentionLowIds = new Set(nodes.filter((n) => isAttentionModelLowShell(n)).map((n) => n.id));
  if (combinedIds.size === 0 && mlpLowIds.size === 0 && attentionLowIds.size === 0) {
    return ensureCombinedShellChildrenExtent(nodes);
  }

  const shellExtents = (pid: string) => {
    let maxR = COMBINED_MODEL_CHILD_MIN_X;
    let maxB = COMBINED_MODEL_CHILD_MIN_Y;
    for (const c of nodes) {
      if (String(c.parentId) !== pid) continue;
      const { width, height } = readNodeOuterSize(c);
      maxR = Math.max(maxR, c.position.x + width);
      maxB = Math.max(maxB, c.position.y + height);
    }
    return {
      needW: Math.max(236, maxR + COMBINED_SHELL_EDGE_PAD_X),
      needH: Math.max(COMBINED_MODEL_CHILD_MIN_Y + 20, maxB + COMBINED_SHELL_EDGE_PAD_Y),
    };
  };

  return ensureCombinedShellChildrenExtent(
    nodes.map((n) => {
      const t = String(n.type);
      const isCombined = t === "combined_model" && combinedIds.has(n.id);
      const isMlpLow = t === "mlp_model" && mlpLowIds.has(n.id);
      const isAttentionLow = t === "attention_only_model" && attentionLowIds.has(n.id);
      if (!isCombined && !isMlpLow && !isAttentionLow) return n;
      const hasKids = nodes.some((c) => c.parentId === n.id);
      if (!hasKids) return n;
      const { needW, needH } = shellExtents(n.id);
      const st = (n.style ?? {}) as Record<string, unknown>;
      const d = (n.data ?? {}) as Record<string, unknown>;
      return {
        ...n,
        style: { ...st, width: needW, height: needH },
        data: {
          ...d,
          __expandedFrame: { width: needW, height: needH },
        },
      };
    }),
  );
}

function addToNodeFrameSize(style: CSSProperties | undefined, dw: number, dh: number): CSSProperties | undefined {
  if ((dw === 0 && dh === 0) || !style || typeof style !== "object") return style;
  const out = { ...style } as Record<string, unknown>;
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
  return out as CSSProperties;
}

/**
 * Legacy / hand-edited graphs: children placed too close to the top or left of a ``combined_model``
 * overlap the parent's I/O handles in hit-testing (children always paint above the parent in RF).
 */
function migrateCombinedModelChildChromeInset(nodes: Node[]): Node[] {
  const combinedIds = new Set(nodes.filter((n) => n.type === "combined_model").map((n) => n.id));
  const mlpLowShellIds = new Set(nodes.filter((n) => isMlpModelLowShell(n)).map((n) => n.id));
  const attentionLowShellIds = new Set(nodes.filter((n) => isAttentionModelLowShell(n)).map((n) => n.id));
  if (combinedIds.size === 0 && mlpLowShellIds.size === 0 && attentionLowShellIds.size === 0) return nodes;

  const shellParentIds = new Set<string>([...combinedIds, ...mlpLowShellIds, ...attentionLowShellIds]);

  const childrenByParent = new Map<string, Node[]>();
  for (const n of nodes) {
    if (!n.parentId) continue;
    const pid = String(n.parentId);
    if (!shellParentIds.has(pid)) continue;
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
    const isInsetShell =
      (n.type === "combined_model" ||
        (n.type === "mlp_model" && isMlpModelLowShell(n)) ||
        (n.type === "attention_only_model" && isAttentionModelLowShell(n))) &&
      del &&
      (del.dx || del.dy);
    if (isInsetShell) {
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
        style: addToNodeFrameSize(n.style, del.dx, del.dy),
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

/** Older combines used ``zIndex: -1`` so the shell sat under children; that blocked I/O handles (hits went to child wrappers). */
function migrateCombinedModelParentZIndex(nodes: Node[]): Node[] {
  return nodes.map((n) => {
    if (String(n.type) !== "combined_model") return n;
    const style = n.style;
    if (!style || typeof style !== "object") return n;
    const s = style as Record<string, unknown>;
    const z = s.zIndex;
    if (z !== -1 && z !== "-1") return n;
    const { zIndex: _z, ...rest } = s;
    const next = rest as Record<string, unknown>;
    return { ...n, style: Object.keys(next).length > 0 ? next : undefined };
  });
}

const SUBGRAPH_SHELL_CHILD_NODE_CLASS = "cr-rf-node--subgraph-shell-child";

/** Direct children of ``combined_model`` / MLP-low shells: ``extent: parent`` + hit-testing so shell handles stay reachable under child wrappers (XYFlow paints children above the shell). */
function ensureCombinedShellChildrenExtent(nodes: Node[]): Node[] {
  const shellIds = new Set(
    nodes
      .filter((n) => String(n.type) === "combined_model" || isMlpModelLowShell(n) || isAttentionModelLowShell(n))
      .map((n) => n.id),
  );
  if (shellIds.size === 0) return nodes;
  return nodes.map((n) => {
    const pid = n.parentId != null && n.parentId !== "" ? String(n.parentId) : "";
    if (!pid || !shellIds.has(pid)) return n;

    const prevCls = (n.className ?? "").trim();
    const parts = prevCls ? prevCls.split(/\s+/).filter(Boolean) : [];
    const needTag = !parts.includes(SUBGRAPH_SHELL_CHILD_NODE_CLASS);
    const nextCls = needTag ? (parts.length ? `${parts.join(" ")} ${SUBGRAPH_SHELL_CHILD_NODE_CLASS}` : SUBGRAPH_SHELL_CHILD_NODE_CLASS) : prevCls;

    const needExtent = n.extent !== "parent";
    if (!needTag && !needExtent) return n;

    return {
      ...n,
      ...(needExtent ? { extent: "parent" as const } : {}),
      ...(needTag ? { className: nextCls } : {}),
    };
  });
}

/** Legacy graphs: tag combined_model ↔ child auto-wires so they use inward bezier routing. */
function migrateCombinedSubgraphIoEdgeTypes(nodes: Node[], edges: Edge[]): Edge[] {
  const combinedIds = new Set(nodes.filter((n) => String(n.type) === "combined_model").map((n) => n.id));
  return edges.map((e) => {
    if (e.type === COMBINED_SUBGRAPH_IO_EDGE_TYPE) return e;
    const sh = (e.sourceHandle ?? "").trim();
    const th = (e.targetHandle ?? "").trim();
    if (e.source && combinedIds.has(e.source) && sh === "tensor_boundary") {
      return { ...e, type: COMBINED_SUBGRAPH_IO_EDGE_TYPE };
    }
    if (e.target && combinedIds.has(e.target) && th === COMBINED_MODEL_RETURN_TARGET_HANDLE) {
      return { ...e, type: COMBINED_SUBGRAPH_IO_EDGE_TYPE };
    }
    return e;
  });
}

/** Node types that emit a unified ``dataset`` source handle (legacy graphs used ``train_dataset`` / ``test_dataset``). */
const UNIFIED_DATASET_SOURCE_NODE_TYPES = new Set<string>([
  "linear_dataset",
  "random_noise_dataset",
  "memorization_a_dataset",
  "memorization_b_dataset",
  "symbolic_func_dataset",
  "teacher_dataset",
  "token_prediction_dataset",
  "circle_random_walk_dataset",
  "circular_motion_dataset",
  "kepler_2d_dataset",
  "unigram_dataset",
  "bigram_low_rank_dataset",
  "in_context_associative_recall_dataset",
  "uniform_linear_motion_dataset",
  "modular_addition_dataset",
  "dataset_mixer",
  "dataset_mixer_b",
  "pcfg_dataset",
  "dyck_dataset",
  "ngram_language_dataset",
  "formal_language_suite_dataset",
  "scan_dataset",
  "cogs_dataset",
  "listops_dataset",
  "tinystories_dataset",
  "phi1_style_dataset",
  "biography_lm_dataset",
  "relation_tuple_dataset",
  "synthetic_playground_dataset",
  "multi_hop_fact_chain_dataset",
  "mnist_dataset",
  "gaussian_blob_dataset",
  "shape_world_dataset",
  "hole_counting_dataset",
  "diffusion_pde_dataset",
  "reaction_diffusion_dataset",
  "advection_dataset",
]);

/**
 * Legacy: two trainer sockets + two source handles for the same logical dataset.
 * New: one ``dataset`` ↔ ``dataset`` edge per trainer (splits live in node ``data``).
 */
function migrateUnifiedDatasetHandles(nodes: Node[], edges: Edge[]): Edge[] {
  const idToType = new Map(nodes.map((n) => [n.id, String(n.type)]));
  const trainerIds = new Set(nodes.filter((n) => String(n.type) === "trainer").map((n) => n.id));

  let next = edges.map((e) => {
    const st = e.source ? idToType.get(e.source) : undefined;
    if (st && UNIFIED_DATASET_SOURCE_NODE_TYPES.has(st)) {
      const sh = (e.sourceHandle ?? "").trim();
      if (sh === "train_dataset" || sh === "test_dataset") {
        return { ...e, sourceHandle: "dataset" };
      }
    }
    return e;
  });

  next = next.map((e) => {
    if (!e.target || !trainerIds.has(e.target)) return e;
    const th = (e.targetHandle ?? "").trim();
    if (th === "train_dataset" || th === "test_dataset") {
      return { ...e, targetHandle: "dataset" };
    }
    return e;
  });

  const seenTrainerDataset = new Set<string>();
  return next.filter((e) => {
    if (!e.target || !trainerIds.has(e.target)) return true;
    if ((e.targetHandle ?? "").trim() !== "dataset") return true;
    const st = e.source ? idToType.get(e.source) : undefined;
    if (!st || !UNIFIED_DATASET_SOURCE_NODE_TYPES.has(st)) return true;
    const key = `${e.target}|${e.source}`;
    if (seenTrainerDataset.has(key)) return false;
    seenTrainerDataset.add(key);
    return true;
  });
}

/** After node type migration: `tensor_list`/`out_tensor_list` → `stream`/`table`. */
function migrateSweepDataTableHandles(nodes: Node[], edges: Edge[]): Edge[] {
  const idToType = new Map(nodes.map((n) => [n.id, String(n.type)]));
  return edges.map((e) => {
    let next = { ...e };
    const st = e.source ? idToType.get(e.source) : undefined;
    const tt = e.target ? idToType.get(e.target) : undefined;
    if (tt === "sweep_data_table" && e.targetHandle === "tensor_list") {
      next.targetHandle = "stream";
    }
    if (st === "sweep_data_table" && e.sourceHandle === "out_tensor_list") {
      next.sourceHandle = "table";
    }
    return next;
  });
}

/** Drop nodes with no registered renderer; drop dangling edges. If none registered, canvas stays empty. */
function sanitizeLoadedGraph(doc: GraphDocument): { nodes: Node[]; edges: Edge[] } {
  const allowed = new Set(Object.keys(researchNodeTypes));
  const raw = fromApiDocument(doc);
  if (allowed.size === 0) {
    return { nodes: [], edges: [] };
  }
  let nodes = migrateObservableVizNodeTypes(
    migrateWeightExtractorToTensorSelector(migrateLegacySweepNodeTypes(raw.nodes)),
  );
  const ids = new Set(nodes.map((n) => n.id));
  let edges = raw.edges.filter((e) => ids.has(e.source) && ids.has(e.target));
  edges = migrateSweepDataTableHandles(nodes, edges);
  edges = migrateObservableUserTargetHandle(nodes, edges);
  edges = migrateObservableUserTrainerSourceHandle(nodes, edges);
  edges = migrateVizEdgeHandles(nodes, edges);
  nodes = migrateTensorSelectorNodeData(nodes);
  nodes = migrateAttentionOnlyParameterKeys(nodes, edges);
  edges = migrateTensorSelectorOutHandles(nodes, edges);
  edges = migrateLayerStripTensorHandles(nodes, edges);
  edges = migrateUnifiedDatasetHandles(nodes, edges);
  nodes = nodes.filter((n) => allowed.has(String(n.type)));
  nodes = migrateCombinedModelInstanceTitles(nodes);
  nodes = migrateCombinedModelParentZIndex(nodes);
  nodes = nodes.map((n) => {
    const d = (n.data ?? {}) as Record<string, unknown>;
    const collapsed = d.__collapsed === true;
    return { ...n, className: withNodeShellClass(n.className, NODE_COLLAPSED_CLASS, collapsed) };
  });
  nodes = nodes.map((n) => normalizeCollapsedNodeSizing(n));
  nodes = stripInvalidParents(nodes);
  nodes = sortNodesParentBeforeChildren(nodes);
  nodes = migrateCombinedModelChildChromeInset(nodes);
  nodes = ensureCombinedShellChildrenExtent(nodes);
  const kept = new Set(nodes.map((n) => n.id));
  edges = edges.filter((e) => kept.has(e.source) && kept.has(e.target));
  edges = migrateCombinedSubgraphIoEdgeTypes(nodes, edges);
  edges = [...edges, ...repairMissingCombinedModelReturnEdges(nodes, edges)];
  return { nodes, edges };
}

/**
 * Restore a saved combined-model template as an expanded ``combined_model`` shell with the same
 * inner nodes, wiring, and ``input-output`` mode as right after Combine.
 */
function instantiateCombinedModelFromLibraryTemplate(
  existingNodes: Node[],
  templateDoc: GraphDocument,
  flowPos: { x: number; y: number },
  opts: { displayName: string; templateId?: string },
): { nodes: Node[]; edges: Edge[] } | null {
  const { nodes: sanNodes, edges: sanEdges } = sanitizeLoadedGraph(templateDoc);
  if (sanNodes.length === 0) return null;

  const detached = sanNodes.map((n) => ({
    ...n,
    parentId: undefined,
    extent: undefined,
    hidden: false,
  }));

  const { nodes: placedInner, edges: innerEdges } = cloneSubgraphForPaste(
    existingNodes,
    { nodes: detached, edges: sanEdges },
    flowPos,
    false,
  );
  const inner = placedInner.map((n) => ({ ...n, selected: false, hidden: false }));
  const childIdSet = new Set(inner.map((n) => n.id));
  const newCombinedId = `combined_model-${Math.random().toString(36).slice(2, 10)}`;

  const bounds = subgraphBoundingRectInFlow(inner, childIdSet);
  const parentW = Math.max(bounds.width + 2 * COMBINED_COMBINE_PAD, 236);
  const parentH = Math.max(bounds.height + 2 * COMBINED_COMBINE_PAD + COMBINED_COMBINE_HEAD, 168);
  const parentPos = {
    x: bounds.x - COMBINED_COMBINE_PAD,
    y: bounds.y - COMBINED_COMBINE_PAD - COMBINED_COMBINE_HEAD,
  };

  const innerById = new Map(inner.map((n) => [n.id, n]));

  const base = defaultCombinedModelData({
    displayName: opts.displayName.trim().slice(0, 200) || "Combined model",
    templateId: opts.templateId,
    sourceNodeCount: inner.length,
    __expandedFrame: { width: parentW, height: parentH },
    ioMode: "input-output",
  }) as Record<string, unknown>;

  const parent: Node = {
    id: newCombinedId,
    type: "combined_model",
    position: parentPos,
    style: { width: parentW, height: parentH },
    data: withNewInstanceTitle(existingNodes, "combined_model", base),
    selected: true,
  };

  const children: Node[] = inner.map((n) => {
    const pid = n.parentId != null && n.parentId !== "" ? String(n.parentId) : "";
    const rootInClip = !pid || !childIdSet.has(pid);
    const abs = clipAbsolutePosition(n, innerById, childIdSet);
    if (rootInClip) {
      return {
        ...n,
        parentId: newCombinedId,
        position: { x: abs.x - parentPos.x, y: abs.y - parentPos.y },
        extent: "parent" as const,
        selected: false,
        hidden: false,
      };
    }
    return { ...n, selected: false, hidden: false };
  });

  const bridgeRewire = computeCombinedModelBridgeRewire(childIdSet, innerEdges, newCombinedId, children);
  const mergedEdges = [...innerEdges, ...bridgeRewire];
  let layoutNodes = sortNodesParentBeforeChildren([parent, ...children]);
  layoutNodes = migrateCombinedModelChildChromeInset(layoutNodes);
  layoutNodes = refitCombinedModelShellsToChildren(layoutNodes);
  const repaired = repairMissingCombinedModelReturnEdges(layoutNodes, mergedEdges);
  let finalEdges = [...mergedEdges, ...repaired];
  finalEdges = migrateCombinedSubgraphIoEdgeTypes(layoutNodes, finalEdges);
  return { nodes: layoutNodes, edges: finalEdges };
}

const VIZ_NODE_TYPES_WITH_TENSOR_IN = new Set([
  "training_visualization",
  "observable_viz",
  "tensor_viz_general",
  "tensor_viz_0d",
  "sweep_data_table",
  "tensor_viz_1d",
  "tensor_viz_2d",
]);

/**
 * Legacy handle ids: `plot` → `tensor` or `tensor_list`; `tensors` → `tensor_list`;
 * activation `tensor` → `tensor_list`; training viz `tensor`/`out_tensor` → `tensor_list`/`out_tensor_list`.
 */
/** Multi-output tensor selector: legacy ``selected_tensor`` / empty → ``tensor_1``. */
function migrateTensorSelectorOutHandles(nodes: Node[], edges: Edge[]): Edge[] {
  const srcTypes = new Map(nodes.map((n) => [n.id, String(n.type)]));
  return edges.map((e) => {
    const st = e.source ? srcTypes.get(e.source) : undefined;
    const sh = e.sourceHandle ?? "";
    if (st === "tensor_selector" && (sh === "selected_tensor" || sh === "")) {
      return { ...e, sourceHandle: "tensor_1" };
    }
    return e;
  });
}

function migrateTensorSelectorNodeData(nodes: Node[]): Node[] {
  return nodes.map((n) => {
    if (String(n.type) !== "tensor_selector") return n;
    const d = (n.data ?? {}) as Record<string, unknown>;
    const hasKeys = Array.isArray(d.selectedTensorKeys) && (d.selectedTensorKeys as unknown[]).length > 0;
    if (hasKeys) return n;
    const key = String(d.selectedTensorKey ?? "").trim();
    if (!key) return n;
    return { ...n, data: { ...d, selectedTensorKeys: [key] } };
  });
}

/**
 * Attention-only models are now wrapped in ``AttentionTokenPredictBundle``, so their
 * attention parameters are named ``block.w_*.{weight,bias}``. Restrict this migration
 * to selectors and weight caches downstream of that model type.
 */
function migrateAttentionOnlyParameterKeys(nodes: Node[], edges: Edge[]): Node[] {
  const attentionModelIds = new Set(
    nodes.filter((n) => String(n.type) === "attention_only_model").map((n) => n.id),
  );
  const weightNodeIds = new Set(
    edges
      .filter((e) => attentionModelIds.has(e.source))
      .map((e) => e.target)
      .filter((id): id is string => Boolean(id)),
  );
  const selectorIds = new Set(
    edges
      .filter((e) => weightNodeIds.has(e.source))
      .map((e) => e.target)
      .filter((id): id is string => Boolean(id)),
  );
  return nodes.map((node) => {
    const data = (node.data ?? {}) as Record<string, unknown>;
    if (selectorIds.has(node.id) && String(node.type) === "tensor_selector") {
      return { ...node, data: migrateAttentionOnlyTensorSelectorData(data) };
    }
    if (weightNodeIds.has(node.id) && String(node.type) === "model_weight_tensors") {
      return { ...node, data: migrateAttentionOnlyWeightTensorPayloads(data) };
    }
    return node;

  });
}

const ATOMIC_LAYER_NODE_TYPES = new Set([
  "linear_layer",
  "activation_layer",
  "layer_norm_layer",
  "rms_norm_layer",
  "embedding_layer",
  "unembedding_layer",
  "absolute_pos_embed_layer",
  "rotary_embed_layer",
  "local_mixing_layer",
  "pairwise_rbf_layer",
  "equivariant_message_layer",
  "energy_readout_layer",
  "relative_pose_encoder_layer",
  "distance_contact_layer",
]);
const FULL_MODEL_STRIP_NODE_TYPES = new Set([
  "mlp_model",
  "gated_mlp_model",
  "moe_mlp_model",
  "mlp_token_model",
  "gated_mlp_token_model",
  "moe_mlp_token_model",
  "kan_model",
  "residual_ln_model",
  "attention_only_model",
  "linear_attention_model",
  "diagonal_ssm_token_model",
  "rwkv_time_mix_token_model",
  "hyena_like_conv_model",
  "slot_attention_token_model",
  "diffusion_score_model",
]);

function nodeUsesPairedTensorStrip(n: Node): boolean {
  const t = String(n.type);
  if (ATOMIC_LAYER_NODE_TYPES.has(t) || FULL_MODEL_STRIP_NODE_TYPES.has(t)) {
    return readNodeCanvasIoMode((n.data ?? {}) as Record<string, unknown>) === "input-output";
  }
  if (t === "combined_model") {
    return readNodeCanvasIoMode((n.data ?? {}) as Record<string, unknown>) === "input-output";
  }
  return false;
}

/** Legacy paired strip used duplicate ``tensor`` ids; remap so XYFlow registers left/right handles. */
function migrateLayerStripTensorHandles(nodes: Node[], edges: Edge[]): Edge[] {
  const idToNode = new Map(nodes.map((n) => [n.id, n]));
  return edges.map((e) => {
    const next = { ...e };
    const tgt = e.target ? idToNode.get(e.target) : undefined;
    const src = e.source ? idToNode.get(e.source) : undefined;
    const th = (e.targetHandle ?? "").trim();
    const sh = (e.sourceHandle ?? "").trim();
    if (tgt && nodeUsesPairedTensorStrip(tgt) && th === "tensor") {
      next.targetHandle = LAYER_STRIP_TARGET_HANDLE;
    }
    if (src && nodeUsesPairedTensorStrip(src) && sh === "tensor") {
      next.sourceHandle = LAYER_STRIP_SOURCE_HANDLE;
    }
    return next;
  });
}

function migrateVizEdgeHandles(nodes: Node[], edges: Edge[]): Edge[] {
  const idToType = new Map(nodes.map((n) => [n.id, String(n.type)]));
  return edges.map((e) => {
    const st = e.source ? idToType.get(e.source) : undefined;
    const tt = e.target ? idToType.get(e.target) : undefined;
    let next = { ...e };

    if (e.targetHandle === "plot" && tt && VIZ_NODE_TYPES_WITH_TENSOR_IN.has(tt)) {
      next.targetHandle =
        tt === "training_visualization"
          ? "tensor_list"
          : tt === "sweep_data_table"
            ? "stream"
            : "tensor";
    }
    if (tt === "tensor_selector" && e.targetHandle === "tensors") {
      next.targetHandle = "tensor_list";
    }
    if (st === "activation" && e.sourceHandle === "tensor") {
      next.sourceHandle = "tensor_list";
    }
    if (tt === "training_visualization" && e.targetHandle === "tensor") {
      next.targetHandle = "tensor_list";
    }
    if (st === "training_visualization" && e.sourceHandle === "out_tensor") {
      next.sourceHandle = "out_tensor_list";
    }
    return next;
  });
}

function newProjectId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `job-${Math.random().toString(36).slice(2, 14)}`;
}

const NODE_COLLAPSED_CLASS = "cr-node-shell--collapsed";
/** Stashed `node.style` width/height while collapsed so expand can restore after resize. */
const NODE_COLLAPSE_RESTORE_STYLE_KEY = "__collapseRestoreStyle";

type CollapseRestoreStyle = { width?: unknown; height?: unknown };

function styleHasExplicitSize(style: unknown): boolean {
  if (!style || typeof style !== "object") return false;
  const s = style as Record<string, unknown>;
  const w = s.width;
  const h = s.height;
  const has = (v: unknown) => v != null && v !== "";
  return has(w) || has(h);
}

/** Collapsed shell on the React Flow node: keep user width, drop fixed height so the box fits header + I/O. */
function collapsedCompactOuterStyle(prev: Record<string, unknown>): CSSProperties {
  const out: CSSProperties = { height: "auto", minHeight: 0 };
  const w = prev.width;
  if (w != null && w !== "") (out as Record<string, unknown>).width = w;
  return out;
}

function stripInvalidParents(nodes: Node[]): Node[] {
  const ids = new Set(nodes.map((n) => n.id));
  return nodes.map((n) => {
    if (n.parentId && !ids.has(String(n.parentId))) {
      return { ...n, parentId: undefined, extent: undefined };
    }
    return n;
  });
}

/** Collapsed + explicit RF height leaves a tall empty shell; compact height, stash full size for expand. */
function normalizeCollapsedNodeSizing(n: Node): Node {
  if (n.type === "combined_model" || isMlpModelLowShell(n)) return n;
  const d = (n.data ?? {}) as Record<string, unknown>;
  if (d.__collapsed !== true) return n;
  if (!styleHasExplicitSize(n.style)) return n;
  const st = (n.style ?? {}) as Record<string, unknown>;
  const stash = d[NODE_COLLAPSE_RESTORE_STYLE_KEY] as CollapseRestoreStyle | undefined;
  const widthForCompact = stash?.width != null && stash.width !== "" ? stash.width : st.width;
  if (stash) {
    return {
      ...n,
      style: collapsedCompactOuterStyle({ width: widthForCompact } as Record<string, unknown>),
    };
  }
  return {
    ...n,
    style: collapsedCompactOuterStyle(st),
    data: {
      ...d,
      [NODE_COLLAPSE_RESTORE_STYLE_KEY]: { width: st.width, height: st.height },
    },
  };
}

function withNodeShellClass(existing: string | undefined, addClass: string, enabled: boolean): string | undefined {
  const parts = new Set(
    String(existing ?? "")
      .split(/\s+/)
      .map((s) => s.trim())
      .filter(Boolean),
  );
  if (enabled) parts.add(addClass);
  else parts.delete(addClass);
  return parts.size ? Array.from(parts).join(" ") : undefined;
}

/** All strict descendants of ``rootId`` (children, grandchildren, …) for parent-child subtrees. */
function collectStrictDescendantIds(nodes: Node[], rootId: string): Set<string> {
  const byParent = new Map<string, string[]>();
  for (const n of nodes) {
    if (n.parentId == null || n.parentId === "") continue;
    const p = String(n.parentId);
    if (!byParent.has(p)) byParent.set(p, []);
    byParent.get(p)!.push(n.id);
  }
  const out = new Set<string>();
  const stack = [...(byParent.get(rootId) ?? [])];
  while (stack.length) {
    const id = stack.pop()!;
    if (out.has(id)) continue;
    out.add(id);
    for (const c of byParent.get(id) ?? []) stack.push(c);
  }
  return out;
}

/** Graph distance from ``ancestorId`` down to ``nodeId`` (1 = direct child). */
function descendantDepthFromAncestor(nodes: Node[], ancestorId: string, nodeId: string): number {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  let depth = 0;
  let cur: string | undefined = nodeId;
  while (cur && cur !== ancestorId) {
    const n = byId.get(cur);
    if (!n?.parentId) break;
    const p = String(n.parentId);
    depth++;
    if (p === ancestorId) return depth;
    cur = p;
  }
  return depth;
}

/**
 * After expanding a ``combined_model`` at ``rootId``, clear ``hidden`` for its subtree then hide
 * direct children of every nested ``combined_model`` that remains collapsed (mirrors inner expand).
 */
function reapplyCollapsedCombinedHiddenInSubtree(nodes: Node[], rootId: string): Node[] {
  const desc = collectStrictDescendantIds(nodes, rootId);
  let next = nodes.map((n) => (desc.has(n.id) ? { ...n, hidden: false } : n));
  const byId = new Map(next.map((n) => [n.id, n]));
  const collapsedCombinedIds = [...desc].filter((id) => {
    const n = byId.get(id);
    const d = (n?.data ?? {}) as Record<string, unknown>;
    return n?.type === "combined_model" && d.__collapsed === true;
  });
  collapsedCombinedIds.sort(
    (a, b) => descendantDepthFromAncestor(next, rootId, a) - descendantDepthFromAncestor(next, rootId, b),
  );
  for (const cmId of collapsedCombinedIds) {
    const hideIds = new Set(next.filter((n) => n.parentId === cmId).map((n) => n.id));
    next = next.map((n) => (hideIds.has(n.id) ? { ...n, hidden: true } : n));
  }
  return next;
}

function formatProjectTabTitle(id: string): string {
  const max = 13;
  const slice = id.slice(0, max);
  return `Project ${slice}${id.length > max ? "…" : ""}`;
}

function formatCanvasTitle(id: string): string {
  const max = 10;
  const slice = id.slice(0, max);
  return `Canvas ${slice}${id.length > max ? "…" : ""}`;
}

/** The single graph a project owns. */
type GraphCanvas = {
  id: string;
  title: string;
  nodes: Node[];
  edges: Edge[];
  savedViewport: Viewport | null | undefined;
  viewportApplyNonce: number;
  dirty: boolean;
  /**
   * When set (Chrome/Edge File System Access API), Graph → Save overwrites this file.
   * Not persisted in workspace JSON.
   */
  localGraphFileHandle?: FileSystemFileHandle;
  /**
   * When the graph was opened from the Templates rail, Graph → Save POSTs an update
   * to ``/api/graph-library/{kind}`` (same entry id). Cleared when loading from disk.
   */
  librarySource?: {
    kind: "workflow" | "template";
    entryId: string;
    tier: GraphFileExportTier;
  };
};

/** A project owns exactly one canvas. Use a separate project for a controlled variant. */
type WorkspaceProject = {
  id: string;
  title: string;
  canvas: GraphCanvas;
};

/** Immutably replace a project's canvas; returns the same project when `fn` is a no-op. */
function withProjectCanvas(
  p: WorkspaceProject,
  fn: (canvas: GraphCanvas) => GraphCanvas,
): WorkspaceProject {
  const next = fn(p.canvas);
  return next === p.canvas ? p : { ...p, canvas: next };
}

/** `withProjectCanvas` over a project list, keyed by project id. */
function mapProjectCanvas(
  list: WorkspaceProject[],
  projectId: string,
  fn: (canvas: GraphCanvas) => GraphCanvas,
): WorkspaceProject[] {
  return list.map((p) => (p.id === projectId ? withProjectCanvas(p, fn) : p));
}

function createEmptyCanvas(id: string, title?: string): GraphCanvas {
  return {
    id,
    title: title ?? formatCanvasTitle(id),
    nodes: [],
    edges: [],
    savedViewport: undefined,
    viewportApplyNonce: 0,
    dirty: false,
  };
}

function createEmptyWorkspaceProject(projectId: string): WorkspaceProject {
  return {
    id: projectId,
    title: formatProjectTabTitle(projectId),
    canvas: createEmptyCanvas(newProjectId(), "base"),
  };
}

function workspaceSnapshotToProjects(snap: WorkspaceSnapshotDTO): WorkspaceProject[] {
  return snap.projects.map((p) => {
    const { nodes, edges } = sanitizeLoadedGraph(p.canvas.document);
    return {
      id: p.id,
      title: p.title,
      canvas: {
        id: p.canvas.id,
        title: p.canvas.title,
        nodes,
        edges,
        savedViewport: p.canvas.document.viewport ?? null,
        viewportApplyNonce: 1,
        dirty: false,
      },
    };
  });
}

function buildWorkspaceSnapshotDTO(
  projects: WorkspaceProject[],
  activeProjectId: string,
  liveCanvas: { projectId: string; document: GraphDocument },
): WorkspaceSnapshotDTO {
  return {
    version: 3,
    active_project_id: activeProjectId,
    projects: projects.map((p) => ({
      id: p.id,
      title: p.title,
      canvas: {
        id: p.canvas.id,
        title: p.canvas.title,
        document:
          liveCanvas.projectId === p.id
            ? liveCanvas.document
            : toApiDocument(p.canvas.nodes, p.canvas.edges, p.canvas.savedViewport ?? null),
      },
    })),
  };
}

function parseGraphDocumentJson(text: string): GraphDocument {
  let raw: unknown;
  try {
    raw = JSON.parse(text) as unknown;
  } catch {
    throw new Error("File is not valid JSON.");
  }
  if (raw === null || typeof raw !== "object") {
    throw new Error("Invalid graph file.");
  }
  const o = raw as Record<string, unknown>;
  if (!Array.isArray(o.nodes) || !Array.isArray(o.edges)) {
    throw new Error("Invalid graph file (missing nodes or edges).");
  }
  return {
    version: typeof o.version === "number" ? o.version : 1,
    nodes: o.nodes as GraphDocument["nodes"],
    edges: o.edges as GraphDocument["edges"],
    viewport: (o.viewport as GraphDocument["viewport"]) ?? null,
  };
}

function safeGraphFilenameStem(raw: string): string {
  const safe = raw.replace(/[^\w.-]+/g, "_").slice(0, 48) || "graph";
  return safe.replace(/\.json$/i, "");
}

function downloadGraphJson(doc: GraphDocument, filenameBase: string) {
  const stem = safeGraphFilenameStem(filenameBase);
  const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${stem}.json`;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

type ShowSaveFilePickerFn = (options?: {
  suggestedName?: string;
  types?: Array<{ description?: string; accept: Record<string, string[]> }>;
}) => Promise<FileSystemFileHandle>;

/**
 * Uses the File System Access API save dialog when available so the user picks path + name.
 * Falls back to a download into the browser default folder when the API is missing (e.g. Firefox).
 */
type SaveGraphFileOutcome = { outcome: "saved" | "cancelled"; handle?: FileSystemFileHandle };

async function saveGraphJsonWithUserLocation(
  doc: GraphDocument,
  filenameBase: string,
): Promise<SaveGraphFileOutcome> {
  const stem = safeGraphFilenameStem(filenameBase);
  const suggestedName = `${stem}.json`;
  const json = JSON.stringify(doc, null, 2);

  const showSave = (window as unknown as { showSaveFilePicker?: ShowSaveFilePickerFn })
    .showSaveFilePicker;

  if (typeof showSave === "function") {
    try {
      const handle = await showSave({
        suggestedName,
        types: [
          {
            description: "ComfyResearch graph JSON",
            accept: { "application/json": [".json"] },
          },
        ],
      });
      const writable = await handle.createWritable();
      await writable.write(json);
      await writable.close();
      return { outcome: "saved", handle };
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        return { outcome: "cancelled" };
      }
      downloadGraphJson(doc, stem);
      return { outcome: "saved" };
    }
  }

  downloadGraphJson(doc, stem);
  return { outcome: "saved" };
}

async function writeGraphDocumentToFileHandle(
  handle: FileSystemFileHandle,
  doc: GraphDocument,
): Promise<void> {
  const writable = await handle.createWritable();
  await writable.write(JSON.stringify(doc, null, 2));
  await writable.close();
}

type ShowOpenFilePickerFn = (options?: {
  multiple?: boolean;
  types?: Array<{ description?: string; accept: Record<string, string[]> }>;
}) => Promise<FileSystemFileHandle[]>;

function ViewportSync({
  saved,
  applyId,
}: {
  saved: Viewport | null | undefined;
  applyId: number;
}) {
  const { setViewport, fitView } = useReactFlow();

  useEffect(() => {
    if (saved === undefined) return;
    requestAnimationFrame(() => {
      if (saved) setViewport(saved);
      else fitView({ padding: 0.2 });
    });
  }, [applyId, fitView, saved, setViewport]);

  return null;
}

type SaveFlyoutId = "file" | "workflow" | "template";
// 25% × 2^(n/3), n = 0…12: every step is the cube root of two times the last.
const CANVAS_ZOOM_LEVELS = Array.from({ length: 13 }, (_, index) => 0.25 * 2 ** (index / 3));

type GraphToolbarProps = {
  onSaveToServer: () => Promise<void>;
  onLoadFromServer: () => Promise<void>;
  onSaveGraphToSourceFile: () => void | Promise<void>;
  canSaveGraphToSourceFile: boolean;
  onSaveGraphToFileTier: (tier: GraphFileExportTier) => void | Promise<void>;
  onSaveGraphAsLibrary: (
    kind: SavedGraphKind,
    tier: GraphFileExportTier,
  ) => void | Promise<void>;
  onExportCanvasPng: () => void | Promise<void>;
  onExportCanvasPdf: () => void | Promise<void>;
  onOpenGraphCompare: () => void;
  onGraphFileLoaded: (doc: GraphDocument, fileHandle?: FileSystemFileHandle | null) => void;
  onGraphFileError: (message: string) => void;
  onAutoLayoutCanvas: () => void;
  onAutoConnectCanvas: () => void;
  onClearCanvas: () => void;
  loading: boolean;
  error: string | null;
  notice: string | null;
};

export function GraphToolbar({
  onSaveToServer,
  onLoadFromServer,
  onSaveGraphToSourceFile,
  canSaveGraphToSourceFile,
  onSaveGraphToFileTier,
  onSaveGraphAsLibrary,
  onExportCanvasPng,
  onExportCanvasPdf,
  onOpenGraphCompare,
  onGraphFileLoaded,
  onGraphFileError,
  onAutoLayoutCanvas,
  onAutoConnectCanvas,
  onClearCanvas,
  loading,
  error,
  notice,
}: GraphToolbarProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [saveFlyout, setSaveFlyout] = useState<SaveFlyoutId | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const { setViewport, fitView } = useReactFlow();
  const { x: viewportX, y: viewportY, zoom: viewportZoom } = useViewport();
  const stepZoom = useCallback(
    (direction: -1 | 1) => {
      const epsilon = 0.002;
      const next =
        direction > 0
          ? CANVAS_ZOOM_LEVELS.find((level) => level > viewportZoom + epsilon) ?? CANVAS_ZOOM_LEVELS.at(-1)!
          : [...CANVAS_ZOOM_LEVELS].reverse().find((level) => level < viewportZoom - epsilon) ?? CANVAS_ZOOM_LEVELS[0];
      void setViewport({ x: viewportX, y: viewportY, zoom: next }, { duration: 180 });
    },
    [setViewport, viewportX, viewportY, viewportZoom],
  );

  const closeSaveMenus = useCallback(() => {
    setSaveFlyout(null);
    setMenuOpen(false);
  }, []);

  useEffect(() => {
    if (!menuOpen) setSaveFlyout(null);
  }, [menuOpen]);

  useDismissOnOutsidePointer(menuOpen, () => setMenuOpen(false), menuRef);

  const combinedErr = error ?? notice;

  return (
    <Panel position="top-center" className="cr-toolbar cr-toolbar--panel">
      <div className="cr-brand">ComfyResearch</div>
      <div className="cr-toolbar__group cr-toolbar__canvas-actions" aria-label="Canvas controls">
        <button type="button" className="cr-toolbar__canvas-btn" aria-label="Auto layout graph" title="Auto layout graph" onClick={onAutoLayoutCanvas}>
          <svg viewBox="0 0 24 24" aria-hidden><path d="M4 4h6v4H4V4zm10 0h6v4h-6V4zM4 16h6v4H4v-4zm10 0h6v4h-6v-4zM7 9h2v2h6V9h2v4H7V9zm5 5h2v2h-2v-2z" fill="currentColor" /></svg>
        </button>
        <button type="button" className="cr-toolbar__canvas-btn" aria-label="Auto-connect trainer wiring" title="Auto-connect trainer wiring" onClick={onAutoConnectCanvas}>
          <svg viewBox="0 0 24 24" aria-hidden><path d="m7.1 13.1 2.8-2.8 1.4 1.4-2.8 2.8a2.5 2.5 0 1 1-3.5-3.5l2.8-2.8 1.4 1.4-2.8 2.8a.5.5 0 1 0 .7.7m9.8-9.9a2.5 2.5 0 0 1 3.5 3.5l-2.8 2.8-1.4-1.4L19 5.3a.5.5 0 0 0-.7-.7l-2.8 2.8-1.4-1.4zM8.9 17.5l6.6-6.6 1.4 1.4-6.6 6.6z" fill="currentColor" /></svg>
        </button>
        <button type="button" className="cr-toolbar__canvas-btn" aria-label="Clear canvas" title="Clear canvas" onClick={onClearCanvas}>
          <svg viewBox="0 0 24 24" aria-hidden><path d="M9 3h6a1 1 0 0 1 1 1v1h4v2H4V5h4V4a1 1 0 0 1 1-1zm-4 6h14l-1 12a2 2 0 0 1-2 1.9H8a2 2 0 0 1-2-1.9L5 9zm4 2v8h2v-8H9zm4 0v8h2v-8h-2z" fill="currentColor" /></svg>
        </button>
        <button
          type="button"
          className="cr-toolbar__canvas-btn"
          aria-label="Fit graph to view"
          title="Fit graph to view"
          onClick={() => void fitView({ padding: 0.12, duration: 320 })}
        >
          <svg viewBox="0 0 24 24" aria-hidden>
            <path d="M9 5H5v4m10-4h4v4m0 6v4h-4M9 19H5v-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
        <button type="button" className="cr-toolbar__canvas-btn" aria-label="Zoom out" title="Zoom out" onClick={() => stepZoom(-1)}>
          <svg viewBox="0 0 24 24" aria-hidden><path d="M6 12h12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
        </button>
        <div className="cr-toolbar__canvas-zoom-readout" aria-label={`Current zoom: ${Math.round(viewportZoom * 100)}%`}>
          <svg viewBox="0 0 24 24" aria-hidden><circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.8" /><path d="m16 16 4 4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
          <span>{Math.round(viewportZoom * 100)}%</span>
        </div>
        <button type="button" className="cr-toolbar__canvas-btn" aria-label="Zoom in" title="Zoom in" onClick={() => stepZoom(1)}>
          <svg viewBox="0 0 24 24" aria-hidden><path d="M12 6v12M6 12h12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
        </button>
      </div>
      {/* <button
        type="button"
        className={`cr-code-mode-toggle${codeMode ? " cr-code-mode-toggle--on" : ""}`}
        aria-pressed={codeMode}
        title={
          codeMode
            ? "Code mode on — canvas and Python notebook side by side"
            : "Code mode off — click to split the workbench and auto-append Python sketches for new nodes"
        }
        onClick={() => onCodeModeChange(!codeMode)}
      >
        Code
      </button> */}
      <div className="cr-graph-menu" ref={menuRef}>
        <button
          type="button"
          className="cr-graph-menu__trigger"
          aria-expanded={menuOpen}
          aria-haspopup="menu"
          onClick={() => setMenuOpen((o) => !o)}
        >
          <span className="cr-graph-menu__trigger-icon" aria-hidden>
            ⬡
          </span>
          Graph
          <span className="cr-graph-menu__chevron" aria-hidden>
            ▾
          </span>
        </button>
        {menuOpen ? (
          <div className="cr-graph-menu__dropdown" role="menu">
            <button
              type="button"
              role="menuitem"
              className="cr-graph-menu__item"
              disabled={!canSaveGraphToSourceFile}
              title={
                canSaveGraphToSourceFile
                  ? "Overwrite the open file on disk (Small export), or update the template on the server (same tier as when opened)."
                  : "Open a template from the Templates rail, load a graph from disk (Chrome/Edge), or use Save graph to file first."
              }
              onClick={() => {
                if (!canSaveGraphToSourceFile) return;
                setMenuOpen(false);
                void onSaveGraphToSourceFile();
              }}
            >
              Save
            </button>
            <div className="cr-graph-menu__nest" role="none">
              <button
                type="button"
                role="menuitem"
                aria-expanded={saveFlyout === "file"}
                aria-haspopup="menu"
                className="cr-graph-menu__item cr-graph-menu__item--nested-trigger"
                onClick={() => setSaveFlyout((f) => (f === "file" ? null : "file"))}
              >
                <span>Save graph to file</span>
                <span className="cr-graph-menu__nested-chevron" aria-hidden>
                  ▸
                </span>
              </button>
              {saveFlyout === "file" ? (
                <div className="cr-graph-menu__nested" role="menu" aria-label="Save graph size">
                  <button
                    type="button"
                    role="menuitem"
                    className="cr-graph-menu__item cr-graph-menu__item--nested"
                    onClick={() => {
                      closeSaveMenus();
                      void onSaveGraphToFileTier("small");
                    }}
                  >
                    Small (only graph)
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    className="cr-graph-menu__item cr-graph-menu__item--nested"
                    onClick={() => {
                      closeSaveMenus();
                      void onSaveGraphToFileTier("medium");
                    }}
                  >
                    Medium (graph + plot)
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    className="cr-graph-menu__item cr-graph-menu__item--nested"
                    onClick={() => {
                      closeSaveMenus();
                      void onSaveGraphToFileTier("large");
                    }}
                  >
                    Large (graph + plot + model)
                  </button>
                </div>
              ) : null}
            </div>
            <button
              type="button"
              role="menuitem"
              className="cr-graph-menu__item"
              onClick={() => {
                setMenuOpen(false);
                void onExportCanvasPng();
              }}
            >
              Save canvas as PNG
            </button>
            <button
              type="button"
              role="menuitem"
              className="cr-graph-menu__item"
              onClick={() => {
                setMenuOpen(false);
                void onExportCanvasPdf();
              }}
            >
              Save canvas as PDF
            </button>
            <div className="cr-graph-menu__nest" role="none">
              <button
                type="button"
                role="menuitem"
                aria-expanded={saveFlyout === "template"}
                aria-haspopup="menu"
                className="cr-graph-menu__item cr-graph-menu__item--nested-trigger"
                onClick={() => setSaveFlyout((f) => (f === "template" ? null : "template"))}
              >
                <span>Save graph as template</span>
                <span className="cr-graph-menu__nested-chevron" aria-hidden>
                  ▸
                </span>
              </button>
              {saveFlyout === "template" ? (
                <div className="cr-graph-menu__nested" role="menu" aria-label="Save template size">
                  <button
                    type="button"
                    role="menuitem"
                    className="cr-graph-menu__item cr-graph-menu__item--nested"
                    onClick={() => {
                      closeSaveMenus();
                      void onSaveGraphAsLibrary("template", "small");
                    }}
                  >
                    Small (only graph)
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    className="cr-graph-menu__item cr-graph-menu__item--nested"
                    onClick={() => {
                      closeSaveMenus();
                      void onSaveGraphAsLibrary("template", "medium");
                    }}
                  >
                    Medium (graph + plot)
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    className="cr-graph-menu__item cr-graph-menu__item--nested"
                    onClick={() => {
                      closeSaveMenus();
                      void onSaveGraphAsLibrary("template", "large");
                    }}
                  >
                    Large (graph + plot + model)
                  </button>
                </div>
              ) : null}
            </div>
            <button
              type="button"
              role="menuitem"
              className="cr-graph-menu__item"
              onClick={() => {
                setMenuOpen(false);
                const showOpen = (window as unknown as { showOpenFilePicker?: ShowOpenFilePickerFn })
                  .showOpenFilePicker;
                if (typeof showOpen === "function") {
                  void (async () => {
                    try {
                      const handles = await showOpen({
                        multiple: false,
                        types: [
                          {
                            description: "ComfyResearch graph JSON",
                            accept: { "application/json": [".json"] },
                          },
                        ],
                      });
                      const handle = handles[0];
                      if (!handle) return;
                      const file = await handle.getFile();
                      const text = await file.text();
                      try {
                        const doc = parseGraphDocumentJson(text);
                        onGraphFileLoaded(doc, handle);
                      } catch (err) {
                        onGraphFileError(err instanceof Error ? err.message : String(err));
                      }
                    } catch (e) {
                      if (e instanceof DOMException && e.name === "AbortError") return;
                      fileRef.current?.click();
                    }
                  })();
                  return;
                }
                fileRef.current?.click();
              }}
            >
              Load graph from file…
            </button>
            <button
              type="button"
              role="menuitem"
              className="cr-graph-menu__item"
              onClick={() => {
                setMenuOpen(false);
                onOpenGraphCompare();
              }}
            >
              Compare canvas to another…
            </button>
            <div className="cr-graph-menu__sep" role="separator" />
            <button
              type="button"
              role="menuitem"
              className="cr-graph-menu__item"
              onClick={() => {
                setMenuOpen(false);
                void onSaveToServer();
              }}
            >
              Save to server
            </button>
            <button
              type="button"
              role="menuitem"
              className="cr-graph-menu__item"
              onClick={() => {
                setMenuOpen(false);
                void onLoadFromServer();
              }}
            >
              Load from server
            </button>
          </div>
        ) : null}
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="cr-graph-menu__file"
          onChange={(e) => {
            const input = e.target;
            const file = input.files?.[0];
            input.value = "";
            if (!file) return;
            void file.text().then(
              (text) => {
                try {
                  const doc = parseGraphDocumentJson(text);
                  onGraphFileLoaded(doc, null);
                } catch (err) {
                  onGraphFileError(err instanceof Error ? err.message : String(err));
                }
              },
              () => onGraphFileError("Could not read the selected file."),
            );
          }}
        />
      </div>
      {loading ? <span className="cr-toolbar__meta">Working…</span> : null}
      {combinedErr ? (
        <span className="cr-toolbar__meta cr-toolbar__meta--err">{combinedErr}</span>
      ) : null}
    </Panel>
  );
}

type ProjectTabBarProps = {
  projects: WorkspaceProject[];
  activeProjectId: string;
  trainByProjectId: Record<string, ProjectTrainSummary>;
  onSelect: (id: string) => void;
  onAdd: () => void;
  onClose: (id: string, e: ReactMouseEvent<HTMLButtonElement>) => void;
  onRenameProject: (id: string, title: string) => void;
};

function ProjectTabBar({
  projects,
  activeProjectId,
  trainByProjectId,
  onSelect,
  onAdd,
  onClose,
  onRenameProject,
}: ProjectTabBarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!editingId) return;
    const el = inputRef.current;
    if (!el) return;
    queueMicrotask(() => {
      el.focus();
      el.select();
    });
  }, [editingId]);

  const beginRename = useCallback((p: WorkspaceProject, e: ReactMouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setEditingId(p.id);
    setDraft(p.title);
  }, []);

  const commitRename = useCallback(() => {
    if (!editingId) return;
    const p = projects.find((x) => x.id === editingId);
    const next = draft.trim();
    if (p && next && next !== p.title) {
      onRenameProject(editingId, next);
    }
    setEditingId(null);
  }, [draft, editingId, onRenameProject, projects]);

  const cancelRename = useCallback(() => {
    setEditingId(null);
  }, []);

  return (
    <div className="cr-project-tabs">
      <div className="cr-project-tabs__list" role="tablist" aria-label="Open projects">
        {projects.map((p) => {
          const train = trainByProjectId[p.id];
          const trainActive = Boolean(train?.hasActiveTraining);
          const tabClass = [
            "cr-project-tabs__tab",
            p.id === activeProjectId ? "cr-project-tabs__tab--active" : "",
            trainActive ? (train?.paused ? "cr-project-tabs__tab--train-paused" : "cr-project-tabs__tab--train-active") : "",
          ]
            .filter(Boolean)
            .join(" ");
          return (
          <div
            key={p.id}
            className={tabClass}
            role="none"
          >
            <div className="cr-project-tabs__tab-row">
            {editingId === p.id ? (
              <div
                className="cr-project-tabs__tab-main cr-project-tabs__tab-main--rename"
                role="tab"
                aria-selected={p.id === activeProjectId}
              >
                <input
                  ref={inputRef}
                  className="cr-project-tabs__rename-input"
                  aria-label="Project name"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      commitRename();
                    } else if (e.key === "Escape") {
                      e.preventDefault();
                      cancelRename();
                    }
                  }}
                  onMouseDown={(e) => e.stopPropagation()}
                />
              </div>
            ) : (
              <button
                type="button"
                role="tab"
                aria-selected={p.id === activeProjectId}
                className="cr-project-tabs__tab-main"
                onClick={() => onSelect(p.id)}
              >
                <span
                  className="cr-project-tabs__title"
                  title="Double-click to rename"
                  onDoubleClick={(e) => beginRename(p, e)}
                >
                  {p.title}
                </span>
                {p.canvas.dirty ? (
                  <span className="cr-project-tabs__dot" title="Unsaved changes" aria-hidden />
                ) : null}
              </button>
            )}
            {projects.length > 1 ? (
              <button
                type="button"
                className="cr-project-tabs__close"
                aria-label={`Close ${p.title}`}
                onClick={(e) => onClose(p.id, e)}
              >
                ×
              </button>
            ) : null}
            </div>
            {trainActive ? (
              <div
                className="cr-project-tabs__train-bar nodrag nopan"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(train?.progressPct ?? 0)}
                aria-label={train?.paused ? "Training paused" : "Training in progress"}
                title={train?.paused ? `Training paused (${train.progressPct}%)` : `Training ${train?.progressPct ?? 0}%`}
              >
                <div
                  className="cr-project-tabs__train-bar-fill"
                  style={{ width: `${Math.min(100, train?.progressPct ?? 0)}%` }}
                />
              </div>
            ) : null}
          </div>
          );
        })}
      </div>
      <button type="button" className="cr-project-tabs__add" aria-label="New project" onClick={onAdd}>
        +
      </button>
    </div>
  );
}

type AddNodeFromLibrary = (
  nodeType: string,
  screenPos?: { x: number; y: number },
  options?: AddNodeOptions,
) => void;

function FlowCanvasExtras({
  addNodeImplRef,
  setNodes,
}: {
  addNodeImplRef: MutableRefObject<AddNodeFromLibrary>;
  setNodes: Dispatch<SetStateAction<Node[]>>;
}) {
  const { screenToFlowPosition, getNodes, getEdges, setEdges } = useReactFlow();

  useEffect(() => {
    const defaultAddScreenPos = (): { x: number; y: number } => {
      const vw = typeof window !== "undefined" ? window.innerWidth : 1200;
      return { x: vw * 0.55, y: 220 };
    };
    addNodeImplRef.current = (nodeType, screenPos, options) => {
      const pos = screenToFlowPosition(screenPos ?? defaultAddScreenPos());
      // observable_user 显式排除:其特殊分支把 DnD options 线程进
      // defaults,generic 路径不识别 options——不排除会被本分支先命中截胡。
      // linear_dataset 同款排除:userLinearDatasetId options 线程分支必须先行。
      // symbolic_func 同款排除:userSymbolicFuncDatasetId options 线程分支先行。
      // combined_model 同款排除:template/options 线程分支先行。
      if (GENERATED_NODE_SPECS[nodeType] && nodeType !== "observable_user" && nodeType !== "linear_dataset" && nodeType !== "symbolic_func_dataset" && nodeType !== "combined_model") {
        // NodeDef-channel generic add-node path : defaults from the generated spec.
        // 经 nodeRegistryDefaults 走 cloneDefaults 深拷贝——
        // 浅拷贝会让数组/嵌套对象 defaults 跨 spawn 共享引用(L2 的 tensor 工具型前置)。
        setNodes((nds) => [
          ...nds,
          appendResearchNode(nds, nodeType, pos, nodeRegistryDefaults(nodeType) ?? {}),
        ]);
        return;
      }
      if (nodeType === "linear_dataset") {
        const uid = options?.userLinearDatasetId?.trim();
        if (uid) {
          void (async () => {
            try {
              const r = await fetch(`/api/user-linear-datasets/${encodeURIComponent(uid)}`);
              if (!r.ok) throw new Error(String(r.status));
              const body = (await r.json()) as { item?: { node_data?: Record<string, unknown> } };
              const raw = body.item?.node_data;
              const data = raw ? { ...defaultLinearDatasetData(), ...raw } : defaultLinearDatasetData();
              setNodes((nds) => [
                ...nds,
                appendResearchNode(nds, "linear_dataset", pos, data as Record<string, unknown>),
              ]);
            } catch {
              setNodes((nds) => [
                ...nds,
                appendResearchNode(
                  nds,
                  "linear_dataset",
                  pos,
                  defaultLinearDatasetData() as Record<string, unknown>,
                ),
              ]);
            }
          })();
          return;
        }
        setNodes((nds) => [
          ...nds,
          appendResearchNode(
            nds,
            "linear_dataset",
            pos,
            defaultLinearDatasetData() as Record<string, unknown>,
          ),
        ]);
        return;
      }
      if (nodeType === "symbolic_func_dataset") {
        const uid = options?.userSymbolicFuncDatasetId?.trim();
        if (uid) {
          void (async () => {
            try {
              const r = await fetch(`/api/user-symbolic-func-datasets/${encodeURIComponent(uid)}`);
              if (!r.ok) throw new Error(String(r.status));
              const body = (await r.json()) as { item?: { node_data?: Record<string, unknown> } };
              const raw = body.item?.node_data;
              const data = raw ? { ...defaultSymbolicFuncDatasetData(), ...raw } : defaultSymbolicFuncDatasetData();
              setNodes((nds) => [
                ...nds,
                appendResearchNode(nds, "symbolic_func_dataset", pos, data as Record<string, unknown>),
              ]);
            } catch {
              setNodes((nds) => [
                ...nds,
                appendResearchNode(
                  nds,
                  "symbolic_func_dataset",
                  pos,
                  defaultSymbolicFuncDatasetData() as Record<string, unknown>,
                ),
              ]);
            }
          })();
          return;
        }
        setNodes((nds) => [
          ...nds,
          appendResearchNode(
            nds,
            "symbolic_func_dataset",
            pos,
            defaultSymbolicFuncDatasetData() as Record<string, unknown>,
          ),
        ]);
        return;
      }
      if (nodeType === "attention_only_model") {
        setNodes((nds) => [
          ...nds,
          appendResearchNode(
            nds,
            "attention_only_model",
            pos,
            defaultAttentionOnlyModelData() as Record<string, unknown>,
          ),
        ]);
        return;
      }
      // Observable-user drops carry four option values that generic defaults
      // cannot reconstruct, so this node keeps a dedicated add path.
      if (nodeType === "observable_user") {
        const o = options ?? {};
        setNodes((nds) => [
          ...nds,
          appendResearchNode(
            nds,
            "observable_user",
            pos,
            defaultObservableUserData({
              userObservableId: o.userObservableId ?? "",
              label: o.label ?? "User observable",
              tensorVizNodeId: o.tensorVizNodeId ?? "",
              tensorSelectorNodeId: o.tensorSelectorNodeId ?? "",
            }) as Record<string, unknown>,
          ),
        ]);
        return;
      }
      if (nodeType === "combined_model") {
        const o = options ?? {};
        const tid = o.combinedModelTemplateId?.trim();
        const displayName = o.combinedModelDisplayName?.trim() || "Combined model";
        const sc = o.combinedModelSourceNodeCount;
        const templateDoc = o.combinedModelTemplateDocument;
        const sourceNodeCount =
          typeof sc === "number" && Number.isFinite(sc) && sc >= 0 ? Math.floor(sc) : 0;

        const fallbackShell = () => {
          setNodes((nds) => [
            ...nds,
            appendResearchNode(
              nds,
              "combined_model",
              pos,
              defaultCombinedModelData({
                displayName,
                templateId: tid || undefined,
                sourceNodeCount,
              }) as Record<string, unknown>,
            ),
          ]);
        };

        const tryFromDocument = (doc: GraphDocument | null | undefined): boolean => {
          if (!doc?.nodes?.length) return false;
          const nds = getNodes();
          const eds = getEdges();
          const built = instantiateCombinedModelFromLibraryTemplate(nds, doc, pos, {
            displayName,
            templateId: tid || undefined,
          });
          if (!built) return false;
          setNodes((n) => sortNodesParentBeforeChildren([...n.map((x) => ({ ...x, selected: false })), ...built.nodes]));
          setEdges([...eds, ...built.edges]);
          return true;
        };

        if (tryFromDocument(templateDoc)) return;

        if (tid) {
          void (async () => {
            try {
              const [w, t] = await Promise.all([
                fetchSavedGraphLibrary("workflow"),
                fetchSavedGraphLibrary("template"),
              ]);
              const hit = w.find((e) => e.id === tid) ?? t.find((e) => e.id === tid);
              if (hit?.document && tryFromDocument(hit.document)) return;
            } catch {
              /* fall through */
            }
            fallbackShell();
          })();
          return;
        }

        fallbackShell();
        return;
      }
    };
  }, [
    addNodeImplRef,
    getEdges,
    getNodes,
    screenToFlowPosition,
    setEdges,
    setNodes,
  ]);

  return null;
}

const GRAPH_ASSIST_MODE_OPTIONS = [
  { id: "manual" as const, label: "manual" },
  { id: "self_driving" as const, label: "self-driving" },
  { id: "physics_of_ai" as const, label: "physics-of-ai agent" },
];

type CoordinateDescentEvent =
  | { type: "tuning_started"; session_id: string }
  | { type: "baseline_evaluated"; session_id?: string; error?: string | null }
  | { type: "round_started"; round_index: number; current_params?: Record<string, unknown> }
  | {
      type: "candidate_evaluated";
      params?: Record<string, unknown>;
      score?: number;
      error?: string | null;
    }
  | { type: "axis_best_selected"; axis_key?: string; score?: number }
  | {
      type: "tuning_complete";
      best_score?: number;
      best_params?: Record<string, unknown>;
      best_curve?: { step_ticks?: number[]; loss_history?: number[]; test_loss_history?: number[] };
      baseline_curve?: { step_ticks?: number[]; loss_history?: number[] };
      target_curve?: { step_ticks?: number[]; loss_history?: number[] };
      ranked_curves?: Array<{
        rank: number;
        score: number;
        final_abs_error?: number;
        smoothness_penalty?: number;
        params?: Record<string, unknown>;
        curve?: { step_ticks?: number[]; loss_history?: number[] };
      }>;
    }
  | { type: "tuning_aborted" };

function mapTuningCompleteToComparisonResult(
  ev: Extract<CoordinateDescentEvent, { type: "tuning_complete" }>,
): AutoTuneComparisonResult {
  const numArr = (a: unknown): number[] =>
    Array.isArray(a) ? a.map((x) => Number(x)).filter((x) => Number.isFinite(x)) : [];
  const bc = ev.baseline_curve;
  const tc = ev.target_curve;
  return {
    baselineStepTicks: numArr(bc?.step_ticks),
    baselineLossHistory: numArr(bc?.loss_history),
    targetStepTicks: numArr(tc?.step_ticks),
    targetLossHistory: numArr(tc?.loss_history),
    ranked: (ev.ranked_curves ?? []).map((r) => ({
      rank: r.rank,
      score: typeof r.score === "number" ? r.score : Number.NaN,
      finalAbsError: typeof r.final_abs_error === "number" ? r.final_abs_error : undefined,
      smoothnessPenalty: typeof r.smoothness_penalty === "number" ? r.smoothness_penalty : undefined,
      params: (r.params ?? {}) as Record<string, unknown>,
      stepTicks: numArr(r.curve?.step_ticks),
      lossHistory: numArr(r.curve?.loss_history),
    })),
    bestScore: typeof ev.best_score === "number" ? ev.best_score : Number.NaN,
  };
}

async function readNdjsonCoordinateDescentStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onEvent: (ev: CoordinateDescentEvent) => void,
): Promise<void> {
  const decoder = new TextDecoder();
  let buffer = "";
  const dispatch = (line: string) => {
    const t = line.trim();
    if (!t) return;
    onEvent(JSON.parse(t) as CoordinateDescentEvent);
  };
  while (true) {
    const { done, value } = await reader.read();
    if (value) buffer += decoder.decode(value, { stream: true });
    for (;;) {
      const nl = buffer.indexOf("\n");
      if (nl < 0) break;
      const line = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      dispatch(line);
    }
    if (done) break;
  }
  dispatch(buffer);
}

function FlowApp({
  addNodeImplRef,
  codeMode,
  onCodeModeChange,
  flowSurfaceKey,
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  owningProjectId,
  setNodesForCanvas,
  setEdgesForCanvas,
  onClearCanvas,
  savedViewport,
  viewportApplyNonce,
  onSaveToServer,
  onLoadFromServer,
  onSaveServerSucceeded,
  onSaveGraphToFile,
  graphFileStemBase,
  onPersistLibraryGraph,
  onGraphFileLoaded,
  onGraphFileError,
  onOpenGraphCompare,
  graphFileHandle,
  librarySource,
  librarySaveDisplayName,
  onSaveLibrarySourceEntry,
  onSaveGraphToSourceFileSucceeded,
  loading,
  error,
  notice,
  readGraphForCanvas,
  onCanvasSelectionChange,
  onRequestCloseRail,
  onRequestCloseNodeInformation,
  onRequestOpenNodesRail,
  nodesRailOpen,
}: {
  codeMode: boolean;
  onCodeModeChange: (next: boolean) => void;
  /** Remount React Flow when switching project/canvas so internal store/viewport cannot desync. */
  flowSurfaceKey: string;
  addNodeImplRef: MutableRefObject<AddNodeFromLibrary>;
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  owningProjectId: string;
  setNodesForCanvas: (projectId: string, updater: SetStateAction<Node[]>) => void;
  setEdgesForCanvas: (projectId: string, updater: SetStateAction<Edge[]>) => void;
  onClearCanvas: () => void;
  savedViewport: Viewport | null | undefined;
  viewportApplyNonce: number;
  onSaveToServer: (doc: GraphDocument) => Promise<void>;
  onLoadFromServer: () => Promise<void>;
  onSaveServerSucceeded: () => void;
  onSaveGraphToFile: (doc: GraphDocument, fileStem: string) => Promise<void>;
  graphFileStemBase: string;
  onPersistLibraryGraph: (
    kind: SavedGraphKind,
    doc: GraphDocument,
    tier: GraphFileExportTier,
  ) => void;
  onGraphFileLoaded: (doc: GraphDocument, fileHandle?: FileSystemFileHandle | null) => void;
  onGraphFileError: (message: string) => void;
  onOpenGraphCompare: (doc: GraphDocument) => void;
  graphFileHandle: FileSystemFileHandle | null;
  librarySource: GraphCanvas["librarySource"] | undefined;
  librarySaveDisplayName: string;
  onSaveLibrarySourceEntry: (kind: "workflow" | "template", entry: SavedGraphEntry) => Promise<void>;
  onSaveGraphToSourceFileSucceeded: () => void;
  loading: boolean;
  error: string | null;
  notice: string | null;
  /** Persisted graph for a project — stays valid after React Flow unmounts (e.g. project tab switch). */
  readGraphForCanvas?: (projectId: string) => { nodes: Node[]; edges: Edge[] } | null;
  onCanvasSelectionChange?: (nodeId: string | null, nodeType?: string | null) => void;
  onRequestCloseRail: () => void;
  onRequestCloseNodeInformation: () => void;
  onRequestOpenNodesRail: () => void;
  nodesRailOpen: boolean;
}) {
  const { screenToFlowPosition, getNodes, getEdges, getViewport, fitView, setViewport } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();
  const canvasWrapRef = useRef<HTMLDivElement | null>(null);
  const activeConnectionRef = useRef<{ nodeId: string; handleId: string | null; handleType: "source" | "target" } | null>(null);
  const connectionCommittedRef = useRef(false);
  const [connectionSnapTarget, setConnectionSnapTarget] = useState<ConnectionSnapTarget | null>(null);
  const connectionSnapTargetRef = useRef<ConnectionSnapTarget | null>(null);
  const connectionPointerCleanupRef = useRef<(() => void) | null>(null);

  const setNodes = useCallback(
    (updater: SetStateAction<Node[]>) => {
      setNodesForCanvas(owningProjectId, updater);
    },
    [owningProjectId, setNodesForCanvas],
  );
  const setEdges = useCallback(
    (updater: SetStateAction<Edge[]>) => {
      setEdgesForCanvas(owningProjectId, updater);
    },
    [owningProjectId, setEdgesForCanvas],
  );
  const getStableNodes = useCallback((): Node[] => {
    const g = readGraphForCanvas?.(owningProjectId);
    if (g?.nodes?.length) return g.nodes;
    const rfNodes = getNodes();
    return rfNodes.length ? rfNodes : (g?.nodes ?? []);
  }, [owningProjectId, readGraphForCanvas, getNodes]);
  const getStableEdges = useCallback((): Edge[] => {
    const g = readGraphForCanvas?.(owningProjectId);
    if (g?.edges?.length) return g.edges;
    const rfEdges = getEdges();
    return rfEdges.length ? rfEdges : (g?.edges ?? []);
  }, [owningProjectId, readGraphForCanvas, getEdges]);

  const researchEdgeTypes = useMemo(
    () => ({
      [COMBINED_SUBGRAPH_IO_EDGE_TYPE]: CombinedSubgraphIoEdge,
      research_default: ResearchDefaultEdge,
    }),
    [],
  );
  type GraphAssistMode = "manual" | "self_driving" | "physics_of_ai";
  const [graphAssistMode, setGraphAssistMode] = useState<GraphAssistMode>("manual");
  const [graphAssistSeedStr, setGraphAssistSeedStr] = useState("");
  const [graphAssistDelaySecStr, setGraphAssistDelaySecStr] = useState("0");
  const [graphAssistLog, setGraphAssistLog] = useState<string[]>([]);
  const [graphAssistBusy, setGraphAssistBusy] = useState(false);
  const [graphAssistHideFailureCross, setGraphAssistHideFailureCross] = useState(false);
  const graphAssistHideFailureCrossRef = useRef(false);
  /** Bumps only when starting self-driving so the assist effect always reads the latest seed field. */
  const [graphAssistRunNonce, setGraphAssistRunNonce] = useState(0);
  const graphAssistAbortRef = useRef<AbortController | null>(null);
  const graphAssistKickRef = useRef(0);
  const [targetCurveModalOpen, setTargetCurveModalOpen] = useState(false);
  const [autoTuneModalOpen, setAutoTuneModalOpen] = useState(false);
  const [autoTuneBusy, setAutoTuneBusy] = useState(false);
  const [autoTuneStatus, setAutoTuneStatus] = useState<string | null>(null);
  const autoTuneSessionRef = useRef<string | null>(null);
  const [autoTuneResultsOpen, setAutoTuneResultsOpen] = useState(false);
  const [autoTuneResults, setAutoTuneResults] = useState<AutoTuneComparisonResult | null>(null);
  const [curveStarerOpen, setCurveStarerOpen] = useState(false);
  const [curveStarerBusy, setCurveStarerBusy] = useState(false);
  const [curveStarerStatus, setCurveStarerStatus] = useState<string | null>(null);
  const [curveStarerEntries, setCurveStarerEntries] = useState<CurveStarerAnalyzedEntry[]>([]);
  const [curveStarerRankBy, setCurveStarerRankBy] = useState<CurveStarerRankBy>("default");
  const [curveStarerProgress, setCurveStarerProgress] = useState<{ total: number; completed: number } | null>(
    null,
  );
  const [curveStarerTargetConfig, setCurveStarerTargetConfig] = useState<CurveStarerTargetConfig | null>(
    null,
  );
  const [curveStarerTargetOptions, setCurveStarerTargetOptions] = useState<{ id: string; label: string }[]>(
    [],
  );

  const refreshCurveStarerTargets = useCallback((graphNodes: Node[], graphEdges: Edge[]) => {
    const curves = collectObservableTrainingCurves(graphNodes, graphEdges);
    setCurveStarerTargetOptions(curves.map((c) => ({ id: c.entryId, label: c.label })));
    if (curves.length === 0) return;
    setCurveStarerTargetConfig((prev) => {
      if (prev && curves.some((c) => c.entryId === prev.entryId)) return prev;
      const suggested = suggestDefaultTarget(curves);
      if (suggested) return suggested;
      const first = curves[0]!;
      return { entryId: first.entryId, objective: "higher", threshold: 0.95 };
    });
  }, []);

  useEffect(() => {
    if (!curveStarerOpen) return;
    refreshCurveStarerTargets(nodes, edges);
  }, [curveStarerOpen, edges, nodes, refreshCurveStarerTargets]);

  useEffect(() => {
    graphAssistHideFailureCrossRef.current = graphAssistHideFailureCross;
    setNodes((nds) =>
      nds.map((n) =>
        n.type === "graph_assist_failure_overlay" ? { ...n, hidden: graphAssistHideFailureCross } : n,
      ),
    );
  }, [graphAssistHideFailureCross, setNodes]);

  const bumpGraphAssistViewport = useCallback(
    async (pauseMs: number) => {
      await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
      fitView({
        padding: 0.16,
        duration: pauseMs > 0 ? Math.min(260, pauseMs) : 0,
        minZoom: 0.04,
        maxZoom: 1.38,
      });
    },
    [fitView],
  );

  const analyzeCurveStarer = useCallback(async (graphNodes: Node[], graphEdges: Edge[], rankBy: CurveStarerRankBy) => {
    setCurveStarerStatus(null);
    setCurveStarerEntries([]);
    setCurveStarerProgress(null);
    setCurveStarerRankBy(rankBy);
    const curves = collectObservableTrainingCurves(graphNodes, graphEdges, {
      minPoints: LPD_MIN_CURVE_POINTS,
    });
    refreshCurveStarerTargets(graphNodes, graphEdges);
    if (curves.length === 0) {
      setCurveStarerStatus(
        "No observable or training viz curves on this canvas. Train with wired observable viz or training viz nodes (≥ 5 steps each for LPD).",
      );
      setCurveStarerBusy(false);
      return;
    }

    const cachedEntries = getCurveStarerCachedEntries(curves);
    if (cachedEntries) {
      setCurveStarerEntries(cachedEntries);
      setCurveStarerStatus("Loaded previous LPD results (experiment unchanged).");
      const suggested = suggestDefaultTarget(cachedEntries);
      if (suggested) setCurveStarerTargetConfig(suggested);
      setCurveStarerBusy(false);
      return;
    }

    setCurveStarerBusy(true);
    setCurveStarerProgress({ total: curves.length, completed: 0 });
    const analyzed: CurveStarerAnalyzedEntry[] = new Array(curves.length);
    try {
      await fetchLpdPredictBatch(
        curves.map((c) => c.points),
        {
          onResult: (index, item) => {
            const curve = curves[index]!;
            analyzed[index] =
              item.ok
                ? { ...curve, lpd: item.result }
                : { ...curve, lpd: null, lpdError: item.error };
            setCurveStarerEntries(analyzed.filter((e): e is CurveStarerAnalyzedEntry => e != null));
            setCurveStarerProgress((prev) =>
              prev ? { ...prev, completed: prev.completed + 1 } : null,
            );
          },
        },
      );
      setCurveStarerCache(curves, analyzed);
      const suggested = suggestDefaultTarget(analyzed);
      if (suggested) setCurveStarerTargetConfig(suggested);
    } catch (e) {
      setCurveStarerStatus(e instanceof Error ? e.message : String(e));
    } finally {
      setCurveStarerProgress(null);
      setCurveStarerBusy(false);
    }
  }, [refreshCurveStarerTargets]);

  const openCurveStarerModal = useCallback(() => {
    setCurveStarerOpen(true);
    refreshCurveStarerTargets(nodes, edges);
  }, [edges, nodes, refreshCurveStarerTargets]);

  useEffect(() => {
    const handleOpenCurveStarer = () => openCurveStarerModal();
    window.addEventListener(OPEN_CURVE_STARER_EVENT, handleOpenCurveStarer);
    return () => window.removeEventListener(OPEN_CURVE_STARER_EVENT, handleOpenCurveStarer);
  }, [openCurveStarerModal]);

  const startCurveStarerAnalysis = useCallback(async () => {
    await analyzeCurveStarer(nodes, edges, curveStarerRankBy);
  }, [analyzeCurveStarer, curveStarerRankBy, edges, nodes]);

  /**
   * Self-driving train must not list ``getNodes`` / ``getEdges`` / ``bumpGraphAssistViewport`` in the
   * graph-assist effect deps: React Flow can refresh those identities whenever the flow store updates,
   * which would abort the in-flight ``/api/train`` request and show the red failure cross.
   */
  const graphAssistReactFlowRef = useRef({
    getNodes,
    getEdges,
    bumpGraphAssistViewport,
  });
  graphAssistReactFlowRef.current = {
    getNodes,
    getEdges,
    bumpGraphAssistViewport,
  };

  /** Latest owning project for graph-assist; avoids an effect dep that re-runs on a project tab switch. */
  const graphAssistScopeRef = useRef({ projectId: owningProjectId });
  graphAssistScopeRef.current = { projectId: owningProjectId };

  // Graph-assist effect deps omit: (1) getNodes/getEdges/bumpGraphAssistViewport/setNodes/setEdges
  // → graphAssistReactFlowRef so React Flow store updates do not abort in-flight /api/train.
  // (2) owningProjectId → graphAssistScopeRef; otherwise switching project tabs
  // re-runs the effect while mode can still be self_driving.
  useEffect(() => {
    if (graphAssistMode === "manual") {
      graphAssistAbortRef.current?.abort();
      graphAssistAbortRef.current = null;
      return;
    }
    const ac = new AbortController();
    graphAssistAbortRef.current = ac;
    let cancelled = false;

    const sleep = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));

    void (async () => {
      setGraphAssistBusy(true);
      setGraphAssistLog([graphAssistMode === "physics_of_ai" ? "physics-of-ai agent…" : "generating…"]);

      const seedTrim = graphAssistSeedStr.trim();
      const seedsParsed = parseGraphAssistSeeds(graphAssistSeedStr);
      let seeds: number[] = [];
      if (seedsParsed.length === 0) {
        const resolvedSeed =
          seedTrim === ""
            ? ((Math.imul(graphAssistKickRef.current, 1597334677) ^ Date.now()) >>> 0)
            : parseGraphAssistSeed(graphAssistSeedStr, 0);
        if (seedTrim === "") {
          setGraphAssistSeedStr(String(resolvedSeed));
        }
        seeds = [resolvedSeed];
      } else {
        seeds = graphAssistMode === "physics_of_ai" ? [seedsParsed[0]!] : seedsParsed;
      }

      const delaySecRaw = Number(graphAssistDelaySecStr.replace(/_/g, "").trim());
      const delaySec = Number.isFinite(delaySecRaw) ? Math.max(0, delaySecRaw) : 0;
      const delayMs = Math.round(delaySec * 1000);
      const isPhysicsOfAi = graphAssistMode === "physics_of_ai";
      // Product decision: self-driving currently builds candidate graphs only.
      // Training is user-triggered (manual) to avoid automatic remote/GPU failures.
      const autoTrainInSelfDriving = isPhysicsOfAi;

      const trainerCleanupRows: Array<{ trainerId: string }> = [];
      try {
        const ns0 = graphAssistReactFlowRef.current.getNodes();
        const baseAnchor = computeSelfDrivingAnchor(ns0);
        const dim = Math.max(1, Math.ceil(Math.sqrt(seeds.length)));
        const gridSlots = graphAssistGridCellOrder(dim).slice(0, seeds.length);

        const pauseBetweenActions = (ms: number) =>
          ms > 0 ? sleep(ms) : new Promise<void>((r) => requestAnimationFrame(() => r()));

        type SelfDriveCanvasCtx = {
          setWorkNodes: (u: SetStateAction<Node[]>) => void;
          setWorkEdges: (u: SetStateAction<Edge[]>) => void;
          getWorkGraph: () => { nodes: Node[]; edges: Edge[] };
          bumpViewport: (ms: number) => Promise<void>;
        };

        const runPlannedSubgraphSteps = async (
          pauseMs: number,
          plan: PlannedRandomTrainer,
          ctx: SelfDriveCanvasCtx,
        ) => {
          const { setWorkNodes, setWorkEdges, getWorkGraph, bumpViewport } = ctx;
          const tid = plan.trainerId;
          const edgeSpawnsTrainerVizImmediately = (ed: Edge) =>
            ed.target === tid &&
            ((ed.targetHandle === "observables" &&
              (ed.sourceHandle === "observables" || ed.sourceHandle === "observable")) ||
              (ed.targetHandle === "loss" && ed.sourceHandle === "loss"));
          for (const step of plan.steps) {
            if (cancelled || ac.signal.aborted) return;
            setWorkNodes((nds) => [
              ...nds,
              appendResearchNode(nds, step.node.type, step.node.position, step.node.data, step.node.id),
            ]);
            await pauseBetweenActions(pauseMs);
            await bumpViewport(pauseMs);
            for (const ed of step.edges) {
              if (cancelled || ac.signal.aborted) return;
              if (edgeSpawnsTrainerVizImmediately(ed)) {
                flushSync(() => {
                  setWorkEdges((eds) => [...eds, ed]);
                });
                const snap = getWorkGraph();
                const fin = ensureTrainerAutoVizes(
                  snap.nodes,
                  snap.edges,
                  tid,
                  plan.observableVizId,
                  plan.trainingVizId,
                );
                if (fin.nodes.length !== snap.nodes.length || fin.edges.length !== snap.edges.length) {
                  flushSync(() => {
                    setWorkNodes(fin.nodes);
                    setWorkEdges(fin.edges);
                  });
                }
              } else {
                setWorkEdges((eds) => [...eds, ed]);
              }
              await pauseBetweenActions(pauseMs);
              await bumpViewport(pauseMs);
            }
          }
        };

        const runTrainForPlan = async (
          plan: PlannedRandomTrainer,
          ctx: SelfDriveCanvasCtx,
        ): Promise<{ ok: boolean; phase?: string; reason?: string }> => {
          const { setWorkNodes, getWorkGraph } = ctx;
          const tid = plan.trainerId;
          const trainingVizId = plan.trainingVizId;
          try {
            const g0 = getWorkGraph();
            const ns = g0.nodes.filter((n) => String(n.type) !== "graph_assist_failure_overlay");
            const es = g0.edges;
            let combos: ReturnType<typeof planTrainSeriesAssignments>;
            try {
              combos = planTrainSeriesAssignments(ns, es, tid);
            } catch (err) {
              console.warn("Graph assist plan series:", err);
              setGraphAssistLog([`Error: ${err instanceof Error ? err.message : String(err)}`]);
              return {
                ok: false,
                phase: "series_plan",
                reason: err instanceof Error ? err.message : String(err),
              };
            }
            if (delayMs > 0) {
              await sleep(delayMs);
              if (cancelled || ac.signal.aborted) return { ok: false, phase: "cancelled", reason: "cancelled" };
            }
            const base = ns.map(serializeNodeForTrain);
            const combo0 = combos[0] ?? [];
            const edgePayload = es.map((e) => ({
              id: e.id,
              source: e.source,
              target: e.target,
              sourceHandle: e.sourceHandle ?? null,
              targetHandle: e.targetHandle ?? null,
            }));
            const nodesPayload = applyAssignmentsAndResolveTrainingLength(base, combo0, edgePayload);
            const totalRuns = combos.length;
            const seriesDual = totalRuns > 1;
            const sweptAxisIds = getSweptAxisIdSet(ns, es, tid);
            const captionLines: string[] | null = seriesDual
              ? formatTrainSeriesSweptLines(combo0, base, sweptAxisIds)
              : null;
            const seriesRunIndex = 0;

            const res = await fetch("/api/train", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              signal: ac.signal,
              body: JSON.stringify({
                trainer_node_id: tid,
                nodes: nodesPayload,
                edges: edgePayload,
              }),
            });

            if (cancelled || ac.signal.aborted) return { ok: false, phase: "cancelled", reason: "cancelled" };
            if (!res.ok) {
              let msg = res.statusText;
              try {
                const j = (await res.json()) as { detail?: unknown };
                if (j.detail != null) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
              } catch {
                /* ignore */
              }
              setGraphAssistLog([`Train failed: ${msg}`]);
              console.warn("Graph assist train failed:", msg);
              return { ok: false, phase: "train_request", reason: msg };
            }

            const reader = res.body?.getReader();
            if (!reader) {
              setGraphAssistLog(["Train failed: empty body"]);
              return { ok: false, phase: "train_stream", reason: "empty response body" };
            }

            patchTrainerHostUi(
              tid,
              {
                active: true,
                progressPct: 0,
                seriesBarPct: seriesDual
                  ? Math.min(100, Math.round((seriesRunIndex / totalRuns) * 100))
                  : 0,
                seriesDual,
                captionLines,
              },
              setWorkNodes,
            );

            const handleStreamProgress = (raw: TrainStreamProgress) => {
              if (raw.type !== "progress") return;
              const total = Math.max(1, raw.total);
              const step = Math.min(Math.max(0, raw.step), total);
              const within = step / total;
              const pctWithin = Math.min(100, Math.round(within * 100));
              let seriesBarPct = pctWithin;
              if (seriesDual) {
                seriesBarPct = Math.min(
                  100,
                  Math.round(((seriesRunIndex + within) / totalRuns) * 100),
                );
              }
              patchTrainerHostUi(
                tid,
                {
                  active: true,
                  progressPct: pctWithin,
                  seriesBarPct,
                  seriesDual,
                  captionLines,
                },
                setWorkNodes,
              );
            };

            const streamResult = await readNdjsonTrainStream(reader, handleStreamProgress);
            // Mirror TrainerNode: surface typed server error events instead of letting
            // them degrade into "ended without complete event".
            if (streamResult.error) throw new Error(streamResult.error);
            const { complete, aborted } = streamResult;
            if (cancelled || ac.signal.aborted || aborted) {
              setGraphAssistLog(["generation cancelled"]);
              return { ok: false, phase: "cancelled", reason: "cancelled" };
            }
            if (!complete) {
              setGraphAssistLog(["Train ended without complete event"]);
              return { ok: false, phase: "train_stream", reason: "ended without complete event" };
            }

            const trainerTy = ns.find((n) => n.id === tid)?.type;
            await flushCheckpointApplyTrainerVizAndHydrateTv0d({
              isCrl: trainerTy === "crl_trainer",
              setNodes: setWorkNodes,
              getNodes: () => getWorkGraph().nodes,
              getEdges: () => getWorkGraph().edges,
              trainerNodeId: tid,
              wires: edgePayload,
              payload: complete,
              extraRefreshDelaysMs: [250, 900],
            });

            const lh = complete.loss_history ?? [];
            const vizOk =
              lh.length >= 2 && (complete.visualization_node_ids ?? []).includes(trainingVizId);
            if (!vizOk) {
              // Self-driving previously treated partial viz payloads as hard failures and painted
              // failure overlays, even when training completed. Keep a warning only.
              console.warn("Graph assist: train finished but viz/metrics look incomplete.");
              setGraphAssistLog([
                "Train completed. Some visualization metadata is incomplete for this candidate.",
              ]);
            }
            return { ok: true };
          } catch (err) {
            if (cancelled || ac.signal.aborted) return { ok: false, phase: "cancelled", reason: "cancelled" };
            const msg = err instanceof Error ? err.message : String(err);
            setGraphAssistLog([`Train failed: ${msg}`]);
            console.warn("Graph assist train error:", err);
            return { ok: false, phase: "train_runtime", reason: msg };
          }
        };

        /** Tight box around the self-driving subgraph (+ auto-spawned vizes), not the huge matrix cell. */
        const computeGraphAssistFailureOverlayBox = (
          nodes: Node[],
          plan: PlannedRandomTrainer,
          cellTopLeft: { x: number; y: number },
        ) => {
          const tid = plan.trainerId;
          const ids = new Set<string>(plan.steps.map((s) => s.node.id));
          ids.add(plan.observableVizId);
          ids.add(plan.trainingVizId);
          for (const n of nodes) {
            if (typeof n.id === "string" && n.id.startsWith(`${tid}__`)) ids.add(n.id);
          }
          const placed = nodes.filter(
            (n) => ids.has(n.id) && n.type !== "graph_assist_failure_overlay",
          );
          const pad = 20;
          let minX = Number.POSITIVE_INFINITY;
          let minY = Number.POSITIVE_INFINITY;
          let maxX = Number.NEGATIVE_INFINITY;
          let maxY = Number.NEGATIVE_INFINITY;
          for (const n of placed) {
            const nAny = n as Node & { measured?: { width?: number; height?: number } };
            const w =
              typeof n.width === "number" && n.width > 0
                ? n.width
                : typeof nAny.measured?.width === "number" && nAny.measured.width > 0
                  ? nAny.measured.width
                  : 420;
            const h =
              typeof n.height === "number" && n.height > 0
                ? n.height
                : typeof nAny.measured?.height === "number" && nAny.measured.height > 0
                  ? nAny.measured.height
                  : 260;
            minX = Math.min(minX, n.position.x);
            minY = Math.min(minY, n.position.y);
            maxX = Math.max(maxX, n.position.x + w);
            maxY = Math.max(maxY, n.position.y + h);
          }
          if (!Number.isFinite(minX) || placed.length === 0) {
            return { x: cellTopLeft.x, y: cellTopLeft.y, width: 860, height: 560 };
          }
          return {
            x: minX - pad,
            y: minY - pad,
            width: Math.max(120, maxX - minX + 2 * pad),
            height: Math.max(100, maxY - minY + 2 * pad),
          };
        };

        const appendGraphAssistFailureOverlay = (
          getGraph: () => { nodes: Node[]; edges: Edge[] },
          plan: PlannedRandomTrainer,
          cellTopLeft: { x: number; y: number },
          setW: (u: SetStateAction<Node[]>) => void,
          detail?: { phase?: string; reason?: string },
        ) => {
          const overlayId = `graph-assist-fail-${plan.trainerId}`;
          const box = computeGraphAssistFailureOverlayBox(getGraph().nodes, plan, cellTopLeft);
          flushSync(() => {
            setW((nds) => {
              if (nds.some((n) => n.id === overlayId)) return nds;
              const next: Node = {
                id: overlayId,
                type: "graph_assist_failure_overlay",
                position: {
                  x: box.x,
                  y: box.y,
                },
                width: box.width,
                height: box.height,
                zIndex: 10050,
                style: {
                  width: box.width,
                  height: box.height,
                  pointerEvents: "none",
                },
                selectable: false,
                draggable: false,
                focusable: false,
                deletable: true,
                hidden: graphAssistHideFailureCrossRef.current,
                data: {
                  phase: detail?.phase ?? "",
                  reason: detail?.reason ?? "",
                },
              };
              return [...nds, next];
            });
          });
        };

        const centerPlanWithinMatrixCell = async (
          plan: PlannedRandomTrainer,
          cellTopLeft: { x: number; y: number },
          getGraph: () => { nodes: Node[]; edges: Edge[] },
          setW: (u: SetStateAction<Node[]>) => void,
        ) => {
          await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
          const planNodeIds = new Set<string>(plan.steps.map((s) => s.node.id));
          planNodeIds.add(plan.observableVizId);
          planNodeIds.add(plan.trainingVizId);
          const placed = getGraph().nodes.filter((n) => planNodeIds.has(n.id));
          if (placed.length === 0) return;

          let minX = Number.POSITIVE_INFINITY;
          let minY = Number.POSITIVE_INFINITY;
          let maxX = Number.NEGATIVE_INFINITY;
          let maxY = Number.NEGATIVE_INFINITY;
          for (const n of placed) {
            const nAny = n as Node & { measured?: { width?: number; height?: number } };
            const w =
              typeof n.width === "number" && n.width > 0
                ? n.width
                : typeof nAny.measured?.width === "number" && nAny.measured.width > 0
                  ? nAny.measured.width
                  : 420;
            const h =
              typeof n.height === "number" && n.height > 0
                ? n.height
                : typeof nAny.measured?.height === "number" && nAny.measured.height > 0
                  ? nAny.measured.height
                  : 260;
            minX = Math.min(minX, n.position.x);
            minY = Math.min(minY, n.position.y);
            maxX = Math.max(maxX, n.position.x + w);
            maxY = Math.max(maxY, n.position.y + h);
          }
          if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
            return;
          }

          const currentCx = (minX + maxX) / 2;
          const currentCy = (minY + maxY) / 2;
          const targetCx = cellTopLeft.x + GRAPH_ASSIST_MATRIX_CELL_W / 2;
          const targetCy = cellTopLeft.y + GRAPH_ASSIST_SUBGRAPH_TOP_DY + GRAPH_ASSIST_MATRIX_CELL_H / 2;
          const dx = targetCx - currentCx;
          const dy = targetCy - currentCy;
          if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) return;

          flushSync(() => {
            setW((nds) =>
              nds.map((n) =>
                planNodeIds.has(n.id)
                  ? {
                      ...n,
                      position: { x: n.position.x + dx, y: n.position.y + dy },
                    }
                  : n,
              ),
            );
          });
        };

        for (let k = 0; k < seeds.length; k++) {
          if (cancelled || ac.signal.aborted) return;
          if (seeds.length > 1) {
            setGraphAssistLog([`generating… (${k + 1}/${seeds.length})`]);
          }
          const getWorkGraph = () => {
            const ext = readGraphForCanvas?.(graphAssistScopeRef.current.projectId);
            if (ext) return ext;
            const r = graphAssistReactFlowRef.current;
            return { nodes: r.getNodes(), edges: r.getEdges() };
          };
          const setWorkNodes = (updater: SetStateAction<Node[]>) => {
            setNodesForCanvas(graphAssistScopeRef.current.projectId, updater);
          };
          const setWorkEdges = (updater: SetStateAction<Edge[]>) => {
            setEdgesForCanvas(graphAssistScopeRef.current.projectId, updater);
          };
          const bumpViewport = async (ms: number) => {
            await graphAssistReactFlowRef.current.bumpGraphAssistViewport(ms);
          };

          const driveCtx: SelfDriveCanvasCtx = {
            setWorkNodes,
            setWorkEdges,
            getWorkGraph,
            bumpViewport,
          };

          // Multiple seeds share the canvas: each candidate graph gets its own matrix cell.
          const [row, col] = gridSlots[k]!;
          const cellTopLeft = {
            x: baseAnchor.x + col * GRAPH_ASSIST_MATRIX_CELL_W,
            y: baseAnchor.y + row * GRAPH_ASSIST_MATRIX_CELL_H,
          };
          const anchor = {
            x: cellTopLeft.x + GRAPH_ASSIST_MATRIX_CELL_INSET_X,
            y: cellTopLeft.y + GRAPH_ASSIST_MATRIX_CELL_INSET_Y,
          };
          const rng = createSeededRng(seeds[k]!);
          const plan = planRandomTrainerSubgraph(anchor, rng);
          trainerCleanupRows.push({ trainerId: plan.trainerId });
          if (cancelled || ac.signal.aborted) return;
          let trainSucceededForSeed = false;

          try {
            if (delayMs > 0) {
              await runPlannedSubgraphSteps(delayMs, plan, driveCtx);
            } else {
              await runPlannedSubgraphSteps(0, plan, driveCtx);
              await sleep(80);
            }
            await centerPlanWithinMatrixCell(plan, cellTopLeft, getWorkGraph, setWorkNodes);

            if (cancelled || ac.signal.aborted) return;

            if (isPhysicsOfAi) {
              setGraphAssistLog(["adding random observables…"]);
              const modelId = findModelNodeIdFromPlan(plan);
              if (!modelId) throw new Error("Generated graph has no model node.");
              const obsSeed = (Math.imul(seeds[k]!, 2654435761) ^ 0x504f41) >>> 0;
              const gBeforeObs = getWorkGraph();
              const obsItems = await createRandomUserObservablesForModel(
                modelId,
                gBeforeObs.nodes,
                gBeforeObs.edges,
                PHYSICS_OF_AI_OBSERVABLE_COUNT,
                obsSeed,
              );
              window.dispatchEvent(new Event(USER_OBSERVABLES_CHANGED));
              const withObs = appendUserObservableNodesToTrainer(
                gBeforeObs.nodes,
                gBeforeObs.edges,
                plan.trainerId,
                modelId,
                obsItems,
              );
              flushSync(() => {
                setWorkNodes(() => withObs.nodes);
                setWorkEdges(() => withObs.edges);
              });
              await bumpViewport(delayMs);
              if (cancelled || ac.signal.aborted) return;
            }

            if (!autoTrainInSelfDriving) {
              continue;
            }

            if (isPhysicsOfAi) {
              setGraphAssistLog(["training…"]);
            }

            const trainRes = await runTrainForPlan(plan, driveCtx);
            if (!trainRes.ok) {
              patchTrainerHostUi(plan.trainerId, null, setWorkNodes);
              appendGraphAssistFailureOverlay(getWorkGraph, plan, cellTopLeft, setWorkNodes, {
                phase: trainRes.phase ?? "train",
                reason: trainRes.reason ?? "unknown error",
              });
              void bumpViewport(0);
              continue;
            }
            trainSucceededForSeed = true;

            if (isPhysicsOfAi) {
              setGraphAssistLog(["CurveStarer · ranking by overall interestingness…"]);
              setCurveStarerOpen(true);
              const graphForCurveStarer = getWorkGraph();
              await analyzeCurveStarer(graphForCurveStarer.nodes, graphForCurveStarer.edges, "interestingness");
              setGraphAssistLog(["physics-of-ai agent complete — CurveStarer"]);
              continue;
            }

          } catch (err) {
            if (cancelled || ac.signal.aborted) return;
            const msg = err instanceof Error ? err.message : String(err);
            if (trainSucceededForSeed) {
              // Post-train processing failed.
              // Do not mark the candidate as training failure.
              setGraphAssistLog([`Post-train analysis warning (seed ${k + 1}): ${msg}`]);
              console.warn("Graph assist post-train analysis warning:", err);
              continue;
            }
            setGraphAssistLog([`Train failed (seed ${k + 1}): ${msg}`]);
            console.warn("Graph assist seed error:", err);
            patchTrainerHostUi(plan.trainerId, null, setWorkNodes);
            appendGraphAssistFailureOverlay(getWorkGraph, plan, cellTopLeft, setWorkNodes, {
              phase: "pre_train",
              reason: msg,
            });
            void bumpViewport(0);
            continue;
          }
        }

        setGraphAssistLog(
          isPhysicsOfAi
            ? ["physics-of-ai agent finished"]
            : seeds.length > 1
              ? [`generation completed (${seeds.length} graphs)`]
              : ["generation completed"],
        );
      } catch (e) {
        if (!cancelled && !ac.signal.aborted) {
          setGraphAssistLog([`Graph assist: ${e instanceof Error ? e.message : String(e)}`]);
          console.warn("Graph assist:", e);
        }
      } finally {
        for (const { trainerId } of trainerCleanupRows) {
          const pid = graphAssistScopeRef.current.projectId;
          patchTrainerHostUi(trainerId, null, (u) => setNodesForCanvas(pid, u));
        }
        setGraphAssistBusy(false);
        // Sync reset: deferring to setTimeout(0) left a window where a canvas click could re-run this
        // effect while mode was still self_driving and generate a duplicate batch.
        setGraphAssistMode("manual");
        if (graphAssistAbortRef.current === ac) graphAssistAbortRef.current = null;
      }
    })();

    return () => {
      cancelled = true;
      ac.abort();
    };
  }, [
    graphAssistMode,
    graphAssistDelaySecStr,
    graphAssistRunNonce,
    setNodesForCanvas,
    setEdgesForCanvas,
    readGraphForCanvas,
    analyzeCurveStarer,
  ]);

  const clipRef = useRef<ClipboardSubgraph | null>(null);
  const [pasteChoiceOpen, setPasteChoiceOpen] = useState(false);
  useBlurOpenSelectOnOutsidePointer();
  const [addNodeSearchScreenPos, setAddNodeSearchScreenPos] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const paneClickForDoubleRef = useRef<{ t: number; x: number; y: number } | null>(null);
  const pointerStartedOnCanvasRef = useRef(false);
  const pressedCanvasNodeIdRef = useRef<string | null>(null);
  const canvasNodePressRef = useRef<{
    nodeId: string;
    startClientX: number;
    startClientY: number;
    interactive: boolean;
    manualDragFallback: boolean;
    startPosition: { x: number; y: number };
    startedDrag: boolean;
    manuallyMoved: boolean;
  } | null>(null);
  const canvasNodePressTimerRef = useRef<number | null>(null);
  const suppressNextNodeClickRef = useRef(false);
  const interactiveNodeSelectionRef = useRef<{
    nodeId: string;
    selection: Array<{ id: string; selected: boolean }>;
  } | null>(null);

  const onPaneClick = useCallback((e: ReactMouseEvent) => {
    const clickedCanvasDirectly = e.target === e.currentTarget;
    const startedOnCanvasDirectly = pointerStartedOnCanvasRef.current;
    pointerStartedOnCanvasRef.current = false;
    if (!clickedCanvasDirectly || !startedOnCanvasDirectly) return;
    onRequestCloseRail();
    onRequestCloseNodeInformation();
    const now = Date.now();
    const prev = paneClickForDoubleRef.current;
    const { clientX, clientY } = e;
    if (
      prev &&
      now - prev.t < 420 &&
      Math.abs(clientX - prev.x) < 10 &&
      Math.abs(clientY - prev.y) < 10
    ) {
      paneClickForDoubleRef.current = null;
      e.preventDefault();
      setAddNodeSearchScreenPos({ x: clientX, y: clientY });
      return;
    }
    paneClickForDoubleRef.current = { t: now, x: clientX, y: clientY };
  }, [onRequestCloseNodeInformation, onRequestCloseRail]);

  const onSelectionChange = useCallback(
    ({ nodes: selectedNodes }: { nodes: Node[] }) => {
      if (!onCanvasSelectionChange) return;
      if (selectedNodes.length === 1) {
        const only = selectedNodes[0]!;
        onCanvasSelectionChange(only.id, only.type ?? null);
        return;
      }
      onCanvasSelectionChange(null, null);
    },
    [onCanvasSelectionChange],
  );

  const finishPaste = useCallback(
    (shareParams: boolean) => {
      setPasteChoiceOpen(false);
      const clip = clipRef.current;
      if (!clip?.nodes.length) return;
      const nds = getNodes();
      const eds = getEdges();
      const anchor = screenToFlowPosition({
        x: typeof window !== "undefined" ? window.innerWidth * 0.5 : 400,
        y: typeof window !== "undefined" ? window.innerHeight * 0.42 : 300,
      });
      const { nodes: pasteNodes, edges: pasteEdges } = cloneSubgraphForPaste(nds, clip, anchor, shareParams);
      setNodes(sortNodesParentBeforeChildren([...nds.map((n) => ({ ...n, selected: false })), ...pasteNodes]));
      setEdges([...eds, ...pasteEdges]);
    },
    [getEdges, getNodes, screenToFlowPosition, setEdges, setNodes],
  );

  useEffect(() => {
    const typingTarget = (t: EventTarget | null) => {
      if (!(t instanceof HTMLElement)) return false;
      const tag = t.tagName;
      return (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        t.isContentEditable ||
        Boolean(t.closest("[contenteditable=true]"))
      );
    };
    const onKey = (e: KeyboardEvent) => {
      if (typingTarget(e.target)) return;
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      if (e.key === "c" || e.key === "C") {
        const sel = nodes.filter((n) => n.selected);
        if (!sel.length) return;
        e.preventDefault();
        const ids = expandCombineSelectionNodeIds(sel.map((n) => n.id), nodes);
        const { nodes: subN, edges: subE } = extractSubgraphByNodeIds(ids, nodes, edges);
        clipRef.current = {
          nodes: structuredClone(subN) as Node[],
          edges: structuredClone(subE) as Edge[],
        };
        return;
      }
      if (e.key !== "v" && e.key !== "V") return;
      const clip = clipRef.current;
      if (!clip?.nodes.length) return;
      e.preventDefault();
      if (pasteChoiceOpen) return;
      setPasteChoiceOpen(true);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [edges, nodes, pasteChoiceOpen]);

  const [combineMenu, setCombineMenu] = useState<null | { x: number; y: number; ids: string[] }>(null);
  const [combineModal, setCombineModal] = useState<null | { ids: string[] }>(null);
  const [loopMenu, setLoopMenu] = useState<null | { x: number; y: number; nodeId: string; nodeType: string }>(null);
  const [loopModal, setLoopModal] = useState<null | { nodeId: string }>(null);
  const [renameCombinedModal, setRenameCombinedModal] = useState<null | { nodeId: string; initialName: string }>(null);
  const [clearCanvasConfirmOpen, setClearCanvasConfirmOpen] = useState(false);

  const runClearCanvas = useCallback(() => {
    setCombineMenu(null);
    setLoopMenu(null);
    setCombineModal(null);
    setLoopModal(null);
    setRenameCombinedModal(null);
    setPasteChoiceOpen(false);
    setAddNodeSearchScreenPos(null);
    onClearCanvas();
  }, [onClearCanvas]);

  useEffect(() => {
    if (!combineMenu && !loopMenu) return;
    const k = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") {
        setCombineMenu(null);
        setLoopMenu(null);
      }
    };
    document.addEventListener("keydown", k);
    return () => document.removeEventListener("keydown", k);
  }, [combineMenu, loopMenu]);

  const onClearCanvasClick = useCallback(() => {
    if (nodes.length > 0 || edges.length > 0) {
      setClearCanvasConfirmOpen(true);
      return;
    }
    runClearCanvas();
  }, [edges.length, nodes.length, runClearCanvas]);

  const openCombineMenuFromSelection = useCallback(
    (e: ReactMouseEvent | globalThis.MouseEvent, idsIn: string[]) => {
      const unique = [...new Set(idsIn)];
      if (unique.length < 2) return;
      e.preventDefault();
      e.stopPropagation();
      setLoopMenu(null);
      setCombineMenu({ x: e.clientX, y: e.clientY, ids: unique });
    },
    [],
  );

  const onSelectionContextMenu = useCallback(
    (e: ReactMouseEvent | globalThis.MouseEvent, selectedNodes: Node[]) => {
      if (selectedNodes.length >= 2) {
        openCombineMenuFromSelection(
          e,
          selectedNodes.map((n) => n.id),
        );
        return;
      }
      if (selectedNodes.length === 1) {
        const only = selectedNodes[0];
        e.preventDefault();
        e.stopPropagation();
        setCombineMenu(null);
        setLoopMenu({
          x: e.clientX,
          y: e.clientY,
          nodeId: only.id,
          nodeType: String(only.type ?? ""),
        });
      }
    },
    [openCombineMenuFromSelection],
  );

  const onNodeContextMenuForCombine = useCallback(
    (e: ReactMouseEvent | globalThis.MouseEvent, node: Node) => {
      const selected = nodes.filter((n) => n.selected);
      if (selected.length >= 2 && node.selected) {
        openCombineMenuFromSelection(
          e,
          selected.map((n) => n.id),
        );
        return;
      }
      if (selected.length === 1 && node.selected) {
        e.preventDefault();
        e.stopPropagation();
        setCombineMenu(null);
        setLoopMenu({
          x: e.clientX,
          y: e.clientY,
          nodeId: node.id,
          nodeType: String(node.type ?? ""),
        });
      }
    },
    [nodes, openCombineMenuFromSelection],
  );

  const onPaneContextMenuForCombine = useCallback(
    (e: ReactMouseEvent | globalThis.MouseEvent) => {
      const selected = nodes.filter((n) => n.selected);
      if (selected.length >= 2) {
        e.preventDefault();
        setLoopMenu(null);
        setCombineMenu({
          x: e.clientX,
          y: e.clientY,
          ids: selected.map((n) => n.id),
        });
        return;
      }
      if (selected.length === 1) {
        const only = selected[0];
        e.preventDefault();
        setCombineMenu(null);
        setLoopMenu({
          x: e.clientX,
          y: e.clientY,
          nodeId: only.id,
          nodeType: String(only.type ?? ""),
        });
      }
    },
    [nodes],
  );

  const confirmCombine = useCallback(
    async (name: string, saveToLibrary: boolean) => {
      if (!combineModal) return;
      const idSet = expandCombineSelectionNodeIds(combineModal.ids, nodes);
      const { nodes: subNodes, edges: subEdges } = extractSubgraphByNodeIds(idSet, nodes, edges);
      let templateId: string | undefined;
      if (saveToLibrary && subNodes.length > 0) {
        const raw = toApiDocument(subNodes, subEdges, null);
        const doc = applyGraphFileExportTier(raw, "small");
        const entry: SavedGraphEntry = {
          id:
            typeof crypto !== "undefined" && "randomUUID" in crypto
              ? crypto.randomUUID()
              : `sg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
          name: name.slice(0, 200),
          tier: "small",
          document: doc,
          savedAt: Date.now(),
          libraryOrigin: "combined_model",
        };
        try {
          await onSaveLibrarySourceEntry("workflow", entry);
          templateId = entry.id;
        } catch (err) {
          onGraphFileError(err instanceof Error ? err.message : String(err));
        }
      }
      const newId = `combined_model-${Math.random().toString(36).slice(2, 10)}`;
      const bridgeRewire = computeCombinedModelBridgeRewire(idSet, edges, newId, nodes);
      setEdges((eds) => {
        const kept = eds.filter((e) => {
          const sIn = idSet.has(e.source);
          const tIn = idSet.has(e.target);
          return (sIn && tIn) || (!sIn && !tIn);
        });
        return [...kept, ...bridgeRewire];
      });
      setNodes((nds) => {
        const cleared = nds.filter((n) => !idSet.has(n.id)).map((n) => ({ ...n, selected: false }));
        const selLocal = nds.filter((n) => idSet.has(n.id));
        const bounds = subgraphBoundingRectInFlow(selLocal, idSet);
        const parentW = Math.max(bounds.width + 2 * COMBINED_COMBINE_PAD, 236);
        const parentH = Math.max(bounds.height + 2 * COMBINED_COMBINE_PAD + COMBINED_COMBINE_HEAD, 168);
        const parentPos = {
          x: bounds.x - COMBINED_COMBINE_PAD,
          y: bounds.y - COMBINED_COMBINE_PAD - COMBINED_COMBINE_HEAD,
        };
        const base = defaultCombinedModelData({
          displayName: name.trim().slice(0, 200) || "Combined model",
          templateId,
          sourceNodeCount: subNodes.length,
          __expandedFrame: { width: parentW, height: parentH },
          ioMode: "input-output",
        }) as Record<string, unknown>;
        const parent: Node = {
          id: newId,
          type: "combined_model",
          position: parentPos,
          style: { width: parentW, height: parentH },
          data: withNewInstanceTitle(cleared, "combined_model", base),
          selected: true,
        };
        const selById = new Map(selLocal.map((n) => [n.id, n]));
        const children: Node[] = selLocal.map((n) => {
          const pid = n.parentId != null && n.parentId !== "" ? String(n.parentId) : "";
          const rootInSelection = !pid || !idSet.has(pid);
          const abs = clipAbsolutePosition(n, selById, idSet);
          if (rootInSelection) {
            return {
              ...n,
              parentId: newId,
              position: { x: abs.x - parentPos.x, y: abs.y - parentPos.y },
              extent: "parent" as const,
              selected: false,
              hidden: false,
            };
          }
          return { ...n, selected: false, hidden: false };
        });
        const merged = sortNodesParentBeforeChildren([...cleared, parent, ...children]);
        return refitCombinedModelShellsToChildren(migrateCombinedModelChildChromeInset(merged));
      });
      setCombineModal(null);
      setCombineMenu(null);
      setLoopMenu(null);
    },
    [
      combineModal,
      edges,
      nodes,
      onGraphFileError,
      onSaveLibrarySourceEntry,
      setEdges,
      setNodes,
    ],
  );

  const cancelCombineModal = useCallback(() => setCombineModal(null), []);

  const cancelLoopModal = useCallback(() => setLoopModal(null), []);
  const cancelRenameCombinedModal = useCallback(() => setRenameCombinedModal(null), []);

  const confirmLoopCount = useCallback(
    async (loopCount: number, loopShareParams: boolean) => {
      if (!loopModal) return;
      const id = loopModal.nodeId;
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const data = { ...((n.data as Record<string, unknown>) ?? {}) };
          if (loopCount < 2) {
            delete data.loopCount;
            delete data.loopShareParams;
          } else {
            data.loopCount = loopCount;
            if (loopShareParams) data.loopShareParams = true;
            else delete data.loopShareParams;
          }
          return { ...n, data };
        }),
      );
      setLoopModal(null);
      setLoopMenu(null);
    },
    [loopModal, setNodes],
  );

  const confirmRenameCombined = useCallback(
    async (name: string) => {
      if (!renameCombinedModal) return;
      const nextName = name.trim().slice(0, 200) || "Combined model";
      const id = renameCombinedModal.nodeId;
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id || String(n.type) !== "combined_model") return n;
          const prev = (n.data ?? {}) as Record<string, unknown>;
          return {
            ...n,
            data: {
              ...prev,
              displayName: nextName,
              [INSTANCE_TITLE_KEY]: nextName,
            },
          };
        }),
      );
      setRenameCombinedModal(null);
      setLoopMenu(null);
    },
    [renameCombinedModal, setNodes],
  );

  const decomposeCombinedModel = useCallback(
    (combinedId: string) => {
      const nodesNow = getNodes();
      const shell = nodesNow.find((n) => n.id === combinedId && String(n.type) === "combined_model");
      if (!shell) return;
      const subtreeIds = collectStrictDescendantIds(nodesNow, combinedId);
      const byId = new Map(nodesNow.map((n) => [n.id, n]));

      setEdges((eds) => {
        const incomingShellEdges = eds.filter((e) => e.target === combinedId);
        const outgoingShellEdges = eds.filter((e) => e.source === combinedId);
        const shellBoundaryOut = outgoingShellEdges.filter((e) => (e.sourceHandle ?? "").trim() === "tensor_boundary");
        const shellReturnIn = incomingShellEdges.filter(
          (e) => (e.targetHandle ?? "").trim() === COMBINED_MODEL_RETURN_TARGET_HANDLE,
        );
        const externalIncoming = incomingShellEdges.filter((e) => !subtreeIds.has(e.source));
        const externalOutgoing = outgoingShellEdges.filter((e) => !subtreeIds.has(e.target));
        const kept = eds.filter((e) => e.source !== combinedId && e.target !== combinedId);
        const seen = new Set(
          kept.map(
            (e) => `${e.source}|${e.sourceHandle ?? ""}|${e.target}|${e.targetHandle ?? ""}|${String(e.type ?? "")}`,
          ),
        );
        const rewired: Edge[] = [];
        const addUnique = (edge: Edge) => {
          const k = `${edge.source}|${edge.sourceHandle ?? ""}|${edge.target}|${edge.targetHandle ?? ""}|${String(edge.type ?? "")}`;
          if (seen.has(k)) return;
          seen.add(k);
          rewired.push(edge);
        };
        for (const extIn of externalIncoming) {
          for (const bridge of shellBoundaryOut) {
            const targetInside = byId.get(bridge.target);
            if (!targetInside || !subtreeIds.has(targetInside.id)) continue;
            addUnique({
              ...extIn,
              id: `e-${Math.random().toString(36).slice(2, 12)}`,
              target: bridge.target,
              targetHandle: bridge.targetHandle ?? null,
            });
          }
        }
        for (const ret of shellReturnIn) {
          for (const extOut of externalOutgoing) {
            const sourceInside = byId.get(ret.source);
            if (!sourceInside || !subtreeIds.has(sourceInside.id)) continue;
            addUnique({
              ...extOut,
              id: `e-${Math.random().toString(36).slice(2, 12)}`,
              source: ret.source,
              sourceHandle: ret.sourceHandle ?? null,
            });
          }
        }
        return [...kept, ...rewired];
      });

      setNodes((nds) => {
        const directChildIds = new Set(
          nds.filter((n) => n.parentId != null && String(n.parentId) === combinedId).map((n) => n.id),
        );
        const next = nds
          .filter((n) => n.id !== combinedId)
          .map((n) => {
            if (n.parentId == null || String(n.parentId) !== combinedId) return n;
            return {
              ...n,
              parentId: undefined,
              extent: undefined,
              hidden: false,
              position: { x: n.position.x + shell.position.x, y: n.position.y + shell.position.y },
              selected: true,
            };
          })
          .map((n) => ({ ...n, selected: directChildIds.has(n.id) }));
        return sortNodesParentBeforeChildren(next);
      });
      setLoopMenu(null);
      setRenameCombinedModal(null);
    },
    [getNodes, setEdges, setNodes],
  );

  const loopModalInitialCount = useMemo(() => {
    if (!loopModal) return 2;
    const n = nodes.find((x) => x.id === loopModal.nodeId);
    const c = readGraphNodeLoopCount(n?.data);
    return c != null && c >= 2 ? c : 2;
  }, [loopModal, nodes]);

  const loopModalInitialShareParams = useMemo(() => {
    if (!loopModal) return false;
    const n = nodes.find((x) => x.id === loopModal.nodeId);
    return readGraphNodeLoopShareParams(n?.data);
  }, [loopModal, nodes]);

  const onDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const updateLibraryDeleteTarget = useCallback(
    (event: Pick<MouseEvent, "clientX" | "clientY">) => {
      const nodesButton = document.querySelector<HTMLElement>('[data-cr-rail-section="nodes"]');
      const rect = nodesButton?.getBoundingClientRect();
      const overCollapsedNodesButton =
        !nodesRailOpen &&
        rect !== undefined &&
        event.clientX >= rect.left &&
        event.clientX <= rect.right &&
        event.clientY >= rect.top &&
        event.clientY <= rect.bottom;
      if (overCollapsedNodesButton) onRequestOpenNodesRail();
      updateLibraryNodeDragTarget(event.clientX, event.clientY);
    },
    [nodesRailOpen, onRequestOpenNodesRail],
  );

  const selectCanvasNodeOnPress = useCallback((nodeId: string) => {
    const selectionChanges = getNodes().flatMap((node) => {
      const selected = node.id === nodeId;
      return node.selected === selected ? [] : [{ type: "select" as const, id: node.id, selected }];
    });
    if (selectionChanges.length > 0) onNodesChange(selectionChanges);
  }, [getNodes, onNodesChange]);

  const beginPressedCanvasNodeDrag = useCallback((nodeId: string) => {
    pressedCanvasNodeIdRef.current = nodeId;
    markLibraryDragNode(nodeId);
    beginLibraryNodeDrag();
  }, []);

  const clearCanvasNodePressTimer = useCallback(() => {
    if (canvasNodePressTimerRef.current === null) return;
    window.clearTimeout(canvasNodePressTimerRef.current);
    canvasNodePressTimerRef.current = null;
  }, []);

  const startCanvasNodePressDrag = useCallback((nodeId: string) => {
    const press = canvasNodePressRef.current;
    if (!press || press.nodeId !== nodeId || press.interactive || press.startedDrag) return;
    press.startedDrag = true;
    clearCanvasNodePressTimer();
    beginPressedCanvasNodeDrag(nodeId);
  }, [beginPressedCanvasNodeDrag, clearCanvasNodePressTimer]);

  // React Flow's onNodeDragStart only fires after its movement threshold; a
  // long press starts the matching visual state before that threshold.
  useEffect(() => {
    const moveCanvasNodePress = (event: PointerEvent) => {
      const press = canvasNodePressRef.current;
      if (!press || press.interactive) return;
      const dx = event.clientX - press.startClientX;
      const dy = event.clientY - press.startClientY;
      if (!press.startedDrag) {
        if (Math.hypot(dx, dy) < 2) return;
        startCanvasNodePressDrag(press.nodeId);
      }
      // A few parameter-label shells use XYFlow's `nodrag` solely to protect
      // their input control. The label itself remains a node gesture, so move
      // it through the same canvas state when XYFlow declines the native drag.
      if (!press.manualDragFallback || Math.hypot(dx, dy) < 2) return;
      const zoom = getViewport().zoom || 1;
      press.manuallyMoved = true;
      setNodes((current) =>
        current.map((node) =>
          node.id === press.nodeId
            ? {
                ...node,
                position: { x: press.startPosition.x + dx / zoom, y: press.startPosition.y + dy / zoom },
                dragging: true,
              }
            : node,
        ),
      );
      updateLibraryDeleteTarget(event);
    };
    const endPressedCanvasNodeDrag = (event?: PointerEvent) => {
      const press = canvasNodePressRef.current;
      clearCanvasNodePressTimer();
      canvasNodePressRef.current = null;
      if (!press) return;
      if (!press.startedDrag) {
        selectCanvasNodeOnPress(press.nodeId);
        return;
      }
      pressedCanvasNodeIdRef.current = null;
      endLibraryNodeDrag();
      if (press.manualDragFallback && press.manuallyMoved) {
        if (event && isOverNodesLibrary(event.clientX, event.clientY)) {
          setNodes((current) => current.filter((node) => node.id !== press.nodeId));
          setEdges((current) => current.filter((edge) => edge.source !== press.nodeId && edge.target !== press.nodeId));
        } else {
          setNodes((current) =>
            current.map((node) => (node.id === press.nodeId ? { ...node, dragging: false } : node)),
          );
        }
      }
      // Prevent React Flow's click handler from converting a completed drag
      // back into a click selection after pointerup.
      suppressNextNodeClickRef.current = true;
      window.setTimeout(() => {
        suppressNextNodeClickRef.current = false;
      }, 0);
    };
    const cancelPressedCanvasNodeDrag = () => endPressedCanvasNodeDrag();
    window.addEventListener("pointermove", moveCanvasNodePress, { passive: true });
    window.addEventListener("pointerup", endPressedCanvasNodeDrag);
    window.addEventListener("pointercancel", cancelPressedCanvasNodeDrag);
    window.addEventListener("blur", cancelPressedCanvasNodeDrag);
    return () => {
      window.removeEventListener("pointermove", moveCanvasNodePress);
      window.removeEventListener("pointerup", endPressedCanvasNodeDrag);
      window.removeEventListener("pointercancel", cancelPressedCanvasNodeDrag);
      window.removeEventListener("blur", cancelPressedCanvasNodeDrag);
    };
  }, [clearCanvasNodePressTimer, getViewport, selectCanvasNodeOnPress, setEdges, setNodes, startCanvasNodePressDrag, updateLibraryDeleteTarget]);

  const onNodeDragStart = useCallback((_event: ReactMouseEvent, node: Node) => {
    const press = canvasNodePressRef.current;
    if (press?.nodeId === node.id) {
      if (!press.interactive) startCanvasNodePressDrag(node.id);
      return;
    }
    beginPressedCanvasNodeDrag(node.id);
  }, [beginPressedCanvasNodeDrag, startCanvasNodePressDrag]);

  const onNodeDrag = useCallback(
    (event: ReactMouseEvent) => updateLibraryDeleteTarget(event),
    [updateLibraryDeleteTarget],
  );

  /** Dropping an existing canvas node over the Nodes rail is a delete gesture. */
  const onNodeDragStop = useCallback(
    (event: ReactMouseEvent, node: Node) => {
      const overNodesRail = isOverNodesLibrary(event.clientX, event.clientY);
      const press = canvasNodePressRef.current;
      if (press?.nodeId === node.id) press.startedDrag = true;
      clearCanvasNodePressTimer();
      pressedCanvasNodeIdRef.current = null;
      endLibraryNodeDrag();
      if (!overNodesRail) return;

      onNodesChange([{ type: "remove", id: node.id }]);
      setEdges((current) => current.filter((edge) => edge.source !== node.id && edge.target !== node.id));
    },
    [clearCanvasNodePressTimer, onNodesChange, setEdges],
  );

  /** Resolve a press against the rendered node rectangle, not only the event
   * target's ancestor chain. Some node bodies (for example KAN reg) contain
   * overflowed/interactive content whose painted area can exceed that chain's
   * ordinary hit box. */
  const findCanvasNodeAtPoint = useCallback((target: EventTarget | null, clientX: number, clientY: number) => {
    const canvas = canvasWrapRef.current;
    if (!canvas) return null;
    if (target instanceof Element) {
      const direct = target.closest<HTMLElement>(".react-flow__node");
      if (direct && canvas.contains(direct)) return direct;
    }
    const candidates = [...canvas.querySelectorAll<HTMLElement>(".react-flow__node")];
    for (let index = candidates.length - 1; index >= 0; index -= 1) {
      const node = candidates[index]!;
      const visibleNode = node.querySelector<HTMLElement>(".cr-node") ?? node;
      const rect = visibleNode.getBoundingClientRect();
      if (clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom) {
        return node;
      }
    }
    return null;
  }, []);

  /*
   * Keep this callback separate from React Flow's movement-based drag
   * callbacks: it is the press-time definition of a canvas node drag.
   */
  const onCanvasPointerDownCapture = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const target = event.target;
    pointerStartedOnCanvasRef.current =
      target instanceof Element && target.classList.contains("react-flow__pane");
    if (event.button !== 0 || !(target instanceof Element)) return;
    const node = findCanvasNodeAtPoint(target, event.clientX, event.clientY);
    if (!node) return;
    const nodeId = node.dataset.id;
    if (!nodeId) return;
    const isInteractive = isInteractiveNodePressTarget(target);
    clearCanvasNodePressTimer();
    // Selection and dragging deliberately share one hit region. Interactive
    // child controls keep their own behavior and are not node gestures.
    if (isInteractive) {
      interactiveNodeSelectionRef.current = {
        nodeId,
        selection: getNodes().map((candidate) => ({ id: candidate.id, selected: Boolean(candidate.selected) })),
      };
      canvasNodePressRef.current = null;
      return;
    }
    interactiveNodeSelectionRef.current = null;
    const startNode = getNodes().find((candidate) => candidate.id === nodeId);
    canvasNodePressRef.current = {
      nodeId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      interactive: false,
      manualDragFallback: Boolean(target.closest(".nodrag")),
      startPosition: { x: startNode?.position.x ?? 0, y: startNode?.position.y ?? 0 },
      startedDrag: false,
      manuallyMoved: false,
    };
    canvasNodePressTimerRef.current = window.setTimeout(() => {
      startCanvasNodePressDrag(nodeId);
    }, 180);
  }, [clearCanvasNodePressTimer, findCanvasNodeAtPoint, getNodes, startCanvasNodePressDrag]);

  const onCanvasWheelCapture = useCallback(
    (event: React.WheelEvent<HTMLDivElement>) => {
      if (event.deltaY === 0) return;
      // Ctrl/Cmd+wheel (and trackpad pinch) zooms unconditionally; a plain
      // wheel zooms only over surfaces without native wheel semantics
      // (issue #170 — scrollable panels/modals keep scrolling).
      if (!event.ctrlKey && !event.metaKey) {
        const target = event.target;
        if (!(target instanceof Element)) return;
        if (!wheelTargetAllowsCanvasZoom(target, event.currentTarget, event.deltaX, event.deltaY)) {
          return;
        }
      }
      event.preventDefault();
      event.stopPropagation();
      const current = getViewport();
      const isMacPinch = event.ctrlKey && /Mac|iPhone|iPad|iPod/.test(navigator.platform);
      const nextZoom = getContinuousWheelZoom(
        current.zoom,
        event.deltaY,
        event.deltaMode,
        isMacPinch,
        CANVAS_ZOOM_LEVELS[0],
        CANVAS_ZOOM_LEVELS.at(-1)!,
      );
      if (nextZoom === current.zoom) return;

      const rect = event.currentTarget.getBoundingClientRect();
      const pointerX = event.clientX - rect.left;
      const pointerY = event.clientY - rect.top;
      const scale = nextZoom / current.zoom;
      void setViewport(
        {
          x: pointerX - (pointerX - current.x) * scale,
          y: pointerY - (pointerY - current.y) * scale,
          zoom: nextZoom,
        },
      );
    },
    [getViewport, setViewport],
  );

  const onCanvasClickCapture = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const node = target.closest<HTMLElement>(".react-flow__node");
    if (!node) return;
    if (suppressNextNodeClickRef.current) {
      suppressNextNodeClickRef.current = false;
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    const snapshot = interactiveNodeSelectionRef.current;
    if (!snapshot || snapshot.nodeId !== node.dataset.id) return;
    interactiveNodeSelectionRef.current = null;
    // Let the child control receive its click, then undo React Flow's bubbling
    // node-selection update so interactive content stays outside both gesture
    // regions.
    window.requestAnimationFrame(() => {
      const wanted = new Map(snapshot.selection.map((entry) => [entry.id, entry.selected]));
      const changes = getNodes().flatMap((candidate) => {
        const selected = wanted.get(candidate.id) ?? false;
        return candidate.selected === selected ? [] : [{ type: "select" as const, id: candidate.id, selected }];
      });
      if (changes.length > 0) onNodesChange(changes);
    });
  }, [getNodes, onNodesChange]);

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      const nodeType = (e.dataTransfer.getData(DND_MIME) || e.dataTransfer.getData(DND_TEXT_PLAIN)).trim();
      if (!nodeType || !(nodeType in researchNodeTypes)) return;
      if (nodeType === "observable_user") {
        const raw = e.dataTransfer.getData(USER_OBSERVABLE_DND_MIME);
        if (!raw) return;
        try {
          const parsed = JSON.parse(raw) as {
            userObservableId?: string;
            label?: string;
            tensorVizNodeId?: string;
            tensorSelectorNodeId?: string;
          };
          addNodeImplRef.current(nodeType, { x: e.clientX, y: e.clientY }, {
            userObservableId: parsed.userObservableId,
            label: parsed.label,
            tensorVizNodeId: parsed.tensorVizNodeId,
            tensorSelectorNodeId: parsed.tensorSelectorNodeId,
          });
        } catch {
          return;
        }
        return;
      }
      if (nodeType === "linear_dataset") {
        const rawLd = e.dataTransfer.getData(USER_LINEAR_DATASET_DND_MIME);
        if (rawLd) {
          try {
            const parsed = JSON.parse(rawLd) as { userLinearDatasetId?: string };
            if (parsed.userLinearDatasetId) {
              addNodeImplRef.current(nodeType, { x: e.clientX, y: e.clientY }, {
                userLinearDatasetId: parsed.userLinearDatasetId,
              });
              return;
            }
          } catch {
            /* fall through */
          }
        }
      }
      if (nodeType === "symbolic_func_dataset") {
        const rawSfd = e.dataTransfer.getData(USER_SYMBOLIC_FUNC_DATASET_DND_MIME);
        if (rawSfd) {
          try {
            const parsed = JSON.parse(rawSfd) as { userSymbolicFuncDatasetId?: string };
            if (parsed.userSymbolicFuncDatasetId) {
              addNodeImplRef.current(nodeType, { x: e.clientX, y: e.clientY }, {
                userSymbolicFuncDatasetId: parsed.userSymbolicFuncDatasetId,
              });
              return;
            }
          } catch {
            /* fall through */
          }
        }
      }
      if (nodeType === "combined_model") {
        const rawCm = e.dataTransfer.getData(COMBINED_MODEL_TEMPLATE_DND_MIME);
        if (rawCm) {
          try {
            const parsed = JSON.parse(rawCm) as {
              templateId?: string;
              displayName?: string;
              sourceNodeCount?: number;
              document?: GraphDocument;
            };
            if (parsed.templateId) {
              addNodeImplRef.current(nodeType, { x: e.clientX, y: e.clientY }, {
                combinedModelTemplateId: parsed.templateId,
                combinedModelDisplayName: parsed.displayName,
                combinedModelSourceNodeCount: parsed.sourceNodeCount,
                combinedModelTemplateDocument: parsed.document,
              });
              return;
            }
          } catch {
            return;
          }
        }
        addNodeImplRef.current(nodeType, { x: e.clientX, y: e.clientY });
        return;
      }
      addNodeImplRef.current(nodeType, { x: e.clientX, y: e.clientY });
    },
    [addNodeImplRef],
  );

  const applyConnectionWithEffects = useCallback(
    (params: Connection, baseNodes: Node[], baseEdges: Edge[]) => applyCanvasConnection(params, baseNodes, baseEdges),
    [],
  );

  const onConnect = useCallback(
    (params: Connection) => {
      connectionCommittedRef.current = true;
      const resolvedParams = connectionSnapTargetRef.current?.connection ?? params;
      const endpointIds = [resolvedParams.source, resolvedParams.target].filter((id): id is string => Boolean(id));
      if (endpointIds.length) updateNodeInternals(endpointIds);
      const next = applyConnectionWithEffects(resolvedParams, nodes, edges);
      setNodes(next.nodes);
      setEdges(next.edges);
      if (endpointIds.length) {
        window.requestAnimationFrame(() => updateNodeInternals(endpointIds));
      }
    },
    [applyConnectionWithEffects, edges, nodes, setEdges, setNodes, updateNodeInternals],
  );

  const findNearestValidConnectionSnap = useCallback(
    (clientX: number, clientY: number): ConnectionSnapTarget | null => {
      const active = activeConnectionRef.current;
      if (!active || typeof document === "undefined") return null;

      const requiredType = active.handleType === "source" ? "target" : "source";
      let closest: ConnectionSnapTarget | null = null;
      let closestDistance = 128;

      document.querySelectorAll<HTMLElement>(".react-flow__handle").forEach((handle) => {
        if (!handle.classList.contains(requiredType)) return;
        const nodeId = handle.closest<HTMLElement>(".react-flow__node")?.dataset.id;
        if (!nodeId) return;

        const handleId = handle.dataset.handleid ?? null;
        const connection: Connection =
          active.handleType === "source"
            ? { source: active.nodeId, sourceHandle: active.handleId, target: nodeId, targetHandle: handleId }
            : { source: nodeId, sourceHandle: handleId, target: active.nodeId, targetHandle: active.handleId };
        if (!isValidCanvasConnection(connection, nodes, edges)) return;

        const rect = handle.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const distance = Math.hypot(centerX - clientX, centerY - clientY);
        if (distance > closestDistance) return;

        closestDistance = distance;
        const position = screenToFlowPosition({ x: centerX, y: centerY });
        closest = { connection, x: position.x, y: position.y };
      });

      return closest;
    },
    [edges, nodes, screenToFlowPosition],
  );

  const onConnectStart = useCallback(
    (_event: MouseEvent | TouchEvent, params: { nodeId: string | null; handleId: string | null; handleType: "source" | "target" | null }) => {
      if (!params.nodeId || (params.handleType !== "source" && params.handleType !== "target")) return;
      connectionPointerCleanupRef.current?.();
      activeConnectionRef.current = { nodeId: params.nodeId, handleId: params.handleId, handleType: params.handleType };
      connectionCommittedRef.current = false;
      connectionSnapTargetRef.current = null;
      setConnectionSnapTarget(null);

      const onPointerMove = (event: MouseEvent) => {
        const next = findNearestValidConnectionSnap(event.clientX, event.clientY);
        connectionSnapTargetRef.current = next;
        setConnectionSnapTarget(next);
      };
      const cleanup = () => {
        window.removeEventListener("mousemove", onPointerMove);
        connectionPointerCleanupRef.current = null;
      };
      connectionPointerCleanupRef.current = cleanup;
      window.addEventListener("mousemove", onPointerMove);
    },
    [findNearestValidConnectionSnap],
  );

  const onConnectEnd = useCallback(() => {
    const snapped = connectionSnapTargetRef.current;
    if (!connectionCommittedRef.current && snapped) onConnect(snapped.connection);
    connectionPointerCleanupRef.current?.();
    activeConnectionRef.current = null;
    connectionSnapTargetRef.current = null;
    setConnectionSnapTarget(null);
  }, [onConnect]);

  const isValidConnection = useCallback(
    (edge: Connection | Edge) => isValidCanvasConnection(edge, nodes, edges),
    [nodes, edges],
  );

  const onAutoConnectCanvas = useCallback(() => {
    // 规划逻辑抽至 graph/connectionRules.planAutoConnectCanvas(纯函数);
    // 此处仅保留副作用应用循环。
    const plannedConnections = planAutoConnectCanvas(nodes, edges);
    if (!plannedConnections.length) return;
    let nextNodes = nodes;
    let nextEdges = edges;
    for (const conn of plannedConnections) {
      const next = applyConnectionWithEffects(conn, nextNodes, nextEdges);
      nextNodes = next.nodes;
      nextEdges = next.edges;
    }
    setNodes(nextNodes);
    setEdges(nextEdges);
    const existingNodeIds = new Set(nodes.map((node) => node.id));
    const affectedNodeIds = [...new Set([
      ...plannedConnections.flatMap((connection) => [connection.source, connection.target]),
      ...nextNodes.filter((node) => !existingNodeIds.has(node.id)).map((node) => node.id),
    ].filter((id): id is string => Boolean(id)))];
    if (affectedNodeIds.length) {
      // Auto Connect bypasses React Flow's pointer lifecycle, so no handle
      // measurement is requested automatically. Wait for the batched node
      // update to paint, then refresh twice to cover node side-effects that
      // alter their DOM layout during the same commit.
      window.requestAnimationFrame(() => {
        updateNodeInternals(affectedNodeIds);
        window.requestAnimationFrame(() => updateNodeInternals(affectedNodeIds));
      });
    }
  }, [applyConnectionWithEffects, edges, nodes, setEdges, setNodes, updateNodeInternals]);

  const onAutoLayoutCanvas = useCallback(() => {
    const currentNodes = getStableNodes();
    if (!currentNodes.length) return;
    const result = layoutResearchGraphNodes(currentNodes, getStableEdges());
    if (!result.changed) {
      void fitView({ padding: 0.18, duration: 320 });
      return;
    }
    setNodes(result.nodes);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        void fitView({ padding: 0.18, duration: 420 });
      });
    });
  }, [fitView, getStableEdges, getStableNodes, setNodes]);

  const selectedTrainerNode = useMemo(() => {
    const selected = nodes.filter((n) => n.selected && n.type === "trainer");
    if (selected.length === 1) return selected[0]!;
    return null;
  }, [nodes]);
  const selectedTrainerId = selectedTrainerNode?.id ?? null;
  const selectedTrainerData = ((selectedTrainerNode?.data ?? {}) as Partial<TrainerNodeData>) ?? {};
  const canCreateTargetCurve = !!selectedTrainerId;
  const canStartAutoTuning =
    !!selectedTrainerId &&
    Array.isArray(selectedTrainerData.targetCurveStepTicks) &&
    Array.isArray(selectedTrainerData.targetCurveLossHistory) &&
    selectedTrainerData.targetCurveStepTicks.length >= 2 &&
    selectedTrainerData.targetCurveLossHistory.length === selectedTrainerData.targetCurveStepTicks.length &&
    !autoTuneBusy;
  const canViewAutoTuneComparison = !!selectedTrainerId && !!selectedTrainerData.autoTuneComparisonResult;

  const axisSuggestions = useMemo(() => buildAutoTuneAxisSuggestions(nodes), [nodes]);

  const patchSelectedTrainerData = useCallback(
    (patch: Partial<TrainerNodeData>) => {
      if (!selectedTrainerId) return;
      setNodes((prev) =>
        prev.map((n) => {
          if (n.id !== selectedTrainerId || n.type !== "trainer") return n;
          const d = (n.data ?? {}) as Record<string, unknown>;
          return { ...n, data: { ...d, ...patch } };
        }),
      );
    },
    [selectedTrainerId, setNodes],
  );

  const applyBestParamsToNodes = useCallback(
    (params: Record<string, unknown>) => {
      const setByPath = (root: Record<string, unknown>, path: string, value: unknown) => {
        const parts = path.split(".").filter(Boolean);
        if (!parts.length) return;
        let cur: Record<string, unknown> = root;
        for (let i = 0; i < parts.length - 1; i++) {
          const k = parts[i]!;
          const nxt = cur[k];
          if (!nxt || typeof nxt !== "object" || Array.isArray(nxt)) {
            const created: Record<string, unknown> = {};
            cur[k] = created;
            cur = created;
          } else {
            cur = nxt as Record<string, unknown>;
          }
        }
        cur[parts[parts.length - 1]!] = value;
      };
      setNodes((prev) =>
        prev.map((n) => {
          const nextData = { ...((n.data ?? {}) as Record<string, unknown>) };
          let touched = false;
          for (const [k, v] of Object.entries(params)) {
            const split = k.split(".");
            if (split.length < 2 || split[0] !== n.id) continue;
            touched = true;
            setByPath(nextData, split.slice(1).join("."), v);
          }
          return touched ? { ...n, data: nextData } : n;
        }),
      );
    },
    [setNodes],
  );

  const runCoordinateDescent = useCallback(
    async (cfg: AutoTuneConfig) => {
      if (!selectedTrainerId) return;
      const tSteps = selectedTrainerData.targetCurveStepTicks ?? [];
      const tLoss = selectedTrainerData.targetCurveLossHistory ?? [];
      if (tSteps.length < 2 || tSteps.length !== tLoss.length) return;
      const sessionId = `cd-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
      autoTuneSessionRef.current = sessionId;
      setAutoTuneBusy(true);
      setAutoTuneStatus("starting coordinate descent...");
      setAutoTuneModalOpen(false);
      try {
        const payload = {
          trainer_node_id: selectedTrainerId,
          nodes: nodes
            .filter((n) => String(n.type) !== "graph_assist_failure_overlay")
            .map(serializeNodeForTrain),
          edges: edges.map((e) => ({
            id: e.id,
            source: e.source,
            target: e.target,
            sourceHandle: e.sourceHandle ?? null,
            targetHandle: e.targetHandle ?? null,
          })),
          session_id: sessionId,
          axes: cfg.axes.map((a) => ({ node_id: a.nodeId, path: a.path, values: a.values })),
          max_rounds: cfg.maxRounds,
          target_step_ticks: tSteps,
          target_loss_history: tLoss,
          score: {
            objective: "closest_to_target",
            end_weight: cfg.endWeight,
            smoothness_weight: cfg.smoothnessWeight,
          },
        };
        const res = await fetch("/api/train/coordinate-descent", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          let detail = res.statusText;
          try {
            const j = (await res.json()) as { detail?: unknown };
            if (j?.detail != null) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          } catch {
            /* ignore */
          }
          throw new Error(detail);
        }
        const reader = res.body?.getReader();
        if (!reader) throw new Error("No response body");
        await readNdjsonCoordinateDescentStream(reader, (ev) => {
          if (ev.type === "round_started") {
            setAutoTuneStatus(`round ${ev.round_index + 1}...`);
            return;
          }
          if (ev.type === "candidate_evaluated") {
            if (ev.error) setAutoTuneStatus(`candidate failed: ${ev.error}`);
            else if (typeof ev.score === "number") setAutoTuneStatus(`candidate score ${ev.score.toExponential(3)}`);
            return;
          }
          if (ev.type === "axis_best_selected") {
            if (typeof ev.score === "number") setAutoTuneStatus(`axis best (${ev.axis_key ?? "axis"}): ${ev.score.toExponential(3)}`);
            return;
          }
          if (ev.type === "tuning_complete") {
            const bestScore = typeof ev.best_score === "number" ? ev.best_score : Number.NaN;
            const summary = Number.isFinite(bestScore)
              ? `auto-tune best score=${bestScore.toExponential(3)}`
              : "auto-tune finished";
            const comparison = mapTuningCompleteToComparisonResult(ev);
            setAutoTuneResults(comparison);
            setAutoTuneResultsOpen(true);
            patchSelectedTrainerData({
              lastAutoTuneSummary: summary,
              autoTuneComparisonResult: comparison,
              lossHistory: ev.best_curve?.loss_history ?? selectedTrainerData.lossHistory,
              stepTicks: ev.best_curve?.step_ticks ?? selectedTrainerData.stepTicks,
              testLossHistory: ev.best_curve?.test_loss_history ?? selectedTrainerData.testLossHistory,
            });
            if (ev.best_params && typeof ev.best_params === "object") {
              applyBestParamsToNodes(ev.best_params as Record<string, unknown>);
            }
            setAutoTuneStatus(summary);
            return;
          }
          if (ev.type === "tuning_aborted") setAutoTuneStatus("auto-tuning aborted");
        });
      } catch (e) {
        setAutoTuneStatus(e instanceof Error ? e.message : String(e));
      } finally {
        setAutoTuneBusy(false);
        autoTuneSessionRef.current = null;
      }
    },
    [
      applyBestParamsToNodes,
      edges,
      nodes,
      patchSelectedTrainerData,
      selectedTrainerData.lossHistory,
      selectedTrainerData.stepTicks,
      selectedTrainerData.targetCurveLossHistory,
      selectedTrainerData.targetCurveStepTicks,
      selectedTrainerData.testLossHistory,
      selectedTrainerId,
    ],
  );

  const abortCoordinateDescent = useCallback(async () => {
    const sessionId = autoTuneSessionRef.current;
    if (!sessionId) return;
    try {
      await fetch("/api/train/coordinate-descent/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, action: "abort" }),
      });
      setAutoTuneStatus("abort requested...");
    } catch (e) {
      setAutoTuneStatus(e instanceof Error ? e.message : String(e));
    }
  }, []);

  return (
    <FlowSurfaceProvider
      value={{
        projectId: owningProjectId,
        applyNodes: setNodes,
        applyEdges: setEdges,
        getNodes: getStableNodes,
        getEdges: getStableEdges,
      }}
    >
      <ShapeCheckOverlayProvider>
        <div
          ref={canvasWrapRef}
          className="cr-canvas-wrap"
          onPointerDownCapture={onCanvasPointerDownCapture}
          onClickCapture={onCanvasClickCapture}
          onWheelCapture={onCanvasWheelCapture}
        >
          <ConnectionSnapContext.Provider value={connectionSnapTarget}>
          <ReactFlow
        key={flowSurfaceKey}
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onConnectStart={onConnectStart}
        onConnectEnd={onConnectEnd}
        isValidConnection={isValidConnection}
        nodeTypes={researchNodeTypes}
        edgeTypes={researchEdgeTypes}
        defaultEdgeOptions={{ type: "research_default" }}
        connectionLineComponent={ResearchConnectionLine}
        connectionRadius={128}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onNodeDragStart={onNodeDragStart}
        onNodeDrag={onNodeDrag}
        onNodeDragStop={onNodeDragStop}
        autoPanOnNodeDrag={false}
        onPaneClick={onPaneClick}
        onSelectionChange={onSelectionChange}
        onPaneContextMenu={onPaneContextMenuForCombine}
        onNodeContextMenu={onNodeContextMenuForCombine}
        onSelectionContextMenu={onSelectionContextMenu}
        zoomOnDoubleClick={false}
        zoomOnScroll={false}
        zoomActivationKeyCode={null}
        minZoom={CANVAS_ZOOM_LEVELS[0]}
        maxZoom={CANVAS_ZOOM_LEVELS.at(-1)!}
        multiSelectionKeyCode="Shift"
        selectionKeyCode={["Meta", "Control"]}
        onlyRenderVisibleElements
        proOptions={{ hideAttribution: true }}
      >
        <FlowCanvasExtras addNodeImplRef={addNodeImplRef} setNodes={setNodes} />
        <SaveLoadBridge
          codeMode={codeMode}
          onCodeModeChange={onCodeModeChange}
          nodes={nodes}
          edges={edges}
          onSaveToServer={onSaveToServer}
          onLoadFromServer={onLoadFromServer}
          onSaveServerSucceeded={onSaveServerSucceeded}
          onSaveGraphToFile={onSaveGraphToFile}
          graphFileStemBase={graphFileStemBase}
          onPersistLibraryGraph={onPersistLibraryGraph}
          onGraphFileLoaded={onGraphFileLoaded}
          onGraphFileError={onGraphFileError}
          onOpenGraphCompare={onOpenGraphCompare}
          graphFileHandle={graphFileHandle}
          librarySource={librarySource}
          librarySaveDisplayName={librarySaveDisplayName}
          onSaveLibrarySourceEntry={onSaveLibrarySourceEntry}
          onSaveGraphToSourceFileSucceeded={onSaveGraphToSourceFileSucceeded}
          loading={loading}
          error={error}
          notice={notice}
          onAutoLayoutCanvas={onAutoLayoutCanvas}
          onAutoConnectCanvas={onAutoConnectCanvas}
          onClearCanvas={onClearCanvasClick}
        />
        <ViewportSync saved={savedViewport} applyId={viewportApplyNonce} />
        {/* Dot color follows the theme; classic --cr-canvas-dot is the exact
            legacy #3d3d47 (tokens.css). React Flow forwards `color` into an
            SVG pattern fill, which resolves var() at paint time. */}
        <Background variant={BackgroundVariant.Dots} gap={18} size={1.1} color="var(--cr-canvas-dot)" />
        <MiniMap pannable zoomable className="cr-minimap" />
        {/* <Panel position="bottom-left" className="cr-clear-canvas-panel nodrag nopan">
          <button
            type="button"
            className="cr-clear-canvas-btn"
            aria-label="Auto layout graph"
            title="Auto layout graph"
            onClick={onAutoLayoutCanvas}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" aria-hidden>
              <path
                fill="currentColor"
                d="M4 4h6v4H4V4zm10 0h6v4h-6V4zM4 16h6v4H4v-4zm10 0h6v4h-6v-4zM7 9h2v2h6V9h2v4H7V9zm5 5h2v2h-2v-2z"
              />
            </svg>
          </button>
          <button
            type="button"
            className="cr-clear-canvas-btn"
            aria-label="Auto-connect trainer wiring"
            title="Auto-connect trainer, dataset, model, optimizer, loss (incl. L1/L2 reg), and observable wires"
            onClick={onAutoConnectCanvas}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" aria-hidden>
              <path
                fill="currentColor"
                d="m7.1 13.1l2.8-2.8l1.4 1.4l-2.8 2.8a2.5 2.5 0 1 1-3.5-3.5l2.8-2.8l1.4 1.4l-2.8 2.8a.5.5 0 1 0 .7.7m9.8-9.9a2.5 2.5 0 0 1 3.5 3.5l-2.8 2.8l-1.4-1.4L19 5.3a.5.5 0 0 0-.7-.7l-2.8 2.8l-1.4-1.4zM8.9 17.5l6.6-6.6l1.4 1.4l-6.6 6.6zm2.8-7.7l1.4-1.4l1.4 1.4l-1.4 1.4z"
              />
            </svg>
          </button>
          <button
            type="button"
            className="cr-clear-canvas-btn"
            aria-label="Clear canvas"
            title="Clear canvas (remove all nodes and wires)"
            onClick={onClearCanvasClick}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" aria-hidden>
              <path
                fill="currentColor"
                d="M9 3h6a1 1 0 0 1 1 1v1h4v2H4V5h4V4a1 1 0 0 1 1-1zm-4 6h14l-1 12a2 2 0 0 1-2 1.9H8a2 2 0 0 1-2-1.9L5 9zm4 2v8h2v-8H9zm4 0v8h2v-8h-2z"
              />
            </svg>
          </button>
        </Panel> */}
        </ReactFlow>
          </ConnectionSnapContext.Provider>
      {combineMenu ? (
        <>
          <div
            role="presentation"
            className="cr-library-panel__ctx-backdrop"
            style={{ zIndex: 10022 }}
            onMouseDown={() => setCombineMenu(null)}
          />
          <div
            role="menu"
            className="cr-library-panel__ctx-menu"
            style={{ left: combineMenu.x, top: combineMenu.y, zIndex: 10023 }}
          >
            <button
              type="button"
              role="menuitem"
              className="cr-library-panel__ctx-item"
              onClick={() => {
                setCombineModal({ ids: combineMenu.ids });
                setCombineMenu(null);
              }}
            >
              Combine
            </button>
          </div>
        </>
      ) : null}
      {loopMenu ? (
        <>
          <div
            role="presentation"
            className="cr-library-panel__ctx-backdrop"
            style={{ zIndex: 10022 }}
            onMouseDown={() => setLoopMenu(null)}
          />
          <div
            role="menu"
            className="cr-library-panel__ctx-menu"
            style={{ left: loopMenu.x, top: loopMenu.y, zIndex: 10023 }}
          >
            <button
              type="button"
              role="menuitem"
              className="cr-library-panel__ctx-item"
              onClick={() => {
                setLoopModal({ nodeId: loopMenu.nodeId });
                setLoopMenu(null);
              }}
            >
              Loop
            </button>
            {loopMenu.nodeType === "combined_model" ? (
              <>
                <button
                  type="button"
                  role="menuitem"
                  className="cr-library-panel__ctx-item"
                  onClick={() => {
                    const n = nodes.find((x) => x.id === loopMenu.nodeId && String(x.type) === "combined_model");
                    const d = (n?.data ?? {}) as Record<string, unknown>;
                    const current =
                      (typeof d[INSTANCE_TITLE_KEY] === "string" && String(d[INSTANCE_TITLE_KEY]).trim()) ||
                      (typeof d.displayName === "string" && d.displayName.trim()) ||
                      "Combined model";
                    setRenameCombinedModal({ nodeId: loopMenu.nodeId, initialName: current });
                    setLoopMenu(null);
                  }}
                >
                  Rename
                </button>
                <button
                  type="button"
                  role="menuitem"
                  className="cr-library-panel__ctx-item"
                  onClick={() => decomposeCombinedModel(loopMenu.nodeId)}
                >
                  Decompose
                </button>
              </>
            ) : null}
          </div>
        </>
      ) : null}
      <CombineModelModal
        open={combineModal !== null}
        selectedCount={combineModal?.ids.length ?? 0}
        defaultName="Combined model"
        onCancel={cancelCombineModal}
        onConfirm={confirmCombine}
      />
      <NodeLoopModal
        open={loopModal !== null}
        initialCount={loopModalInitialCount}
        initialShareParams={loopModalInitialShareParams}
        onCancel={cancelLoopModal}
        onConfirm={confirmLoopCount}
      />
      <RenameCombinedModelModal
        open={renameCombinedModal !== null}
        initialName={renameCombinedModal?.initialName ?? "Combined model"}
        onCancel={cancelRenameCombinedModal}
        onConfirm={confirmRenameCombined}
      />
      <PasteNodesChoiceModal
        open={pasteChoiceOpen}
        onCancel={() => setPasteChoiceOpen(false)}
        onConfirm={finishPaste}
      />
      <ConfirmModal
        open={clearCanvasConfirmOpen}
        title="Clear canvas"
        message="Clear all nodes and wires from this canvas? This cannot be undone."
        onCancel={() => setClearCanvasConfirmOpen(false)}
        onConfirm={() => {
          setClearCanvasConfirmOpen(false);
          runClearCanvas();
        }}
      />
      <TargetCurveModal
        open={targetCurveModalOpen}
        sourceSteps={selectedTrainerData.stepTicks ?? []}
        sourceLoss={selectedTrainerData.lossHistory ?? []}
        initialSteps={selectedTrainerData.targetCurveStepTicks ?? []}
        initialLoss={selectedTrainerData.targetCurveLossHistory ?? []}
        onCancel={() => setTargetCurveModalOpen(false)}
        onSave={(steps, values) => {
          patchSelectedTrainerData({
            targetCurveStepTicks: steps,
            targetCurveLossHistory: values,
          });
          setTargetCurveModalOpen(false);
          setAutoTuneStatus(`target curve saved (${steps.length} points)`);
        }}
      />
      <AutoTuneConfigModal
        open={autoTuneModalOpen}
        suggestions={axisSuggestions}
        onCancel={() => setAutoTuneModalOpen(false)}
        onRun={(cfg) => {
          void runCoordinateDescent(cfg);
        }}
      />
      <AutoTuneResultsModal
        open={autoTuneResultsOpen}
        result={autoTuneResults}
        onClose={() => setAutoTuneResultsOpen(false)}
      />
      <CurveStarerModal
        open={curveStarerOpen}
        busy={curveStarerBusy}
        progress={curveStarerProgress}
        statusMessage={curveStarerStatus}
        entries={curveStarerEntries}
        rankBy={curveStarerRankBy}
        onRankByChange={setCurveStarerRankBy}
        targetConfig={curveStarerTargetConfig}
        targetCurveOptions={curveStarerTargetOptions}
        onTargetConfigChange={setCurveStarerTargetConfig}
        nodes={nodes}
        edges={edges}
        onStartAnalyze={() => void startCurveStarerAnalysis()}
        onClose={() => setCurveStarerOpen(false)}
      />
      <AddNodeSearchModal
        open={addNodeSearchScreenPos !== null}
        placeAt={addNodeSearchScreenPos}
        onClose={() => setAddNodeSearchScreenPos(null)}
        onAdd={(nodeType, screenPos, options) => {
          addNodeImplRef.current(nodeType, screenPos, options);
        }}
      />
        </div>
      </ShapeCheckOverlayProvider>
    </FlowSurfaceProvider>
  );
}

type SaveLoadBridgeProps = {
  codeMode: boolean;
  onCodeModeChange: (next: boolean) => void;
  nodes: Node[];
  edges: Edge[];
  onSaveToServer: (doc: GraphDocument) => Promise<void>;
  onLoadFromServer: () => Promise<void>;
  onSaveServerSucceeded: () => void;
  onSaveGraphToFile: (doc: GraphDocument, fileStem: string) => Promise<void>;
  graphFileStemBase: string;
  onPersistLibraryGraph: (
    kind: SavedGraphKind,
    doc: GraphDocument,
    tier: GraphFileExportTier,
  ) => void;
  onGraphFileLoaded: (doc: GraphDocument, fileHandle?: FileSystemFileHandle | null) => void;
  onGraphFileError: (message: string) => void;
  onAutoLayoutCanvas: () => void;
  onAutoConnectCanvas: () => void;
  onClearCanvas: () => void;
  onOpenGraphCompare: (doc: GraphDocument) => void;
  graphFileHandle: FileSystemFileHandle | null;
  librarySource: GraphCanvas["librarySource"] | undefined;
  librarySaveDisplayName: string;
  onSaveLibrarySourceEntry: (kind: "workflow" | "template", entry: SavedGraphEntry) => Promise<void>;
  onSaveGraphToSourceFileSucceeded: () => void;
  loading: boolean;
  error: string | null;
  notice: string | null;
};

function SaveLoadBridge({
  codeMode,
  onCodeModeChange,
  nodes,
  edges,
  onSaveToServer,
  onLoadFromServer,
  onSaveServerSucceeded,
  onSaveGraphToFile,
  graphFileStemBase,
  onPersistLibraryGraph,
  onGraphFileLoaded,
  onGraphFileError,
  onAutoLayoutCanvas,
  onAutoConnectCanvas,
  onClearCanvas,
  onOpenGraphCompare,
  graphFileHandle,
  librarySource,
  librarySaveDisplayName,
  onSaveLibrarySourceEntry,
  onSaveGraphToSourceFileSucceeded,
  loading,
  error,
  notice,
}: SaveLoadBridgeProps) {
  const { getViewport } = useReactFlow();

  const doSave = useCallback(async () => {
    const vp = getViewport();
    await onSaveToServer(toApiDocument(nodes, edges, vp));
    onSaveServerSucceeded();
  }, [edges, getViewport, nodes, onSaveServerSucceeded, onSaveToServer]);

  const saveToFileByTier = useCallback(
    (tier: GraphFileExportTier) => {
      const vp = getViewport();
      const raw = toApiDocument(nodes, edges, vp);
      const doc = applyGraphFileExportTier(raw, tier);
      const stem =
        tier === "large"
          ? graphFileStemBase
          : tier === "medium"
            ? `${graphFileStemBase}-medium`
            : `${graphFileStemBase}-small`;
      return onSaveGraphToFile(doc, stem);
    },
    [edges, getViewport, graphFileStemBase, nodes, onSaveGraphToFile],
  );

  const saveAsLibrary = useCallback(
    (kind: SavedGraphKind, tier: GraphFileExportTier) => {
      const vp = getViewport();
      const raw = toApiDocument(nodes, edges, vp);
      const doc = applyGraphFileExportTier(raw, tier);
      onPersistLibraryGraph(kind, doc, tier);
    },
    [edges, getViewport, nodes, onPersistLibraryGraph],
  );

  const openGraphCompare = useCallback(() => {
    const vp = getViewport();
    onOpenGraphCompare(toApiDocument(nodes, edges, vp));
  }, [edges, getViewport, nodes, onOpenGraphCompare]);

  const exportCanvasPng = useCallback(async () => {
    try {
      const stem = safeGraphFilenameStem(graphFileStemBase);
      const blob = await captureFlowRendererToPngBlob();
      await saveBlobWithUserLocation(blob, `${stem}-canvas.png`, "PNG image", {
        "image/png": [".png"],
      });
    } catch (e) {
      onGraphFileError(e instanceof Error ? e.message : String(e));
    }
  }, [graphFileStemBase, onGraphFileError]);

  const exportCanvasPdf = useCallback(async () => {
    try {
      const stem = safeGraphFilenameStem(graphFileStemBase);
      const png = await captureFlowRendererToPngBlob();
      const pdfBlob = await pngBlobToSinglePagePdfBlob(png);
      await saveBlobWithUserLocation(pdfBlob, `${stem}-canvas.pdf`, "PDF document", {
        "application/pdf": [".pdf"],
      });
    } catch (e) {
      onGraphFileError(e instanceof Error ? e.message : String(e));
    }
  }, [graphFileStemBase, onGraphFileError]);

  const saveGraphToSourceFile = useCallback(async () => {
    if (graphFileHandle) {
      const vp = getViewport();
      const raw = toApiDocument(nodes, edges, vp);
      const doc = applyGraphFileExportTier(raw, "small");
      try {
        await writeGraphDocumentToFileHandle(graphFileHandle, doc);
        onSaveGraphToSourceFileSucceeded();
      } catch (e) {
        onGraphFileError(e instanceof Error ? e.message : String(e));
      }
      return;
    }
    if (librarySource) {
      const vp = getViewport();
      const raw = toApiDocument(nodes, edges, vp);
      const doc = applyGraphFileExportTier(raw, librarySource.tier);
      const label = librarySaveDisplayName.trim() || "Untitled";
      const entry: SavedGraphEntry = {
        id: librarySource.entryId,
        name: label.slice(0, 200),
        tier: librarySource.tier,
        document: doc,
        savedAt: Date.now(),
      };
      try {
        await onSaveLibrarySourceEntry(librarySource.kind, entry);
        onSaveGraphToSourceFileSucceeded();
      } catch (e) {
        onGraphFileError(e instanceof Error ? e.message : String(e));
      }
    }
  }, [
    edges,
    getViewport,
    graphFileHandle,
    librarySaveDisplayName,
    librarySource,
    nodes,
    onGraphFileError,
    onSaveGraphToSourceFileSucceeded,
    onSaveLibrarySourceEntry,
  ]);

  return (
    <GraphToolbar
      onSaveToServer={doSave}
      onLoadFromServer={onLoadFromServer}
      onSaveGraphToSourceFile={saveGraphToSourceFile}
      canSaveGraphToSourceFile={graphFileHandle != null || librarySource != null}
      onSaveGraphToFileTier={saveToFileByTier}
      onSaveGraphAsLibrary={saveAsLibrary}
      onOpenGraphCompare={openGraphCompare}
      onExportCanvasPng={exportCanvasPng}
      onExportCanvasPdf={exportCanvasPdf}
      onGraphFileLoaded={onGraphFileLoaded}
      onGraphFileError={onGraphFileError}
      onAutoLayoutCanvas={onAutoLayoutCanvas}
      onAutoConnectCanvas={onAutoConnectCanvas}
      onClearCanvas={onClearCanvas}
      loading={loading}
      error={error}
      notice={notice}
    />
  );
}

const WORKBENCH_RAIL_PANEL_LS_KEY = "comfyresearch.workbenchRailPanelWidthPx";
const WORKBENCH_RAIL_PANEL_MIN = 200;
const WORKBENCH_RAIL_PANEL_GRIP_PX = 6;
/** Match `.cr-rail` width in `index.css`. */
const WORKBENCH_RAIL_LEFT_NAV_PX = 68;
/** Minimum space left for the graph column (canvas + optional code/chat). */
const WORKBENCH_RAIL_PANEL_VIEWPORT_RESERVE = 24;

function clampWorkbenchRailPanelWidthPx(px: number, viewportW: number): number {
  const max = Math.max(
    WORKBENCH_RAIL_PANEL_MIN + 24,
    viewportW -
      WORKBENCH_RAIL_LEFT_NAV_PX -
      WORKBENCH_RAIL_PANEL_GRIP_PX -
      WORKBENCH_RAIL_PANEL_VIEWPORT_RESERVE,
  );
  return Math.min(max, Math.max(WORKBENCH_RAIL_PANEL_MIN, Math.round(px)));
}

function readStoredWorkbenchRailPanelWidthPx(viewportW: number): number | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(WORKBENCH_RAIL_PANEL_LS_KEY);
    if (raw == null || raw === "") return null;
    const n = Number(raw);
    if (!Number.isFinite(n)) return null;
    return clampWorkbenchRailPanelWidthPx(n, viewportW);
  } catch {
    return null;
  }
}

function defaultWorkbenchRailPanelWidthPx(viewportW: number): number {
  return clampWorkbenchRailPanelWidthPx(Math.min(288, Math.round(viewportW * 0.34)), viewportW);
}

export function ResearchCanvas() {
  const addNodeImplRef = useRef<AddNodeFromLibrary>(() => {});
  const initialWorkspaceProjectRef = useRef<WorkspaceProject | null>(null);
  if (!initialWorkspaceProjectRef.current) {
    initialWorkspaceProjectRef.current = createEmptyWorkspaceProject(newProjectId());
  }
  const initialWorkspaceProject = initialWorkspaceProjectRef.current;
  const initialProjectId = initialWorkspaceProject.id;

  const [projects, setProjects] = useState<WorkspaceProject[]>(() => [initialWorkspaceProject]);
  const [activeProjectId, setActiveProjectId] = useState(initialProjectId);
  const [projectClosePending, setProjectClosePending] = useState<{
    projectId: string;
    title: string;
    summary: ProjectTrainSummary;
  } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [railSection, setRailSection] = useState<RailPrimarySection | null>("nodes");
  const [canvasSelectedNodeId, setCanvasSelectedNodeId] = useState<string | null>(null);
  const [nodeInformationOpen, setNodeInformationOpen] = useState(false);
  const [nodeInformationContent, setNodeInformationContent] = useState<Pick<OpenNodeInformationDetail, "title" | "text" | "code" | "mode"> | null>(null);
  const [workbenchRailPanelWidthPx, setWorkbenchRailPanelWidthPx] = useState(() => {
    if (typeof window === "undefined") return 288;
    const w = window.innerWidth;
    return readStoredWorkbenchRailPanelWidthPx(w) ?? defaultWorkbenchRailPanelWidthPx(w);
  });
  const workbenchRailPanelWidthPxRef = useRef(workbenchRailPanelWidthPx);
  workbenchRailPanelWidthPxRef.current = workbenchRailPanelWidthPx;
  const [railPanelGripDragging, setRailPanelGripDragging] = useState(false);
  const railPanelGripDragRef = useRef<{ startX: number; startWidth: number } | null>(null);
  const handleRailSelect = useCallback((section: RailPrimarySection) => {
    setRailSection(section);
  }, []);

  const onCanvasSelectionChange = useCallback((nodeId: string | null, nodeType?: string | null) => {
    setCanvasSelectedNodeId(nodeId);
    if (nodeId && isObservableModelNodeType(nodeType)) {
      setRailSection("observables");
    }
  }, []);
  useEffect(() => {
    const handleOpenNodeInformation = (event: Event) => {
      const detail = (event as CustomEvent<OpenNodeInformationDetail>).detail;
      if (!detail?.nodeId) return;
      setCanvasSelectedNodeId(detail.nodeId);
      setNodeInformationContent({ title: detail.title, text: detail.text, code: detail.code, mode: detail.mode });
      setNodeInformationOpen(true);
    };
    window.addEventListener(OPEN_NODE_INFORMATION_EVENT, handleOpenNodeInformation);
    return () => window.removeEventListener(OPEN_NODE_INFORMATION_EVENT, handleOpenNodeInformation);
  }, []);
  const [templateList, setTemplateList] = useState<SavedGraphEntry[]>([]);
  const bumpTemplateLibraryListeners = useCallback(() => {
    window.dispatchEvent(new Event(GRAPH_TEMPLATE_LIBRARY_CHANGED));
  }, []);
  const [railDeletingMessage, setRailDeletingMessage] = useState<string | null>(null);
  const railDeleteDepthRef = useRef(0);
  const withRailDeleteOverlay = useCallback(async (message: string, fn: () => Promise<void>) => {
    railDeleteDepthRef.current += 1;
    if (railDeleteDepthRef.current === 1) setRailDeletingMessage(message);
    try {
      await fn();
    } finally {
      railDeleteDepthRef.current -= 1;
      if (railDeleteDepthRef.current <= 0) {
        railDeleteDepthRef.current = 0;
        setRailDeletingMessage(null);
      }
    }
  }, []);

  const [librarySaveDraft, setLibrarySaveDraft] = useState<LibrarySaveDraft | null>(null);
  const libraryDraftRef = useRef<LibrarySaveDraft | null>(null);
  const [graphCompareOpen, setGraphCompareOpen] = useState(false);
  const [compareSourceDoc, setCompareSourceDoc] = useState<GraphDocument | null>(null);
  const { loadWorkspace, saveWorkspace, loading, error } = useWorkspaceApi();

  const projectsRef = useRef(projects);
  const activeProjectIdRef = useRef(activeProjectId);
  const activeCanvasIdRef = useRef("");
  /** Latest canvas nodes for notebook codegen (e.g. trainer runner \`fn_*\` names match \`pySlugForNode\`). */
  const activeCanvasNodesRef = useRef<Node[]>([]);
  const templateDeepLinkHandledRef = useRef<string | null>(null);
  projectsRef.current = projects;
  activeProjectIdRef.current = activeProjectId;

  useEffect(() => {
    let cancelled = false;
    void loadWorkspace()
      .then((snap) => {
        if (cancelled) return;
        const next = workspaceSnapshotToProjects(snap);
        const preferredActive = activeProjectIdRef.current;
        const merged = mergeWorkspaceHydrateWithLocalProjects(
          next,
          projectsRef.current,
          initialWorkspaceProject,
        );
        setProjects(merged);
        setActiveProjectId(
          merged.some((p) => p.id === preferredActive) ? preferredActive : snap.active_project_id,
        );
      })
      .catch(() => {
        if (cancelled) return;
        /* Keep the local workspace when the server snapshot is unavailable. */
      });
    return () => {
      cancelled = true;
    };
  }, [loadWorkspace]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        await migrateLegacyLocalStorageToServer();
        const t = await fetchSavedGraphLibrary("template");
        if (!cancelled) {
          setTemplateList(t);
          bumpTemplateLibraryListeners();
        }
      } catch (e) {
        if (!cancelled) {
          setNotice(e instanceof Error ? e.message : "Could not load templates from the server.");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [bumpTemplateLibraryListeners]);

  const activeProject = useMemo(
    () => projects.find((p) => p.id === activeProjectId) ?? projects[0],
    [projects, activeProjectId],
  );

  const activeCanvas = useMemo(
    () => projects.find((x) => x.id === activeProjectId)?.canvas,
    [projects, activeProjectId],
  );

  const nodeForInformation = nodeInformationOpen && canvasSelectedNodeId && activeCanvas
    ? activeCanvas.nodes.find((node) => node.id === canvasSelectedNodeId) ?? null
    : null;

  const trainByProjectId = useMemo(() => {
    const out: Record<string, ProjectTrainSummary> = {};
    for (const p of projects) {
      out[p.id] = summarizeProjectTrainActivity(p.canvas);
    }
    return out;
  }, [projects]);

  const [codeMode, setCodeMode] = useState(false);
  const onWorkbenchCodeModeChange = useCallback((next: boolean) => {
    setCodeMode(next);
  }, []);
  useEffect(() => {
    const onResize = () => {
      const vw = window.innerWidth;
      setWorkbenchRailPanelWidthPx((prev) => clampWorkbenchRailPanelWidthPx(prev, vw));
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    if (!railPanelGripDragging) return;
    const prevSelect = document.body.style.userSelect;
    const prevCursor = document.body.style.cursor;
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    return () => {
      document.body.style.userSelect = prevSelect;
      document.body.style.cursor = prevCursor;
    };
  }, [railPanelGripDragging]);

  const onRailPanelGripPointerDown = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    railPanelGripDragRef.current = { startX: e.clientX, startWidth: workbenchRailPanelWidthPxRef.current };
    e.currentTarget.setPointerCapture(e.pointerId);
    setRailPanelGripDragging(true);
  }, []);

  const onRailPanelGripPointerMove = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const drag = railPanelGripDragRef.current;
    if (!drag) return;
    const next = clampWorkbenchRailPanelWidthPx(
      drag.startWidth + (e.clientX - drag.startX),
      window.innerWidth,
    );
    workbenchRailPanelWidthPxRef.current = next;
    setWorkbenchRailPanelWidthPx(next);
  }, []);

  const onRailPanelGripPointerUp = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    railPanelGripDragRef.current = null;
    setRailPanelGripDragging(false);
    try {
      window.localStorage.setItem(WORKBENCH_RAIL_PANEL_LS_KEY, String(workbenchRailPanelWidthPxRef.current));
    } catch {
      /* ignore */
    }
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  }, []);

  const onRailPanelGripKeyDown = useCallback((e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const delta = e.key === "ArrowLeft" ? -16 : 16;
    setWorkbenchRailPanelWidthPx((prev) => {
      const next = clampWorkbenchRailPanelWidthPx(prev + delta, window.innerWidth);
      workbenchRailPanelWidthPxRef.current = next;
      try {
        window.localStorage.setItem(WORKBENCH_RAIL_PANEL_LS_KEY, String(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  const [codeNotebooks, setCodeNotebooks] = useState<Record<string, CodeNotebookCell[]>>({});
  const [notebookFocusRequest, setNotebookFocusRequest] = useState<{ cellId: string; nonce: number } | null>(null);
  const notebookCanvasIdRef = useRef<string | null>(null);
  const prevCodeModeForNotebookRef = useRef(false);
  const prevNodeIdsForNotebookRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!activeCanvas) return;
    const cid = activeCanvas.id;
    const canvasSwitched = notebookCanvasIdRef.current !== cid;
    if (canvasSwitched) {
      notebookCanvasIdRef.current = cid;
    }
    const codeModeTurnedOn = codeMode && !prevCodeModeForNotebookRef.current;
    prevCodeModeForNotebookRef.current = codeMode;

    if (canvasSwitched || codeModeTurnedOn) {
      prevNodeIdsForNotebookRef.current = new Set(activeCanvas.nodes.map((n) => n.id));
      // Entering Code view (or switching canvas while Code is on) used to only sync node ids and return,
      // so templates like grokking never got cells — cells were only appended when *new* nodes were added.
      if (codeMode) {
        setCodeNotebooks((m) => {
          const existing = m[cid] ?? [];
          // Train-from-Graph appends only `trainer_runner` cells; opening Code must still seed definitions.
          const runnerTail = existing.filter((c) => c.nodeType === "trainer_runner");
          const hasNonRunnerCells = existing.some((c) => c.nodeType !== "trainer_runner");
          if (hasNonRunnerCells) return m;
          const nodes = activeCanvas.nodes;
          const edges = activeCanvas.edges;
          const nextCells: CodeNotebookCell[] = [];
          for (const n of nodes) {
            if (n.type === "graph_assist_failure_overlay" || n.parentId) continue;
            const nt = String(n.type ?? "unknown");
            if (shouldOmitNotebookCell(nt)) continue;
            nextCells.push({
              id: `nb-${newProjectId()}`,
              nodeId: n.id,
              nodeType: nt,
              source: buildNodeDefinitionPython(n, { nodes, edges }),
            });
          }
          if (nextCells.length === 0 && runnerTail.length === 0) return m;
          return { ...m, [cid]: [...nextCells, ...runnerTail] };
        });
      }
      return;
    }

    if (!codeMode) {
      prevNodeIdsForNotebookRef.current = new Set(activeCanvas.nodes.map((n) => n.id));
      return;
    }

    const prev = prevNodeIdsForNotebookRef.current;
    const added = activeCanvas.nodes.filter((n) => !prev.has(n.id));
    const meaningful = added.filter(
      (n) => n.type !== "graph_assist_failure_overlay" && !n.parentId,
    );
    // Loading a graph or pasting a large subgraph replaces many ids at once — skip auto-cells to avoid a huge notebook.
    const chunk = meaningful.length > 0 && meaningful.length <= 8;
    if (chunk) {
      setCodeNotebooks((m) => {
        const list = m[cid] ?? [];
        const nextCells = [...list];
        for (const n of meaningful) {
          const nt = String(n.type ?? "unknown");
          if (shouldOmitNotebookCell(nt)) continue;
          nextCells.push({
            id: `nb-${newProjectId()}`,
            nodeId: n.id,
            nodeType: nt,
            source: buildNodeDefinitionPython(n, {
              nodes: activeCanvas.nodes,
              edges: activeCanvas.edges,
            }),
          });
        }
        return { ...m, [cid]: nextCells };
      });
    }
    prevNodeIdsForNotebookRef.current = new Set(activeCanvas.nodes.map((n) => n.id));
  }, [activeCanvas, codeMode]);

  useEffect(() => {
    if (!codeMode || !activeCanvas) return;
    const cid = activeCanvas.id;
    const trainerById = new Map(
      activeCanvas.nodes
        .filter((n) => n.type === "trainer" || n.type === "crl_trainer")
        .map((n) => [n.id, n] as const),
    );
    if (!trainerById.size) return;
    setCodeNotebooks((m) => {
      const list = m[cid];
      if (!list || list.length === 0) return m;
      let changed = false;
      const next = list.map((cell) => {
        if (cell.nodeType !== "trainer" && cell.nodeType !== "crl_trainer") return cell;
        const trainer = trainerById.get(cell.nodeId);
        if (!trainer) return cell;
        const regenerated = buildNodeDefinitionPython(trainer, {
          nodes: activeCanvas.nodes,
          edges: activeCanvas.edges,
        });
        if (regenerated === cell.source) return cell;
        changed = true;
        return { ...cell, source: regenerated };
      });
      if (!changed) return m;
      return { ...m, [cid]: next };
    });
  }, [activeCanvas, codeMode]);

  const onTrainerTrainPressed = useCallback((trainerNodeId: string) => {
    const snippet = buildTrainerRunnerPython(trainerNodeId, activeCanvasNodesRef.current);
    const runnerCellId = `runner-${trainerNodeId}`;
    setCodeNotebooks((m) => {
      const cid = activeCanvasIdRef.current;
      if (!cid) return m;
      const prev = m[cid] ?? [];
      const list = [...prev];
      const existingIdx = list.findIndex((c) => c.id === runnerCellId);
      const cell: CodeNotebookCell = {
        id: runnerCellId,
        nodeId: trainerNodeId,
        nodeType: "trainer_runner",
        source: snippet,
      };
      if (existingIdx >= 0) {
        list.splice(existingIdx, 1);
      }
      // Append at notebook end: cell order follows node creation order, so the trainer definition cell
      // can sit before optimizer / loss cells added later — inserting after trainer would wrongly place
      // this cell above those dependencies.
      list.push(cell);
      return { ...m, [cid]: list };
    });
  }, []);

  const openDatasetSpecInNotebook = useCallback((nodeId: string, graphNodeType: string, specPython: string) => {
    setCodeMode(true);
    const cid = activeCanvasIdRef.current;
    if (!cid) return;
    const cellId = `dataset-spec-${nodeId}`;
    setCodeNotebooks((m) => {
      const prev = m[cid] ?? [];
      const list = [...prev];
      const idx = list.findIndex((c) => c.id === cellId);
      const cell: CodeNotebookCell = {
        id: cellId,
        nodeId,
        nodeType: graphNodeType,
        source: specPython,
      };
      if (idx >= 0) {
        list[idx] = cell;
      } else {
        list.push(cell);
      }
      return { ...m, [cid]: list };
    });
    setNotebookFocusRequest((prev) => ({
      cellId,
      nonce: (prev?.nonce ?? 0) + 1,
    }));
  }, []);

  const openModelSpecInNotebook = useCallback((nodeId: string, graphNodeType: string, specPython: string) => {
    setCodeMode(true);
    const cid = activeCanvasIdRef.current;
    if (!cid) return;
    const cellId = `model-spec-${nodeId}`;
    setCodeNotebooks((m) => {
      const prev = m[cid] ?? [];
      const list = [...prev];
      const idx = list.findIndex((c) => c.id === cellId);
      const cell: CodeNotebookCell = {
        id: cellId,
        nodeId,
        nodeType: graphNodeType,
        source: specPython,
      };
      if (idx >= 0) {
        list[idx] = cell;
      } else {
        list.push(cell);
      }
      return { ...m, [cid]: list };
    });
    setNotebookFocusRequest((prev) => ({
      cellId,
      nonce: (prev?.nonce ?? 0) + 1,
    }));
  }, []);

  const openOptimizerSpecInNotebook = useCallback((nodeId: string, graphNodeType: string, specPython: string) => {
    setCodeMode(true);
    const cid = activeCanvasIdRef.current;
    if (!cid) return;
    const cellId = `optimizer-spec-${nodeId}`;
    setCodeNotebooks((m) => {
      const prev = m[cid] ?? [];
      const list = [...prev];
      const idx = list.findIndex((c) => c.id === cellId);
      const cell: CodeNotebookCell = {
        id: cellId,
        nodeId,
        nodeType: graphNodeType,
        source: specPython,
      };
      if (idx >= 0) {
        list[idx] = cell;
      } else {
        list.push(cell);
      }
      return { ...m, [cid]: list };
    });
    setNotebookFocusRequest((prev) => ({
      cellId,
      nonce: (prev?.nonce ?? 0) + 1,
    }));
  }, []);

  activeCanvasIdRef.current = activeCanvas?.id ?? "";
  activeCanvasNodesRef.current = activeCanvas?.nodes ?? [];

  const graphCompareTargets = useMemo((): GraphCompareTarget[] => {
    if (!activeCanvas) return [];
    const out: GraphCompareTarget[] = [];
    for (const p of projects) {
      if (p.id === activeProjectId) continue;
      const c = p.canvas;
      out.push({
        key: p.id,
        projectTitle: p.title,
        canvasTitle: c.title,
        document: toApiDocument(c.nodes, c.edges, c.savedViewport ?? null),
      });
    }
    return out;
  }, [projects, activeProjectId, activeCanvas]);

  const onNodesChange: OnNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setProjects((list) =>
        mapProjectCanvas(list, activeProjectId, (c) => {
          const augmented = augmentNodeRemovesWithAttentionLowExpansion(
            augmentNodeRemovesWithMlpLowExpansion(changes, c.nodes),
            c.nodes,
          );
          const nextNodes = applyNodeChanges(augmented, c.nodes);
          return { ...c, nodes: nextNodes, dirty: true };
        }),
      );
    },
    [activeProjectId],
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setProjects((list) =>
        mapProjectCanvas(list, activeProjectId, (c) => ({
          ...c,
          edges: normalizeOptimizerLrScheduleEdgeTargets(applyEdgeChanges(changes, c.edges), c.nodes),
          dirty: true,
        })),
      );
    },
    [activeProjectId],
  );

  const setNodesForCanvas = useCallback((projectId: string, updater: SetStateAction<Node[]>) => {
    setProjects((list) =>
      mapProjectCanvas(list, projectId, (c) => {
        const nextNodes =
          typeof updater === "function" ? (updater as (n: Node[]) => Node[])(c.nodes) : updater;
        if (nextNodes === c.nodes) return c;
        return { ...c, nodes: nextNodes, dirty: true };
      }),
    );
  }, []);

  const setEdgesForCanvas = useCallback((projectId: string, updater: SetStateAction<Edge[]>) => {
    setProjects((list) =>
      mapProjectCanvas(list, projectId, (c) => {
        const nextEdges =
          typeof updater === "function" ? (updater as (e: Edge[]) => Edge[])(c.edges) : updater;
        if (nextEdges === c.edges) return c;
        return { ...c, edges: nextEdges, dirty: true };
      }),
    );
  }, []);

  const addProject = useCallback(() => {
    const id = newProjectId();
    setProjects((list) => [...list, createEmptyWorkspaceProject(id)]);
    setActiveProjectId(id);
    setNotice(null);
  }, []);

  const finalizeCloseProject = useCallback(
    async (id: string) => {
      const p = projects.find((x) => x.id === id);
      if (p) {
        await abortProjectTraining(summarizeProjectTrainActivity(p.canvas).trainers);
      }
      setNotice(null);
      setProjects((list) => {
        if (list.length <= 1) return list;
        const filtered = list.filter((proj) => proj.id !== id);
        setActiveProjectId((cur) => (cur === id ? filtered[0]!.id : cur));
        return filtered;
      });
      setProjectClosePending(null);
    },
    [projects],
  );

  const requestCloseProject = useCallback(
    (id: string, e: ReactMouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      if (projects.length <= 1) return;
      const p = projects.find((x) => x.id === id);
      if (!p) return;
      const summary = summarizeProjectTrainActivity(p.canvas);
      if (summary.hasActiveTraining) {
        setProjectClosePending({ projectId: id, title: p.title, summary });
        return;
      }
      void finalizeCloseProject(id);
    },
    [finalizeCloseProject, projects],
  );

  const closeProject = requestCloseProject;

  const clearActiveCanvas = useCallback(() => {
    let clearedCanvasId: string | null = null;
    setProjects((list) =>
      mapProjectCanvas(list, activeProjectId, (c) => {
        clearedCanvasId = c.id;
        return {
          ...c,
          nodes: [],
          edges: [],
          savedViewport: null,
          viewportApplyNonce: c.viewportApplyNonce + 1,
          dirty: true,
          localGraphFileHandle: undefined,
          librarySource: undefined,
        };
      }),
    );
    if (clearedCanvasId) {
      setCodeNotebooks((m) => {
        if (!m[clearedCanvasId!]) return m;
        return { ...m, [clearedCanvasId!]: [] };
      });
    }
    setNotice(null);
  }, [activeProjectId]);

  const applyDocToActiveCanvas = useCallback(
    (doc: GraphDocument, fileHandle?: FileSystemFileHandle | null) => {
      const { nodes, edges } = sanitizeLoadedGraph(doc);
      setProjects((list) =>
        mapProjectCanvas(list, activeProjectId, (c) => {
          return {
            ...c,
            nodes,
            edges,
            savedViewport: doc.viewport ?? null,
            viewportApplyNonce: c.viewportApplyNonce + 1,
            dirty: false,
            ...(fileHandle === undefined
              ? {}
              : fileHandle === null
                ? { localGraphFileHandle: undefined, librarySource: undefined }
                : {
                    localGraphFileHandle: fileHandle,
                    librarySource: undefined,
                  }),
          };
        }),
      );
      setNotice(null);
    },
    [activeProjectId],
  );

  const onSaveToServer = useCallback(
    async (doc: GraphDocument) => {
      const body = buildWorkspaceSnapshotDTO(
        projectsRef.current,
        activeProjectIdRef.current,
        { projectId: activeProjectIdRef.current, document: doc },
      );
      await saveWorkspace(body);
    },
    [saveWorkspace],
  );

  const onLoadFromServer = useCallback(async () => {
    const snap = await loadWorkspace();
    const next = workspaceSnapshotToProjects(snap);
    setProjects(next);
    setActiveProjectId(snap.active_project_id);
    setNotice(null);
  }, [loadWorkspace]);

  const onSaveGraphToFile = useCallback(
    async (doc: GraphDocument, fileStem: string) => {
      const { outcome, handle } = await saveGraphJsonWithUserLocation(doc, fileStem);
      if (outcome === "cancelled") return;
      setProjects((list) =>
        mapProjectCanvas(list, activeProjectId, (c) => ({
          ...c,
          dirty: false,
          ...(handle ? { localGraphFileHandle: handle, librarySource: undefined } : {}),
        })),
      );
      setNotice(null);
    },
    [activeProjectId],
  );

  const onSaveGraphToSourceFileSucceeded = useCallback(() => {
    setProjects((list) =>
      mapProjectCanvas(list, activeProjectIdRef.current, (c) => ({ ...c, dirty: false })),
    );
    setNotice(null);
  }, []);

  const onSaveGraphToLibraryEntry = useCallback(
    async (kind: SavedGraphKind, entry: SavedGraphEntry) => {
      const next = await addSavedGraphEntry(kind, entry);
      if (kind === "workflow") {
        if (entry.libraryOrigin === "combined_model") {
          window.dispatchEvent(new Event(GRAPH_COMBINED_MODEL_LIBRARY_CHANGED));
        }
      } else if (kind === "template") {
        setTemplateList(next);
        bumpTemplateLibraryListeners();
      }
    },
    [bumpTemplateLibraryListeners],
  );

  const onGraphFileLoaded = useCallback(
    (doc: GraphDocument, fileHandle?: FileSystemFileHandle | null) => {
      applyDocToActiveCanvas(doc, fileHandle);
    },
    [applyDocToActiveCanvas],
  );

  const onGraphFileError = useCallback((message: string) => {
    setNotice(message);
  }, []);

  const onSaveServerSucceeded = useCallback(() => {
    setProjects((list) =>
      mapProjectCanvas(list, activeProjectIdRef.current, (c) => ({ ...c, dirty: false })),
    );
  }, []);

  const dismissLibraryModal = useCallback(() => {
    libraryDraftRef.current = null;
    setLibrarySaveDraft(null);
  }, []);

  const confirmLibrarySave = useCallback(async (trimmedName: string) => {
    const d = libraryDraftRef.current;
    if (!d) return;
    const defaultName = d.kind === "workflow" ? "Untitled workflow" : "Untitled template";
    const finalName = trimmedName.trim() || defaultName;
    const entry: SavedGraphEntry = {
      id:
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `sg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      name: finalName,
      tier: d.tier,
      document: d.document,
      savedAt: Date.now(),
    };
    try {
      const next = await addSavedGraphEntry(d.kind, entry);
      libraryDraftRef.current = null;
      setLibrarySaveDraft(null);
      if (d.kind === "template") {
        setTemplateList(next);
        bumpTemplateLibraryListeners();
      }
      setNotice(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setNotice(`Could not save to server: ${msg}`);
    }
  }, [bumpTemplateLibraryListeners]);

  const onPersistLibraryGraph = useCallback(
    (kind: SavedGraphKind, doc: GraphDocument, tier: GraphFileExportTier) => {
      const draft: LibrarySaveDraft = { kind, document: doc, tier };
      libraryDraftRef.current = draft;
      setLibrarySaveDraft(draft);
    },
    [],
  );

  const openSavedGraphInNewProject = useCallback(
    (entry: SavedGraphEntry, kind: "template") => {
      const existing = projectsRef.current.find(
        (p) =>
          p.canvas.librarySource?.kind === "template" && p.canvas.librarySource.entryId === entry.id,
      );
      if (existing) {
        setActiveProjectId(existing.id);
        setNotice(null);
        return;
      }
      const id = newProjectId();
      const canvasId = newProjectId();
      const { nodes, edges } = sanitizeLoadedGraph(entry.document);
      setProjects((list) => [
        ...list,
        {
          id,
          title: entry.name.slice(0, 48) || formatProjectTabTitle(id),
          canvas: {
            id: canvasId,
            title: entry.name.slice(0, 40) || formatCanvasTitle(canvasId),
            nodes,
            edges,
            savedViewport: entry.document.viewport ?? null,
            viewportApplyNonce: 1,
            dirty: false,
            librarySource: {
              kind,
              entryId: entry.id,
              tier: entry.tier,
            },
          },
        },
      ]);
      setActiveProjectId(id);
      setNotice(null);
    },
    [],
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const rail = params.get("rail");
    if (rail === "templates") {
      setRailSection("templates");
    }
    const templateId = params.get("templateId") ?? params.get("template_id");
    if (!templateId) return;
    if (templateDeepLinkHandledRef.current === templateId) return;
    if (templateList.length === 0) return;

    const entry = templateList.find((item) => item.id === templateId);
    templateDeepLinkHandledRef.current = templateId;
    setRailSection("templates");
    if (!entry) {
      setNotice(`Template link not found: ${templateId}`);
      return;
    }

    openSavedGraphInNewProject(entry, "template");
    setNotice(`Opened template: ${entry.name}`);
    const t = window.setTimeout(() => {
      const url = new URL(window.location.href);
      url.searchParams.delete("templateId");
      url.searchParams.delete("template_id");
      window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
    }, 0);
    return () => window.clearTimeout(t);
  }, [openSavedGraphInNewProject, templateList]);

  const deleteTemplateEntry = useCallback(
    async (entry: SavedGraphEntry) => {
      await withRailDeleteOverlay("Deleting template…", async () => {
        try {
          const next = await removeSavedGraphEntry("template", entry.id);
          setTemplateList(next);
          bumpTemplateLibraryListeners();
          if (entry.libraryOrigin === "combined_model") {
            window.dispatchEvent(new Event(GRAPH_COMBINED_MODEL_LIBRARY_CHANGED));
          }
        } catch (e) {
          setNotice(e instanceof Error ? e.message : "Could not delete template.");
        }
      });
    },
    [bumpTemplateLibraryListeners, withRailDeleteOverlay],
  );

  const renameTemplateEntry = useCallback(async (entry: SavedGraphEntry, newName: string) => {
    const trimmed = newName.trim();
    if (!trimmed || trimmed === entry.name) return;
    try {
      const updated: SavedGraphEntry = {
        ...entry,
        name: trimmed.slice(0, 200),
        savedAt: Date.now(),
      };
      const next = await addSavedGraphEntry("template", updated);
      setTemplateList(next);
      bumpTemplateLibraryListeners();
      setNotice(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Could not rename template.";
      setNotice(msg);
      throw e instanceof Error ? e : new Error(msg);
    }
  }, [bumpTemplateLibraryListeners]);

  const renameProject = useCallback(
    (id: string, title: string) => {
      setProjects((list) =>
        list.map((p) => {
          if (p.id !== id) return p;
          if (id !== activeProjectId) return { ...p, title };
          return withProjectCanvas({ ...p, title }, (c) => ({ ...c, dirty: true }));
        }),
      );
    },
    [activeProjectId],
  );

  const readGraphForCanvas = useCallback((projectId: string) => {
    const c = projectsRef.current.find((x) => x.id === projectId)?.canvas;
    if (!c) return null;
    return { nodes: c.nodes, edges: c.edges };
  }, []);

  const graphActions = useMemo(
    () => ({
      addNode: (nodeType: string, screenPos?: { x: number; y: number }, options?: AddNodeOptions) => {
        addNodeImplRef.current(nodeType, screenPos, options);
      },
      setFlowNodes: (updater: SetStateAction<Node[]>) => {
        if (!activeProject) return;
        setNodesForCanvas(activeProject.id, updater);
      },
      setFlowEdges: (updater: SetStateAction<Edge[]>) => {
        if (!activeProject) return;
        setEdgesForCanvas(activeProject.id, updater);
      },
    }),
    [
      activeCanvas,
      activeProject,
      setEdgesForCanvas,
      setNodesForCanvas,
    ],
  );

  const openGraphCompareDoc = useCallback((doc: GraphDocument) => {
    setCompareSourceDoc(doc);
    setGraphCompareOpen(true);
  }, []);

  const closeGraphCompare = useCallback(() => {
    setGraphCompareOpen(false);
    setCompareSourceDoc(null);
  }, []);

  if (!activeProject || !activeCanvas) {
    return null;
  }

  const templateRailEntries = templateList.filter((e) => e.libraryOrigin !== "combined_model");
  const classicPaperTemplates = templateRailEntries.filter(isClassicPaperReproductionTemplate);
  const blogTemplates = templateRailEntries.filter((e) => !isClassicPaperReproductionTemplate(e));

  return (
    <ReactFlowProvider>
      <ResearchGraphProvider value={graphActions}>
          <div className="cr-shell">
          <ProjectTabBar
            projects={projects}
            activeProjectId={activeProjectId}
            trainByProjectId={trainByProjectId}
            onSelect={(id) => {
              setActiveProjectId(id);
              setNotice(null);
            }}
            onAdd={addProject}
            onClose={closeProject}
            onRenameProject={renameProject}
          />
          {projectClosePending ? (
            <div
              className="cr-modal-backdrop cr-modal-backdrop--project-close"
              role="presentation"
              onMouseDown={(e) => {
                if (e.target === e.currentTarget) setProjectClosePending(null);
              }}
            >
              <div
                className="cr-modal cr-modal--project-close"
                role="dialog"
                aria-modal="true"
                aria-labelledby="cr-project-close-title"
                onMouseDown={(e) => e.stopPropagation()}
              >
                <h2 id="cr-project-close-title" className="cr-modal__title">
                  Stop training and close project?
                </h2>
                <p className="cr-modal__hint">
                  {projectClosePending.summary.paused ? "Training is paused" : "Training is in progress"} in{" "}
                  <strong>{projectClosePending.title}</strong> ({projectClosePending.summary.progressPct}%).
                  Closing will abort the run on the server.
                </p>
                <div className="cr-modal__actions">
                  <button
                    type="button"
                    className="cr-modal__btn cr-modal__btn--ghost"
                    onClick={() => setProjectClosePending(null)}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    className="cr-modal__btn cr-modal__btn--primary"
                    onClick={() => void finalizeCloseProject(projectClosePending.projectId)}
                  >
                    Stop training and close
                  </button>
                </div>
              </div>
            </div>
          ) : null}
          <div
            className="cr-workbench"
            style={{ "--cr-rail-panel-width": `${workbenchRailPanelWidthPx}px` } as CSSProperties}
          >
            <LeftNavRail
              activeSection={railSection}
              onSelectSection={handleRailSelect}
            />
            <>
                <div
                  className="cr-workbench-rail-panel"
                  style={{ width: workbenchRailPanelWidthPx }}
                  hidden={!railSection}
                >
                  <div className="cr-workbench-rail-panel__slot" hidden={railSection !== "nodes"}>
                    <NodesLibraryPanel />
                  </div>
                  {railSection === "templates" ? (
                    <div className="cr-workbench-rail-panel__slot">
                      <SavedGraphLibraryPanel
                        title="Templates"
                        emptyHint='Use Graph → "Save graph as template" for a reusable starter. Click an item to open it in a new tab.'
                        sectionGroups={[
                          {
                            title: "Reproduce classic paper",
                            entries: classicPaperTemplates,
                            displayEntryName: (entry) => (
                              entry.name.trimStart().toLowerCase().startsWith("repro:")
                                ? entry.name
                                : `repro: ${entry.name}`
                            ),
                            collapsible: true,
                            defaultCollapsed: true,
                          },
                          {
                            title: "Blogs",
                            entries: blogTemplates,
                            collapsible: true,
                            defaultCollapsed: true,
                          },
                        ]}
                        onOpen={(e) => openSavedGraphInNewProject(e, "template")}
                        onDelete={deleteTemplateEntry}
                        onRename={renameTemplateEntry}
                      />
                    </div>
                  ) : null}
                  {railSection === "observables" ? (
                    <div className="cr-workbench-rail-panel__slot">
                      <ObservablePanel
                        nodes={activeCanvas.nodes}
                        edges={activeCanvas.edges}
                        selectedNodeId={canvasSelectedNodeId}
                      />
                    </div>
                  ) : null}
                </div>
                <div
                  className={`cr-workbench-rail-panel__grip${railPanelGripDragging ? " cr-workbench-rail-panel__grip--active" : ""}`}
                  role="separator"
                  aria-orientation="vertical"
                  aria-label="Resize side panel"
                  tabIndex={0}
                  hidden={!railSection}
                  onPointerDown={onRailPanelGripPointerDown}
                  onPointerMove={onRailPanelGripPointerMove}
                  onPointerUp={onRailPanelGripPointerUp}
                  onPointerCancel={onRailPanelGripPointerUp}
                  onKeyDown={onRailPanelGripKeyDown}
                />
            </>
            <div className="cr-canvas-code-split">
              <div className="cr-canvas-code-split__canvas">
                <FlowApp
                  addNodeImplRef={addNodeImplRef}
                  codeMode={codeMode}
                  onCodeModeChange={onWorkbenchCodeModeChange}
                  flowSurfaceKey={`${activeProject.id}-${activeCanvas.id}`}
                  nodes={activeCanvas.nodes}
                  edges={activeCanvas.edges}
                  onNodesChange={onNodesChange}
                  onEdgesChange={onEdgesChange}
                  owningProjectId={activeProject.id}
                  setNodesForCanvas={setNodesForCanvas}
                  setEdgesForCanvas={setEdgesForCanvas}
                  onClearCanvas={clearActiveCanvas}
                  savedViewport={activeCanvas.savedViewport}
                  viewportApplyNonce={activeCanvas.viewportApplyNonce}
                  onSaveToServer={onSaveToServer}
                  onLoadFromServer={onLoadFromServer}
                  onSaveServerSucceeded={onSaveServerSucceeded}
                  onSaveGraphToFile={onSaveGraphToFile}
                  graphFileStemBase={`graph-${activeProjectId.slice(0, 8)}-${activeCanvas.id.slice(0, 8)}`}
                  onPersistLibraryGraph={onPersistLibraryGraph}
                  onGraphFileLoaded={onGraphFileLoaded}
                  onGraphFileError={onGraphFileError}
                  onOpenGraphCompare={openGraphCompareDoc}
                  graphFileHandle={activeCanvas.localGraphFileHandle ?? null}
                  librarySource={activeCanvas.librarySource}
                  librarySaveDisplayName={activeProject.title}
                  onSaveLibrarySourceEntry={onSaveGraphToLibraryEntry}
                  onSaveGraphToSourceFileSucceeded={onSaveGraphToSourceFileSucceeded}
                  loading={loading}
                  error={error}
                  notice={notice}
                  readGraphForCanvas={readGraphForCanvas}
                  onCanvasSelectionChange={onCanvasSelectionChange}
                  onRequestCloseRail={() => setRailSection(null)}
                  onRequestCloseNodeInformation={() => {
                    setNodeInformationOpen(false);
                    setNodeInformationContent(null);
                  }}
                  onRequestOpenNodesRail={() => {
                    // Commit the rail's regular open state before React Flow
                    // processes the next drag coordinate, so viewport
                    // compensation keeps the dragged node under the pointer.
                    flushSync(() => setRailSection("nodes"));
                  }}
                  nodesRailOpen={railSection === "nodes"}
                />
              </div>
            </div>
            {nodeForInformation ? (
              <NodeInformationPanel
                node={nodeForInformation}
                informationTitle={nodeInformationContent?.title}
                informationText={nodeInformationContent?.text}
                code={nodeInformationContent?.code}
                initialMode={nodeInformationContent?.mode}
                onUpdateData={(key, value) => {
                  setNodesForCanvas(activeProject.id, (current) =>
                    current.map((node) =>
                      node.id === nodeForInformation.id
                        ? { ...node, data: { ...(node.data as Record<string, unknown>), [key]: value } }
                        : node,
                    ),
                  );
                }}
              />
            ) : null}
          </div>
          <div className="cr-build-stamp" title="If this label is missing, the page is not this build.">
            ComfyResearch
          </div>
          <DeletingBusyOverlay open={railDeletingMessage !== null} message={railDeletingMessage ?? "Deleting…"} />
          <LibrarySaveModal
            draft={librarySaveDraft}
            onDismiss={dismissLibraryModal}
            onConfirm={confirmLibrarySave}
          />
          {graphCompareOpen && compareSourceDoc ? (
            <GraphCompareModal
              open
              onClose={closeGraphCompare}
              sourceLabel={`${activeProject.title} / ${activeCanvas.title}`}
              sourceDocument={compareSourceDoc}
              targets={graphCompareTargets}
            />
          ) : null}
          </div>
      </ResearchGraphProvider>
    </ReactFlowProvider>
  );
}
