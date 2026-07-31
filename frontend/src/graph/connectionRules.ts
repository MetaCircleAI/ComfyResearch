/** Canvas 连接规则纯函数(自 ResearchCanvas.tsx 逐字抽出,零行为)。
 *
 * 三件套 + 4 个类型谓词 helper。签名要点:
 * - isValidCanvasConnection 带 edges(tableViz 分支查上游 sweep 状态);
 * - applyCanvasConnection 的新节点 id 仍走 appendResearchNode 默认随机——金标侧
 *   做 $new0/$new1 归一化（注入 idFactory 会触及 4 个 spawn 调用点）；
 * - planAutoConnectCanvas 只做规划(最近邻/去重/handle 优先级),apply 循环留组件。
 * cascade 分支序与默认 return true 尾是行为本体,严禁重排。
 */
import { addEdge, type Connection, type Edge, type Node } from "@xyflow/react";

import {
  COMBINED_MODEL_RETURN_TARGET_HANDLE,
  isLayerStripSourceHandle,
  isLayerStripTargetHandle,
} from "./layerStripHandles";
import { readNodeCanvasIoMode } from "./nodeCanvasIoMode";
import { appendResearchNode } from "./nodeInstanceTitle";
import { spawnConfigFor } from "./observableVizTrainerSpawn";
import {
  isObservableVizFlowType,
  observableVizAllowsTensorVizChain,
  type ObservableVizVariant,
} from "./observableVizVariant";
import {
  defaultTensorKeyForMultiChoices,
  tensorChoicesFromSourceHandle,
  type FlowNodeBare,
} from "./resolveUpstreamTensor";
import { tableVizTensorConnectable, tableVizTensorListChoices } from "./tableVizRegressor";
import {
  TRAINER_AUTO_OBSERVABLE_VIZ_DY,
  TRAINER_AUTO_TRAINING_VIZ_DY,
  TRAINER_AUTO_VIZ_DX,
} from "./trainerAutoVizSpawn";
import { defaultTensorSelectorData, isTensorSelectorSourceHandle } from "../components/nodes/tensorSelectorDefaults";
import { defaultTrainingVisualizationData } from "../components/nodes/trainingVisualizationDefaults";
import { GENERATED_NODE_SPECS } from "../generated/generatedNodeSpecs";
import { nodeRegistryTypesWithFamily } from "./nodeRegistrySpec";

/** 类型集由 family 派生（Python definitions 持有真相，golden SHA 验证恒等）。
 * ai4science 5 层例外是有意保留的差异。后端经
 * ai4science_alias remap(pairwise_rbf→linear 等)组装,capability 集(13)
 * 不含它们是正确语义;canvas 连接面上它们以自身类型渲染 strip,故在此。 */
const AI4_SCIENCE_STRIP_EXCEPTIONS = [
  "pairwise_rbf_layer",
  "equivariant_message_layer",
  "energy_readout_layer",
  "relative_pose_encoder_layer",
  "distance_contact_layer",
] as const;

const ATOMIC_LAYER_FAMILY_TYPES: ReadonlySet<string> = new Set(nodeRegistryTypesWithFamily("atomic_layer_model"));

const ATOMIC_LAYER_CONN_TYPES: ReadonlySet<string> = new Set([
  ...ATOMIC_LAYER_FAMILY_TYPES,
  ...AI4_SCIENCE_STRIP_EXCEPTIONS,
]);

/** 28 员恒等式:canvas_trainer_model_source ∖ atomic_layer_model ∖ {combined_model}。 */
const FULL_MODEL_CANVAS_CONN_TYPES: ReadonlySet<string> = new Set(
  nodeRegistryTypesWithFamily("canvas_trainer_model_source").filter(
    (t) => !ATOMIC_LAYER_FAMILY_TYPES.has(t) && t !== "combined_model",
  ),
);

const OPTIMIZER_CANVAS_CONN_TYPES: ReadonlySet<string> = new Set(nodeRegistryTypesWithFamily("optimizer_node"));

/** lr-schedule 源资格 family 派生(lr_schedule/mup_lr_schedule/
 * cyclic_lr_schedule 恰三员)。mup 专属路径仍按 type 恒等——family 只放宽
 * "是 lr schedule 源"的成员资格(plain 集 = family ∖ mup)。 */
const TRAINER_LR_SCHEDULE_TYPES: ReadonlySet<string> = new Set(nodeRegistryTypesWithFamily("trainer_lr_schedule"));
const PLAIN_LR_SCHEDULE_SOURCE_TYPES: ReadonlySet<string> = new Set(
  [...TRAINER_LR_SCHEDULE_TYPES].filter((t) => t !== "mup_lr_schedule"),
);

/** autoConnect 的四个集合由 family 派生。 */
const AUTOCONNECT_DATASET_TYPES: ReadonlySet<string> = new Set(nodeRegistryTypesWithFamily("canvas_trainer_autoconnect_dataset"));

const PRIMARY_LOSS_CONN_TYPES: ReadonlySet<string> = new Set(nodeRegistryTypesWithFamily("trainer_primary_loss"));
const LOSS_SOCKET_AUX_CONN_TYPES: ReadonlySet<string> = new Set(nodeRegistryTypesWithFamily("trainer_loss_socket_aux"));

const MODEL_INITIALIZATION_TARGET_TYPES: ReadonlySet<string> = new Set([
  ...FULL_MODEL_CANVAS_CONN_TYPES,
  "combined_model",
  ...ATOMIC_LAYER_CONN_TYPES,
]);

/** 其余连接集合也由 definitions 中声明的 family 派生。 */
const TENSOR_VIZ_CONN_TYPES: ReadonlySet<string> = new Set(nodeRegistryTypesWithFamily("observable_user_tensor_viz_display"));
const TENSOR_MULTI_INPUT_CONN_TYPES: ReadonlySet<string> = new Set(nodeRegistryTypesWithFamily("canvas_tensor_multi_input"));
const LAYER_STRIP_CHAIN_CONN_TYPES: ReadonlySet<string> = new Set(nodeRegistryTypesWithFamily("canvas_layer_strip_chain"));
const VIZ_COMMENT_SOURCE_TYPES: ReadonlySet<string> = new Set([
  ...nodeRegistryTypesWithFamily("observable_user_tensor_viz_display"),
  ...nodeRegistryTypesWithFamily("canvas_comment_source"),
]);
const DATASET_TENSOR_LIST_SOURCE_TYPES: ReadonlySet<string> = new Set(nodeRegistryTypesWithFamily("canvas_dataset_source"));

const SINGLE_TENSOR_TARGET_TYPES: ReadonlySet<string> = new Set([
  ...nodeRegistryTypesWithFamily("observable_user_tensor_viz_display"),
  ...nodeRegistryTypesWithFamily("canvas_single_tensor_target"),
]);

/** autoConnect model 集:atomic **family 13**,不含 ai4science 例外。 */
const AUTOCONNECT_MODEL_TYPES: ReadonlySet<string> = new Set([
  ...FULL_MODEL_CANVAS_CONN_TYPES,
  "combined_model",
  ...ATOMIC_LAYER_FAMILY_TYPES,
]);

function isTensorConstantOrGeneratedSource(sourceNode: Node | undefined, sh: string | null | undefined): boolean {
  const s = (sh ?? "").trim();
  if (s !== "tensor") return false;
  return sourceNode?.type === "tensor_constant" || sourceNode?.type === "tensor_linspace" || sourceNode?.type === "fake_tensor";
}

/** Effective rank and series endpoint gap expose a 0-D tensor on source handle ``tensor``. */
function isScalarAnalysisTensorSource(sourceNode: Node | undefined, sh: string | null | undefined): boolean {
  const s = (sh ?? "").trim();
  if (s !== "tensor") return false;
  const t = sourceNode?.type;
  return t === "effective_rank" || t === "series_endpoint_gap" || t === "basic_calculator";
}

/** PCA / SVD tensor outputs used like upstream tensors for viz, statistics, and layer chains. */
function isAnalysisTensorOutSource(sourceNode: Node | undefined, sh: string | null | undefined): boolean {
  const s = (sh ?? "").trim();
  const t = sourceNode?.type;
  if (t === "pca") {
    return s === "transformed_tensor" || s === "principal_components" || s === "explained_variance_ratio";
  }
  if (t === "svd") {
    return s === "u" || s === "s" || s === "v";
  }
  return false;
}

function isTrainerLikeCanvasType(t: string | undefined | null): boolean {
  return t === "trainer" || t === "crl_trainer";
}

export function applyCanvasConnection(
  params: Connection,
  baseNodes: Node[],
  baseEdges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
      if (!params.source || !params.target) return { nodes: baseNodes, edges: baseEdges };

      const singleTensorTargetTypes = SINGLE_TENSOR_TARGET_TYPES;
      const targetNode = baseNodes.find((n) => n.id === params.target);
      const sourceNode = baseNodes.find((n) => n.id === params.source);
      const th = params.targetHandle ?? "";
      const sh = params.sourceHandle ?? "";
      const tensorSingleFanIn =
        (targetNode != null && singleTensorTargetTypes.has(String(targetNode.type)) && th === "tensor") ||
        (targetNode?.type === "dimension_permutator" && th === "tensor_in");
      if (targetNode && sourceNode && tensorSingleFanIn) {
        const choices = tensorChoicesFromSourceHandle(
          baseNodes as FlowNodeBare[],
          sourceNode as FlowNodeBare,
          sh,
          baseEdges,
        );
        if (choices.length > 1) {
          const selectedKey = defaultTensorKeyForMultiChoices(choices);
          const tsPos = {
            x: (sourceNode.position.x + targetNode.position.x) / 2,
            y: (sourceNode.position.y + targetNode.position.y) / 2,
          };
          const newTs: Node = appendResearchNode(baseNodes, "tensor_selector", tsPos, {
            ...(defaultTensorSelectorData() as Record<string, unknown>),
            selectedTensorKey: selectedKey,
            selectedTensorKeys: [selectedKey],
          });
          let wired = addEdge(
            {
              source: params.source,
              target: newTs.id,
              sourceHandle: sh || null,
              targetHandle: "tensor_list",
            },
            baseEdges,
          );
          const fanInHandle = targetNode.type === "dimension_permutator" ? "tensor_in" : "tensor";
          wired = addEdge(
            {
              source: newTs.id,
              target: params.target,
              sourceHandle: "tensor_1",
              targetHandle: fanInHandle,
            },
            wired,
          );
          return { nodes: [...baseNodes, newTs], edges: wired };
        }
      }

      let nextNodes = baseNodes;
      const optimizerCanvasTypesForConnect = OPTIMIZER_CANVAS_CONN_TYPES;
      let connectParams = params;
      if (
        targetNode &&
        sourceNode &&
        optimizerCanvasTypesForConnect.has(String(targetNode.type)) &&
        (params.targetHandle ?? "") === "optimizer_lr_schedule"
      ) {
        const sh0 = params.sourceHandle ?? "";
        if (PLAIN_LR_SCHEDULE_SOURCE_TYPES.has(String(sourceNode.type)) && sh0 === "lr_schedule") {
          connectParams = { ...params, targetHandle: "lr_schedule" };
        } else if (sourceNode.type === "mup_lr_schedule" && sh0 === "mup_lr_schedule") {
          connectParams = { ...params, targetHandle: "mup_lr_schedule" };
        }
      }
      let edgesForAdd = baseEdges;
      if (
        targetNode &&
        isObservableVizFlowType(String(targetNode.type)) &&
        (connectParams.targetHandle ?? "") === "tensor"
      ) {
        edgesForAdd = baseEdges.filter(
          (e) => !(e.target === connectParams.target && (e.targetHandle ?? "") === "tensor"),
        );
      }
      let nextEdges = addEdge(connectParams, edgesForAdd);

      if (
        params.targetHandle === "observables" &&
        (params.sourceHandle === "observables" || params.sourceHandle === "observable")
      ) {
        const trainer = nextNodes.find((n) => n.id === params.target);
        const obs = nextNodes.find((n) => n.id === params.source);
        const spawn = isTrainerLikeCanvasType(String(trainer?.type ?? "")) && obs?.type ? spawnConfigFor(String(obs.type)) : undefined;
        if (spawn && trainer) {
          const already = nextEdges.some((e) => {
            if (
              e.source !== params.target ||
              e.sourceHandle !== "observable_results" ||
              e.targetHandle !== "tensor"
            ) {
              return false;
            }
            const vn = nextNodes.find((x) => x.id === e.target);
            const d = (vn?.data ?? {}) as {
              pairedObservableId?: string;
              pairedTrainerId?: string;
              vizVariant?: ObservableVizVariant;
            };
            return (
              vn?.type === "observable_viz" &&
              d.pairedObservableId === params.source &&
              d.pairedTrainerId === params.target &&
              d.vizVariant === spawn.vizVariant
            );
          });
          if (!already) {
            const vizNode = appendResearchNode(
              nextNodes,
              "observable_viz",
              {
                x: trainer.position.x + TRAINER_AUTO_VIZ_DX,
                y: trainer.position.y + TRAINER_AUTO_OBSERVABLE_VIZ_DY,
              },
              spawn.defaultData(params.source, params.target, obs) as Record<string, unknown>,
            );
            nextNodes = [...nextNodes, vizNode];
            nextEdges = addEdge(
              {
                source: params.target,
                target: vizNode.id,
                sourceHandle: "observable_results",
                targetHandle: "tensor",
              },
              nextEdges,
            );
          }
        }
      }

      if (params.targetHandle === "loss" && params.sourceHandle === "loss") {
        const trainer = nextNodes.find((n) => n.id === params.target);
        const lossNode = nextNodes.find((n) => n.id === params.source);
        const isSupportedLoss = Boolean(
          lossNode?.type && nodeRegistryTypesWithFamily("trainer_loss_viz_spawn").includes(lossNode.type),
        );
        if (trainer?.type === "trainer" && isSupportedLoss) {
          const already = nextEdges.some((e) => {
            if (
              e.source !== params.target ||
              e.sourceHandle !== "loss_results" ||
              e.targetHandle !== "tensor_list"
            ) {
              return false;
            }
            const vn = nextNodes.find((x) => x.id === e.target);
            return vn?.type === "training_visualization";
          });
          if (!already) {
            const vizNode = appendResearchNode(
              nextNodes,
              "training_visualization",
              {
                x: trainer.position.x + TRAINER_AUTO_VIZ_DX,
                y: trainer.position.y + TRAINER_AUTO_TRAINING_VIZ_DY,
              },
              defaultTrainingVisualizationData() as Record<string, unknown>,
            );
            nextNodes = [...nextNodes, vizNode];
            nextEdges = addEdge(
              {
                source: params.target,
                target: vizNode.id,
                sourceHandle: "loss_results",
                targetHandle: "tensor_list",
              },
              nextEdges,
            );
          }
        }
      }

      if (params.targetHandle === "model" && params.sourceHandle === "model") {
        const trainer = nextNodes.find((n) => n.id === params.target);
        if (trainer?.type === "crl_trainer") {
          const already = nextEdges.some((e) => {
            if (
              e.source !== params.target ||
              e.sourceHandle !== "loss_results" ||
              e.targetHandle !== "tensor_list"
            ) {
              return false;
            }
            const vn = nextNodes.find((x) => x.id === e.target);
            return vn?.type === "training_visualization";
          });
          if (!already) {
            const vizNode = appendResearchNode(
              nextNodes,
              "training_visualization",
              {
                x: trainer.position.x + TRAINER_AUTO_VIZ_DX,
                y: trainer.position.y + TRAINER_AUTO_TRAINING_VIZ_DY,
              },
              defaultTrainingVisualizationData() as Record<string, unknown>,
            );
            nextNodes = [...nextNodes, vizNode];
            nextEdges = addEdge(
              {
                source: params.target,
                target: vizNode.id,
                sourceHandle: "loss_results",
                targetHandle: "tensor_list",
              },
              nextEdges,
            );
          }
        }
      }

      return { nodes: nextNodes, edges: nextEdges };
}

/** 声明式入口端口判定。null = target 未声明 ports(cascade 原样穿透);
 * 声明型 return-style 全权接管:th 不匹配任何 in-port → false(与原 return-style
 * 分支一致);匹配则 accepts 逐条判(type 恒等 / family 成员 / sh ∈ handles)。
 * 插入位置铁律:必须在首个被接管分支(observable_user)的
 * **原位**,不得提前——否则声明型的 false 收紧会短路更早的允许路径。 */
const FAMILY_MEMBER_CACHE = new Map<string, ReadonlySet<string>>();
function familyMembers(f: string): ReadonlySet<string> {
  let m = FAMILY_MEMBER_CACHE.get(f);
  if (!m) {
    m = new Set(nodeRegistryTypesWithFamily(f as never));
    FAMILY_MEMBER_CACHE.set(f, m);
  }
  return m;
}

function generatedInPortsVerdict(
  targetNode: Node | undefined,
  th: string,
  sourceNode: Node | undefined,
  sh: string,
): boolean | null {
  const ports = targetNode?.type != null ? GENERATED_NODE_SPECS[String(targetNode.type)]?.ports?.in : undefined;
  if (!ports) return null;
  const t = (th ?? "").trim();
  const port = ports.find((p) => p.id === t);
  if (!port) return false;
  const s = (sh ?? "").trim();
  const srcType = String(sourceNode?.type ?? "");
  return port.accepts.some(
    (a) =>
      (a.type == null || srcType === a.type) &&
      (a.family == null || familyMembers(a.family).has(srcType)) &&
      // ioMode 条件接受——语义严格镜像 fullModel*/combinedModel* 谓词
      // 的 readNodeCanvasIoMode(source.data) 检查。
      (a.ioMode == null ||
        readNodeCanvasIoMode((sourceNode?.data ?? {}) as Record<string, unknown>) === a.ioMode) &&
      a.handles.includes(s),
  );
}

export function isValidCanvasConnection(edge: Connection | Edge, nodes: Node[], edges: Edge[]): boolean {
      /* While dragging, XYFlow may validate before a target exists — do not block the connection line. */
      if (!edge.target) return true;

      const targetNode = nodes.find((n) => n.id === edge.target);
      const sourceNode = nodes.find((n) => n.id === edge.source);
      const sh = edge.sourceHandle ?? "";
      const th = edge.targetHandle ?? "";
      if (sourceNode?.type === "image_dataset_displayer") return false;
      const tensorVizTypes = TENSOR_VIZ_CONN_TYPES;
      const atomicLayerTypes = ATOMIC_LAYER_CONN_TYPES;
      const fullModelCanvasTypes = FULL_MODEL_CANVAS_CONN_TYPES;
      const modelInitializationTargetTypes = MODEL_INITIALIZATION_TARGET_TYPES;
      const tensorMultiInputTypes = TENSOR_MULTI_INPUT_CONN_TYPES;
      const isTensorMultiInputSource = (n: Node | undefined, sourceHandle: string) =>
        n != null &&
        (tensorMultiInputTypes.has(String(n.type)) || n.type === "elementwise_transform") &&
        sourceHandle === "tensor";
      const acceptsTensorMultiInputTarget = (n: Node | undefined, targetHandle: string) =>
        n != null &&
        (n.type === "statistics2" || n.type === "tensor_viz_scatter" || tensorMultiInputTypes.has(String(n.type))) &&
        /^tensor_\d+$/.test(targetHandle);

      const combinedModelTensorIoTarget = (n: Node | undefined, th: string) =>
        n?.type === "combined_model" &&
        readNodeCanvasIoMode((n?.data ?? {}) as Record<string, unknown>) === "input-output" &&
        isLayerStripTargetHandle(th);

      const combinedModelTensorIoSource = (n: Node | undefined, sh: string) =>
        n?.type === "combined_model" &&
        readNodeCanvasIoMode((n?.data ?? {}) as Record<string, unknown>) === "input-output" &&
        isLayerStripSourceHandle(sh);

      const fullModelTensorIoTarget = (n: Node | undefined, th: string) =>
        fullModelCanvasTypes.has(String(n?.type)) &&
        readNodeCanvasIoMode((n?.data ?? {}) as Record<string, unknown>) === "input-output" &&
        isLayerStripTargetHandle(th);

      const fullModelTensorIoSource = (n: Node | undefined, sh: string) =>
        fullModelCanvasTypes.has(String(n?.type)) &&
        readNodeCanvasIoMode((n?.data ?? {}) as Record<string, unknown>) === "input-output" &&
        isLayerStripSourceHandle(sh);

      const fullModelModelSocket = (n: Node | undefined, h: string) =>
        fullModelCanvasTypes.has(String(n?.type)) &&
        h === "model" &&
        readNodeCanvasIoMode((n?.data ?? {}) as Record<string, unknown>) === "model";

      const atomicLayerTensorIoTarget = (n: Node | undefined, handle: string) =>
        atomicLayerTypes.has(String(n?.type)) &&
        readNodeCanvasIoMode((n?.data ?? {}) as Record<string, unknown>) === "input-output" &&
        isLayerStripTargetHandle(handle);

      const atomicLayerTensorSourceOut = (n: Node | undefined, sh: string) => {
        if (!n || !atomicLayerTypes.has(String(n.type))) return false;
        const mode = readNodeCanvasIoMode((n.data ?? {}) as Record<string, unknown>);
        if (mode === "model") return sh === "tensor";
        return isLayerStripSourceHandle(sh);
      };

      /** Paired tensor strip nodes (not ``nn.Module`` layers) used in combined / attention subgraph chains. */
      const layerStripChainNodeTypes = LAYER_STRIP_CHAIN_CONN_TYPES;

      const layerStripChainTensorIoTarget = (n: Node | undefined, handle: string) =>
        layerStripChainNodeTypes.has(String(n?.type)) &&
        readNodeCanvasIoMode((n?.data ?? {}) as Record<string, unknown>) === "input-output" &&
        isLayerStripTargetHandle(handle);

      const layerStripChainTensorSourceOut = (n: Node | undefined, sh: string) => {
        if (!n || !layerStripChainNodeTypes.has(String(n.type))) return false;
        const mode = readNodeCanvasIoMode((n.data ?? {}) as Record<string, unknown>);
        if (mode === "model") return (sh ?? "").trim() === "tensor";
        return isLayerStripSourceHandle(sh);
      };

      const atomicOrChainStripTensorSourceOut = (n: Node | undefined, sh: string) =>
        atomicLayerTensorSourceOut(n, sh) || layerStripChainTensorSourceOut(n, sh);

      /** Right output in `model` mode: canonical `tensor`; accept legacy `model` handle id from older graphs. */
      const combinedModelModelModeOut = (n: Node | undefined, h: string) =>
        n?.type === "combined_model" &&
        readNodeCanvasIoMode((n?.data ?? {}) as Record<string, unknown>) === "model" &&
        (h === "tensor" || h === "model");

      const combinedModelTensorBoundarySource = (n: Node | undefined, h: string) =>
        n?.type === "combined_model" &&
        h === "tensor_boundary" &&
        readNodeCanvasIoMode((n?.data ?? {}) as Record<string, unknown>) === "input-output";

      const combinedModelTensorReturnTarget = (n: Node | undefined, h: string) =>
        n?.type === "combined_model" &&
        h === COMBINED_MODEL_RETURN_TARGET_HANDLE &&
        readNodeCanvasIoMode((n?.data ?? {}) as Record<string, unknown>) === "input-output";

      /** Valid sources for ``tensor_in`` on atomic / full-model / combined-model paired strips. */
      const layerStripTensorInSource = (sourceNode: Node | undefined, sh: string) => {
        if (!sourceNode) return false;
        if (atomicOrChainStripTensorSourceOut(sourceNode, sh)) return true;
        if (combinedModelTensorIoSource(sourceNode, sh)) return true;
        if (fullModelTensorIoSource(sourceNode, sh)) return true;
        const s = (sh ?? "").trim();
        if (isTensorMultiInputSource(sourceNode, s)) return true;
        if (isTensorConstantOrGeneratedSource(sourceNode, s)) return true;
        if (isScalarAnalysisTensorSource(sourceNode, s)) return true;
        if (sourceNode.type === "statistics" && s === "tensor") return true;
        if (sourceNode.type === "statistics2" && s === "tensor") return true;
        if (sourceNode.type === "dimension_permutator" && s === "tensor_out") return true;
        if ((sourceNode.type === "tensor_slicing" || sourceNode.type === "elementwise_transform") && s === "tensor") return true;
        if (sourceNode.type === "tensor_selector" && isTensorSelectorSourceHandle(sh)) return true;
        if (isAnalysisTensorOutSource(sourceNode, sh)) return true;
        if (sourceNode.type === "training_visualization" && s === "out_tensor_list") return true;
        if (
          (sourceNode.type === "observable_viz" || sourceNode.type === "observable_accuracy") &&
          s === "out_tensor"
        ) {
          return true;
        }
        return false;
      };
      const isDatasetTensorListSource = (n: Node | undefined, sourceHandle: string) => {
        // -C1d:37 员 if-chain == canvas_dataset_source family(swap 前逐员核对)。
        if (!n || !DATASET_TENSOR_LIST_SOURCE_TYPES.has(String(n.type))) return false;
        return (
          sourceHandle === "dataset" ||
          sourceHandle === "train_dataset" ||
          sourceHandle === "test_dataset"
        );
      };

      if (sourceNode && combinedModelTensorBoundarySource(sourceNode, sh)) {
        if (targetNode?.type === "dimension_permutator" && th === "tensor_in") return true;
        if ((targetNode?.type === "tensor_slicing" || targetNode?.type === "elementwise_transform") && th === "tensor") return true;
        if (targetNode && atomicLayerTypes.has(String(targetNode.type)) && atomicLayerTensorIoTarget(targetNode, th)) {
          return true;
        }
        if (targetNode && layerStripChainTensorIoTarget(targetNode, th)) return true;
        return false;
      }

      if (targetNode && combinedModelTensorReturnTarget(targetNode, th)) {
        if (!sourceNode) return false;
        if (atomicOrChainStripTensorSourceOut(sourceNode, sh)) return true;
        if (fullModelTensorIoSource(sourceNode, sh)) return true;
        if (combinedModelTensorIoSource(sourceNode, sh)) return true;
        if (combinedModelModelModeOut(sourceNode, sh)) return true;
        const s = (sh ?? "").trim();
        if (isTensorMultiInputSource(sourceNode, s)) return true;
        if (isTensorConstantOrGeneratedSource(sourceNode, s)) return true;
        if (isScalarAnalysisTensorSource(sourceNode, s)) return true;
        if (sourceNode.type === "statistics" && s === "tensor") return true;
        if (sourceNode.type === "statistics2" && s === "tensor") return true;
        if (sourceNode.type === "dimension_permutator" && (s === "tensor_out" || s === "tensor")) return true;
        return false;
      }

      /** Same tensor sources allowed into ``tensor_viz_*`` / ``tensor`` on observable mirror panels. */
      const analysisTensorFeedLikeTensorViz = (src: Node | undefined, s: string): boolean => {
        if (isAnalysisTensorOutSource(src, s)) return true;
        if (src?.type === "statistics" && s === "tensor") return true;
        if (src?.type === "statistics2" && s === "tensor") return true;
        if (isTensorMultiInputSource(src, s)) return true;
        if (isTensorConstantOrGeneratedSource(src, s)) return true;
        if (isScalarAnalysisTensorSource(src, s)) return true;
        if (src?.type === "tensor_selector" && isTensorSelectorSourceHandle(s)) return true;
        if (tensorVizTypes.has(String(src?.type)) && s === "out_tensor") return true;
        if (src?.type === "training_visualization" && s === "out_tensor_list") return true;
        if (isTrainerLikeCanvasType(src?.type) && (s === "loss_results" || s === "observable_results")) {
          return true;
        }
        if (observableVizAllowsTensorVizChain(src) && s === "out_tensor") return true;
        if (src?.type === "dimension_permutator" && s === "tensor_out") return true;
        if ((src?.type === "tensor_slicing" || src?.type === "elementwise_transform") && s === "tensor") {
          return true;
        }
        if (atomicOrChainStripTensorSourceOut(src, s)) return true;
        if (fullModelTensorIoSource(src, s)) return true;
        if (fullModelModelSocket(src, s)) return true;
        if (combinedModelModelModeOut(src, s)) return true;
        if (combinedModelTensorIoSource(src, s)) return true;
        return false;
      };

      if (targetNode && tensorVizTypes.has(String(targetNode.type))) {
        if (th !== "tensor") return false;
        return analysisTensorFeedLikeTensorViz(sourceNode, sh);
      }

      if (targetNode && isObservableVizFlowType(String(targetNode.type))) {
        if (th != null && th !== "" && th !== "tensor") return false;
        if (isTrainerLikeCanvasType(sourceNode?.type)) {
          if (sh != null && sh !== "" && sh !== "observable_results") return false;
          return true;
        }
        return analysisTensorFeedLikeTensorViz(sourceNode, sh);
      }

      if (targetNode?.type === "dimension_permutator") {
        if (th !== "tensor_in") return false;
        if (isAnalysisTensorOutSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "statistics" && sh === "tensor") return true;
        if (sourceNode?.type === "statistics2" && sh === "tensor") return true;
        if (isTensorMultiInputSource(sourceNode, sh)) return true;
        if (isTensorConstantOrGeneratedSource(sourceNode, sh)) return true;
        if (isScalarAnalysisTensorSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "tensor_selector" && isTensorSelectorSourceHandle(sh)) return true;
        if (tensorVizTypes.has(String(sourceNode?.type)) && sh === "out_tensor") return true;
        if (sourceNode?.type === "training_visualization" && sh === "out_tensor_list") return true;
        if (
          isTrainerLikeCanvasType(sourceNode?.type) &&
          (sh === "loss_results" || sh === "observable_results")
        ) {
          return true;
        }
        if (observableVizAllowsTensorVizChain(sourceNode) && sh === "out_tensor") return true;
        if (sourceNode?.type === "dimension_permutator" && sh === "tensor_out") return true;
        if (atomicOrChainStripTensorSourceOut(sourceNode, sh)) return true;
        if (fullModelTensorIoSource(sourceNode, sh)) return true;
        if (fullModelModelSocket(sourceNode, sh)) return true;
        if (combinedModelModelModeOut(sourceNode, sh)) return true;
        if (combinedModelTensorIoSource(sourceNode, sh)) return true;
        return false;
      }

      if (targetNode?.type === "tensor_slicing" || targetNode?.type === "elementwise_transform") {
        if (th !== "tensor") return false;
        if (isAnalysisTensorOutSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "statistics" && sh === "tensor") return true;
        if (sourceNode?.type === "statistics2" && sh === "tensor") return true;
        if (isTensorMultiInputSource(sourceNode, sh)) return true;
        if (isTensorConstantOrGeneratedSource(sourceNode, sh)) return true;
        if (isScalarAnalysisTensorSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "tensor_selector" && isTensorSelectorSourceHandle(sh)) return true;
        if (tensorVizTypes.has(String(sourceNode?.type)) && sh === "out_tensor") return true;
        if (sourceNode?.type === "training_visualization" && sh === "out_tensor_list") return true;
        if (
          isTrainerLikeCanvasType(sourceNode?.type) &&
          (sh === "loss_results" || sh === "observable_results")
        ) {
          return true;
        }
        if (observableVizAllowsTensorVizChain(sourceNode) && sh === "out_tensor") return true;
        if (sourceNode?.type === "dimension_permutator" && sh === "tensor_out") return true;
        if (
          (sourceNode?.type === "tensor_slicing" ||
            sourceNode?.type === "elementwise_transform" ||
            sourceNode?.type === "smoothing_curve" ||
            sourceNode?.type === "derivative_curve") &&
          sh === "tensor"
        ) {
          return true;
        }
        if (atomicOrChainStripTensorSourceOut(sourceNode, sh)) return true;
        if (fullModelTensorIoSource(sourceNode, sh)) return true;
        if (fullModelModelSocket(sourceNode, sh)) return true;
        if (combinedModelModelModeOut(sourceNode, sh)) return true;
        if (combinedModelTensorIoSource(sourceNode, sh)) return true;
        return false;
      }

      if (targetNode?.type === "shape_checker" || targetNode?.type === "tensor_reader") {
        if (th !== "tensor" && th !== "") return false;
        if (sourceNode?.type === "activation" && sh === "tensor_list") return true;
        if (sourceNode?.type === "tensor_selector" && isTensorSelectorSourceHandle(sh)) return true;
        if (sourceNode?.type === "training_visualization" && sh === "out_tensor_list") return true;
        if (
          isTrainerLikeCanvasType(sourceNode?.type) &&
          (sh === "loss_results" || sh === "observable_results")
        ) {
          return true;
        }
        if (observableVizAllowsTensorVizChain(sourceNode) && sh === "out_tensor") return true;
        if (isAnalysisTensorOutSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "statistics" && sh === "tensor") return true;
        if (sourceNode?.type === "statistics2" && sh === "tensor") return true;
        if (isTensorMultiInputSource(sourceNode, sh)) return true;
        if (isTensorConstantOrGeneratedSource(sourceNode, sh)) return true;
        if (isScalarAnalysisTensorSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "dimension_permutator" && sh === "tensor_out") return true;
        if (
          (sourceNode?.type === "tensor_slicing" ||
            sourceNode?.type === "elementwise_transform" ||
            sourceNode?.type === "smoothing_curve" ||
            sourceNode?.type === "derivative_curve") &&
          sh === "tensor"
        ) {
          return true;
        }
        if (tensorVizTypes.has(String(sourceNode?.type)) && sh === "out_tensor") return true;
        if (atomicOrChainStripTensorSourceOut(sourceNode, sh)) return true;
        if (fullModelTensorIoSource(sourceNode, sh)) return true;
        if (fullModelModelSocket(sourceNode, sh)) return true;
        if (combinedModelModelModeOut(sourceNode, sh)) return true;
        if (combinedModelTensorIoSource(sourceNode, sh)) return true;
        return false;
      }

      // 声明式 ports 判定(首批:observable_user/dataset_mixer/_b/
      // input_sampler——四个 return-style 分支已删,由 defs 的 ports 声明接管)。
      {
        const portsVerdict = generatedInPortsVerdict(targetNode, th, sourceNode, sh);
        if (portsVerdict != null) return portsVerdict;
      }

      if (
        sourceNode &&
        tensorVizTypes.has(String(sourceNode.type)) &&
        sh === "out_tensor"
      ) {
        if ((targetNode?.type === "pca" || targetNode?.type === "svd") && th === "tensor") return true;
        if (targetNode?.type === "statistics" && th === "tensor") return true;
        if (acceptsTensorMultiInputTarget(targetNode, th)) return true;
        if (isScalarAnalysisTensorSource(targetNode, th)) return true;
        if (targetNode?.type === "dimension_permutator" && th === "tensor_in") return true;
        if ((targetNode?.type === "tensor_slicing" || targetNode?.type === "elementwise_transform") && th === "tensor") return true;
        if (combinedModelTensorIoTarget(targetNode, th)) return true;
      }

      if (targetNode?.type === "tensor_selector") {
        if (th !== "tensor_list") return false;
        if (sourceNode?.type === "activation" && sh === "tensor_list") return true;
        if (sourceNode?.type === "training_visualization" && sh === "out_tensor_list") return true;
        if (observableVizAllowsTensorVizChain(sourceNode) && sh === "out_tensor") return true;
        if (isDatasetTensorListSource(sourceNode, sh)) return true;
        if (
          isTrainerLikeCanvasType(sourceNode?.type) &&
          (sh === "loss_results" || sh === "observable_results")
        ) {
          return true;
        }
        if (sourceNode?.type === "table_viz" && sh === "tensor") {
          return sourceNode.id
            ? tableVizTensorListChoices(nodes, edges, sourceNode.id).length >= 1
            : false;
        }
        if (sourceNode?.type === "model_weight_tensors" && sh === "tensor_list") return true;
        return false;
      }

      if (targetNode?.type === "pca" || targetNode?.type === "svd") {
        if (th !== "tensor") return false;
        if (sourceNode?.type === "activation" && sh === "tensor_list") return true;
        /* Tensor selector exposes only one source handle; allow empty id for RF compatibility. */
        if (sourceNode?.type === "tensor_selector" && isTensorSelectorSourceHandle(sh)) {
          return true;
        }
        if (sourceNode?.type === "dimension_permutator" && sh === "tensor_out") return true;
        if (
          (sourceNode?.type === "tensor_slicing" ||
            sourceNode?.type === "elementwise_transform" ||
            sourceNode?.type === "smoothing_curve" ||
            sourceNode?.type === "derivative_curve") &&
          sh === "tensor"
        ) {
          return true;
        }
        if (atomicOrChainStripTensorSourceOut(sourceNode, sh)) return true;
        if (fullModelTensorIoSource(sourceNode, sh)) return true;
        if (combinedModelModelModeOut(sourceNode, sh)) return true;
        if (combinedModelTensorIoSource(sourceNode, sh)) return true;
        return false;
      }

      if (targetNode?.type === "statistics") {
        if (th !== "tensor") return false;
        if (sourceNode?.type === "activation" && sh === "tensor_list") return true;
        if (sourceNode?.type === "tensor_selector" && isTensorSelectorSourceHandle(sh)) {
          return true;
        }
        if (sourceNode?.type === "training_visualization" && sh === "out_tensor_list") {
          return true;
        }
        if (isAnalysisTensorOutSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "statistics" && sh === "tensor") {
          return true;
        }
        if (sourceNode?.type === "statistics2" && sh === "tensor") {
          return true;
        }
        if (isTensorMultiInputSource(sourceNode, sh)) {
          return true;
        }
        if (isTensorConstantOrGeneratedSource(sourceNode, sh)) {
          return true;
        }
        if (sourceNode?.type === "dimension_permutator" && sh === "tensor_out") return true;
        if ((sourceNode?.type === "tensor_slicing" || sourceNode?.type === "elementwise_transform") && sh === "tensor") return true;
        if (atomicOrChainStripTensorSourceOut(sourceNode, sh)) return true;
        if (fullModelTensorIoSource(sourceNode, sh)) return true;
        if (combinedModelModelModeOut(sourceNode, sh)) return true;
        if (combinedModelTensorIoSource(sourceNode, sh)) return true;
        return false;
      }

      if (targetNode?.type === "statistics2") {
        if (!acceptsTensorMultiInputTarget(targetNode, th)) return false;
        if (sourceNode?.type === "activation" && sh === "tensor_list") return true;
        if (sourceNode?.type === "tensor_selector" && isTensorSelectorSourceHandle(sh)) {
          return true;
        }
        if (sourceNode?.type === "training_visualization" && sh === "out_tensor_list") {
          return true;
        }
        if (isAnalysisTensorOutSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "statistics" && sh === "tensor") return true;
        if (sourceNode?.type === "statistics2" && sh === "tensor") return true;
        if (isTensorMultiInputSource(sourceNode, sh)) return true;
        if (isTensorConstantOrGeneratedSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "dimension_permutator" && sh === "tensor_out") return true;
        if ((sourceNode?.type === "tensor_slicing" || sourceNode?.type === "elementwise_transform") && sh === "tensor") return true;
        if (atomicOrChainStripTensorSourceOut(sourceNode, sh)) return true;
        if (fullModelTensorIoSource(sourceNode, sh)) return true;
        if (combinedModelModelModeOut(sourceNode, sh)) return true;
        if (combinedModelTensorIoSource(sourceNode, sh)) return true;
        return false;
      }

      if (
        targetNode &&
        (tensorMultiInputTypes.has(String(targetNode.type)) || targetNode.type === "tensor_viz_scatter")
      ) {
        if (!acceptsTensorMultiInputTarget(targetNode, th)) return false;
        if (sourceNode?.type === "activation" && sh === "tensor_list") return true;
        if (sourceNode?.type === "tensor_selector" && isTensorSelectorSourceHandle(sh)) {
          return true;
        }
        if (sourceNode?.type === "training_visualization" && sh === "out_tensor_list") {
          return true;
        }
        if (
          isTrainerLikeCanvasType(sourceNode?.type) &&
          (sh === "loss_results" || sh === "observable_results")
        ) {
          return true;
        }
        if (observableVizAllowsTensorVizChain(sourceNode) && sh === "out_tensor") return true;
        if (isAnalysisTensorOutSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "statistics" && sh === "tensor") return true;
        if (sourceNode?.type === "statistics2" && sh === "tensor") return true;
        if (isTensorMultiInputSource(sourceNode, sh)) return true;
        if (isTensorConstantOrGeneratedSource(sourceNode, sh)) return true;
        if (isScalarAnalysisTensorSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "dimension_permutator" && sh === "tensor_out") return true;
        if (
          (sourceNode?.type === "tensor_slicing" ||
            sourceNode?.type === "elementwise_transform" ||
            sourceNode?.type === "smoothing_curve" ||
            sourceNode?.type === "derivative_curve") &&
          sh === "tensor"
        ) {
          return true;
        }
        if (sourceNode?.type === "table_viz" && sh === "tensor") {
          return sourceNode.id ? tableVizTensorConnectable(nodes, edges, sourceNode.id) : false;
        }
        if (atomicOrChainStripTensorSourceOut(sourceNode, sh)) return true;
        if (fullModelTensorIoSource(sourceNode, sh)) return true;
        if (combinedModelModelModeOut(sourceNode, sh)) return true;
        if (combinedModelTensorIoSource(sourceNode, sh)) return true;
        return false;
      }

      if (
        targetNode?.type === "effective_rank" ||
        targetNode?.type === "series_endpoint_gap" ||
        targetNode?.type === "smoothing_curve" ||
        targetNode?.type === "derivative_curve"
      ) {
        if (th !== "tensor") return false;
        if (sourceNode?.type === "tensor_selector" && isTensorSelectorSourceHandle(sh)) return true;
        if (sourceNode?.type === "training_visualization" && sh === "out_tensor_list") return true;
        if (observableVizAllowsTensorVizChain(sourceNode) && sh === "out_tensor") return true;
        if (isAnalysisTensorOutSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "statistics" && sh === "tensor") return true;
        if (sourceNode?.type === "statistics2" && sh === "tensor") return true;
        if (isTensorMultiInputSource(sourceNode, sh)) return true;
        if (isTensorConstantOrGeneratedSource(sourceNode, sh)) return true;
        if (sourceNode?.type === "dimension_permutator" && sh === "tensor_out") return true;
        if (
          (sourceNode?.type === "tensor_slicing" ||
            sourceNode?.type === "elementwise_transform" ||
            sourceNode?.type === "smoothing_curve" ||
            sourceNode?.type === "derivative_curve") &&
          sh === "tensor"
        ) {
          return true;
        }
        if (atomicOrChainStripTensorSourceOut(sourceNode, sh)) return true;
        if (fullModelTensorIoSource(sourceNode, sh)) return true;
        if (combinedModelModelModeOut(sourceNode, sh)) return true;
        if (combinedModelTensorIoSource(sourceNode, sh)) return true;
        return false;
      }

      if (targetNode?.type === "model_weight_tensors") {
        if (th === "model") {
          return (
            fullModelModelSocket(sourceNode, sh) ||
            fullModelTensorIoSource(sourceNode, sh) ||
            combinedModelModelModeOut(sourceNode, sh) ||
            (sourceNode?.type === "linear_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "activation_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "layer_norm_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "rms_norm_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "embedding_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "unembedding_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "absolute_pos_embed_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "rotary_embed_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "local_mixing_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "afno_patch_embed_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "afno_spectral_mixer_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "afno_encoder_block_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "afno_patch_decode_layer" && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            combinedModelTensorIoSource(sourceNode, sh) ||
            (sourceNode?.type === "model_checkpoint" && sh === "model")
          );
        }
        return false;
      }

      if (targetNode?.type === "visualize_kan") {
        if (th === "model") {
          if (sourceNode?.type === "kan_model") {
            return Boolean(
              fullModelModelSocket(sourceNode, sh) || fullModelTensorIoSource(sourceNode, sh),
            );
          }
          if (sourceNode?.type === "model_checkpoint" && sh === "model") return true;
          return false;
        }
        if (th === "dataset") {
          return (
            (sourceNode?.type === "linear_dataset" ||
              sourceNode?.type === "random_noise_dataset" ||
              sourceNode?.type === "memorization_a_dataset" ||
              sourceNode?.type === "memorization_b_dataset" ||
              sourceNode?.type === "symbolic_func_dataset" ||
              sourceNode?.type === "teacher_dataset" ||
              sourceNode?.type === "uniform_linear_motion_dataset" ||
              sourceNode?.type === "kepler_2d_dataset" ||
              sourceNode?.type === "diffusion_pde_dataset" ||
              sourceNode?.type === "reaction_diffusion_dataset" ||
              sourceNode?.type === "advection_dataset" ||
              sourceNode?.type === "dataset_mixer" ||
              sourceNode?.type === "dataset_mixer_b") &&
            (sh === "dataset" || sh === "train_dataset" || sh === "test_dataset")
          );
        }
        return false;
      }

      if (targetNode?.type === "prediction") {
        if (th === "model") {
          return (
            fullModelModelSocket(sourceNode, sh) ||
            fullModelTensorIoSource(sourceNode, sh) ||
            combinedModelModelModeOut(sourceNode, sh) ||
            combinedModelTensorIoSource(sourceNode, sh) ||
            (sourceNode != null && atomicLayerTypes.has(String(sourceNode.type)) && atomicLayerTensorSourceOut(sourceNode, sh)) ||
            (sourceNode?.type === "model_checkpoint" && sh === "model")
          );
        }
        if (th === "dataset") {
          return isDatasetTensorListSource(sourceNode, sh);
        }
        return false;
      }

      if (
        (targetNode && atomicLayerTypes.has(String(targetNode.type)) && atomicLayerTensorIoTarget(targetNode, th)) ||
        (targetNode && layerStripChainTensorIoTarget(targetNode, th)) ||
        combinedModelTensorIoTarget(targetNode, th) ||
        fullModelTensorIoTarget(targetNode, th)
      ) {
        if (!isLayerStripTargetHandle(th)) return false;
        if (!sourceNode) return false;
        return layerStripTensorInSource(sourceNode, sh);
      }

      if (targetNode?.type === "activation") {
        if (th === "model") {
          return (
            fullModelModelSocket(sourceNode, sh) ||
            fullModelTensorIoSource(sourceNode, sh) ||
            combinedModelModelModeOut(sourceNode, sh) ||
            (sourceNode?.type === "model_checkpoint" && sh === "model") ||
            (isTrainerLikeCanvasType(sourceNode?.type) && sh === "checkpoint")
          );
        }
        if (th === "dataset") {
          return (
            (sourceNode?.type === "linear_dataset" ||
              sourceNode?.type === "random_noise_dataset" ||
              sourceNode?.type === "memorization_a_dataset" ||
              sourceNode?.type === "memorization_b_dataset" ||
              sourceNode?.type === "symbolic_func_dataset" ||
              sourceNode?.type === "teacher_dataset" ||
              sourceNode?.type === "uniform_linear_motion_dataset" ||
              sourceNode?.type === "kepler_2d_dataset" ||
              sourceNode?.type === "diffusion_pde_dataset" ||
              sourceNode?.type === "reaction_diffusion_dataset" ||
              sourceNode?.type === "advection_dataset" ||
              sourceNode?.type === "dataset_mixer" ||
              sourceNode?.type === "dataset_mixer_b") &&
            (sh === "dataset" || sh === "train_dataset" || sh === "test_dataset")
          );
        }
        return false;
      }

      if (sourceNode?.type === "random_input_distribution") {
        // 后 target=input_sampler 已由 ports verdict 提前接管——此 mirror
        // 分支保留作旧 cascade 对照/后续清理参考,实际不再决定该路径。
        return targetNode?.type === "input_sampler" && th === "distribution" && sh === "input_distribution";
      }

      if (sourceNode?.type === "input_sampler") {
        if (sh !== "sample_tensor") return false;
        // 后 target=teacher_dataset 已由 ports verdict 提前接管——本行仅为
        // input_sampler 源块的 return-style 尾巴(非 teacher 目标一律 false),
        // teacher 路径实际不再到达此处；保留该镜像分支。
        return targetNode?.type === "teacher_dataset" && (th === "train_input" || th === "test_input");
      }

      if (sourceNode?.type === "tensor_selector") {
        if (!isTensorSelectorSourceHandle(sh)) return false;
        if ((targetNode?.type === "pca" || targetNode?.type === "svd") && th === "tensor") return true;
        if (targetNode?.type === "statistics" && th === "tensor") return true;
        if (acceptsTensorMultiInputTarget(targetNode, th)) return true;
        if (isScalarAnalysisTensorSource(targetNode, th)) return true;
        if (
          (targetNode?.type === "shape_checker" || targetNode?.type === "tensor_reader") &&
          (th === "tensor" || th === "")
        ) {
          return true;
        }
        return (
          (tensorVizTypes.has(String(targetNode?.type)) && th === "tensor") ||
          (isObservableVizFlowType(String(targetNode?.type)) && th === "tensor") ||
          (targetNode?.type === "dimension_permutator" && th === "tensor_in") ||
          ((targetNode?.type === "tensor_slicing" || targetNode?.type === "elementwise_transform") && th === "tensor") ||
          combinedModelTensorIoTarget(targetNode, th)
        );
      }

      if (sourceNode?.type === "dimension_permutator") {
        if (sh !== "tensor_out") return false;
        if ((targetNode?.type === "pca" || targetNode?.type === "svd") && th === "tensor") return true;
        if (targetNode?.type === "statistics" && th === "tensor") return true;
        if (acceptsTensorMultiInputTarget(targetNode, th)) return true;
        if (isScalarAnalysisTensorSource(targetNode, th)) return true;
        if (tensorVizTypes.has(String(targetNode?.type)) && th === "tensor") return true;
        if (isObservableVizFlowType(String(targetNode?.type)) && th === "tensor") return true;
        if (targetNode?.type === "dimension_permutator" && th === "tensor_in") return true;
        if ((targetNode?.type === "tensor_slicing" || targetNode?.type === "elementwise_transform") && th === "tensor") return true;
        if (targetNode?.type === "regressor" && th === "tensor") return true;
        if (combinedModelTensorIoTarget(targetNode, th)) return true;
        return false;
      }

      if (
        sourceNode?.type === "effective_rank" ||
        sourceNode?.type === "series_endpoint_gap" ||
        sourceNode?.type === "smoothing_curve" ||
        sourceNode?.type === "derivative_curve"
      ) {
        if (sh !== "tensor") return false;
        if ((targetNode?.type === "pca" || targetNode?.type === "svd") && th === "tensor") return true;
        if (tensorVizTypes.has(String(targetNode?.type)) && th === "tensor") return true;
        if (isObservableVizFlowType(String(targetNode?.type)) && th === "tensor") return true;
        if (targetNode?.type === "statistics" && th === "tensor") return true;
        if (acceptsTensorMultiInputTarget(targetNode, th)) return true;
        if (targetNode?.type === "dimension_permutator" && th === "tensor_in") return true;
        if ((targetNode?.type === "tensor_slicing" || targetNode?.type === "elementwise_transform") && th === "tensor") return true;
        if (combinedModelTensorIoTarget(targetNode, th)) return true;
        return false;
      }

      if (sourceNode?.type === "statistics") {
        if (sh !== "tensor") return false;
        if ((targetNode?.type === "pca" || targetNode?.type === "svd") && th === "tensor") return true;
        if (tensorVizTypes.has(String(targetNode?.type)) && th === "tensor") return true;
        if (isObservableVizFlowType(String(targetNode?.type)) && th === "tensor") return true;
        if (targetNode?.type === "statistics" && th === "tensor") return true;
        if (acceptsTensorMultiInputTarget(targetNode, th)) return true;
        if (targetNode?.type === "dimension_permutator" && th === "tensor_in") return true;
        if ((targetNode?.type === "tensor_slicing" || targetNode?.type === "elementwise_transform") && th === "tensor") return true;
        if (combinedModelTensorIoTarget(targetNode, th)) return true;
        return false;
      }

      if (sourceNode?.type === "statistics2") {
        if (sh !== "tensor") return false;
        if ((targetNode?.type === "pca" || targetNode?.type === "svd") && th === "tensor") return true;
        if (tensorVizTypes.has(String(targetNode?.type)) && th === "tensor") return true;
        if (isObservableVizFlowType(String(targetNode?.type)) && th === "tensor") return true;
        if (targetNode?.type === "statistics" && th === "tensor") return true;
        if (acceptsTensorMultiInputTarget(targetNode, th)) return true;
        if (isScalarAnalysisTensorSource(targetNode, th)) return true;
        if (targetNode?.type === "regressor" && th === "tensor") return true;
        if (targetNode?.type === "dimension_permutator" && th === "tensor_in") return true;
        if ((targetNode?.type === "tensor_slicing" || targetNode?.type === "elementwise_transform") && th === "tensor") return true;
        if (combinedModelTensorIoTarget(targetNode, th)) return true;
        return false;
      }

      if (sourceNode && tensorMultiInputTypes.has(String(sourceNode.type))) {
        if (sh !== "tensor") return false;
        if ((targetNode?.type === "pca" || targetNode?.type === "svd") && th === "tensor") return true;
        if (tensorVizTypes.has(String(targetNode?.type)) && th === "tensor") return true;
        if (isObservableVizFlowType(String(targetNode?.type)) && th === "tensor") return true;
        if (targetNode?.type === "statistics" && th === "tensor") return true;
        if (acceptsTensorMultiInputTarget(targetNode, th)) return true;
        if (isScalarAnalysisTensorSource(targetNode, th)) return true;
        if (targetNode?.type === "regressor" && th === "tensor") return true;
        if (targetNode?.type === "dimension_permutator" && th === "tensor_in") return true;
        if ((targetNode?.type === "tensor_slicing" || targetNode?.type === "elementwise_transform") && th === "tensor") return true;
        if (combinedModelTensorIoTarget(targetNode, th)) return true;
        return false;
      }

      if (isTensorConstantOrGeneratedSource(sourceNode, sh)) {
        if ((targetNode?.type === "pca" || targetNode?.type === "svd") && th === "tensor") return true;
        if (tensorVizTypes.has(String(targetNode?.type)) && th === "tensor") return true;
        if (isObservableVizFlowType(String(targetNode?.type)) && th === "tensor") return true;
        if (targetNode?.type === "statistics" && th === "tensor") return true;
        if (acceptsTensorMultiInputTarget(targetNode, th)) return true;
        if (isScalarAnalysisTensorSource(targetNode, th)) return true;
        if (targetNode?.type === "regressor" && th === "tensor") return true;
        if (targetNode?.type === "dimension_permutator" && th === "tensor_in") return true;
        if ((targetNode?.type === "tensor_slicing" || targetNode?.type === "elementwise_transform") && th === "tensor") return true;
        if (combinedModelTensorIoTarget(targetNode, th)) return true;
        return false;
      }

      if (sourceNode?.type === "activation") {
        if (sh === "tensor_list") {
          return (
            targetNode?.type === "tensor_selector" ||
            targetNode?.type === "pca" ||
            targetNode?.type === "svd" ||
            targetNode?.type === "statistics" ||
            targetNode?.type === "statistics2" ||
            targetNode?.type === "tensor_add" ||
            targetNode?.type === "tensor_stack" ||
            targetNode?.type === "tensor_concat" ||
            targetNode?.type === "basic_calculator"
          );
        }
        return false;
      }

      if (sourceNode?.type === "model_weight_tensors") {
        if (sh === "tensor_list") {
          return targetNode?.type === "tensor_selector" && th === "tensor_list";
        }
        return false;
      }

      const isInitSourceNode = (t: string | undefined) =>
        t === "mup_initialization" ||
        t === "saxe_initialization" ||
        t === "symmetrized_mlp_init" ||
        t === "rank_aligned_initialization" ||
        t === "idnns_initialization";
      if (isInitSourceNode(sourceNode?.type)) {
        if (sh !== "initialization") return false;
        return Boolean(
          targetNode &&
            modelInitializationTargetTypes.has(String(targetNode.type)) &&
            th === "initialization",
        );
      }

      if (
        targetNode &&
        modelInitializationTargetTypes.has(String(targetNode.type)) &&
        th === "initialization"
      ) {
        return isInitSourceNode(sourceNode?.type) && sh === "initialization";
      }

      if (targetNode?.type === "trainer" || targetNode?.type === "crl_trainer") {
        if (!sh || !th) return false;
        if (th === "observables" && (sh === "observable" || sh === "observables")) return true;
        if (th === "env") {
          return targetNode.type === "crl_trainer" && sourceNode?.type === "crl_env_config" && sh === "env";
        }
        if (th === "model") {
          if (targetNode.type === "crl_trainer") {
            return Boolean(sourceNode?.type === "crl_residual_mlp" && sh === "model");
          }
          return Boolean(
            sourceNode &&
              ((atomicLayerTypes.has(String(sourceNode.type)) && atomicLayerTensorSourceOut(sourceNode, sh)) ||
                combinedModelTensorIoSource(sourceNode, sh) ||
                combinedModelModelModeOut(sourceNode, sh) ||
                fullModelTensorIoSource(sourceNode, sh) ||
                fullModelModelSocket(sourceNode, sh)),
          );
        }
        // batch_schedule socket 只收 cyclic_batch_schedule(必须在
        // `sh === th` 尾之前收紧,否则任意 batch_schedule↔batch_schedule 放行)。
        if (th === "batch_schedule") {
          return sourceNode?.type === "cyclic_batch_schedule" && sh === "batch_schedule";
        }
        return sh === th;
      }

      const optimizerCanvasTypes = OPTIMIZER_CANVAS_CONN_TYPES;
      if (targetNode && optimizerCanvasTypes.has(String(targetNode.type))) {
        if (th === "lr_schedule") {
          return PLAIN_LR_SCHEDULE_SOURCE_TYPES.has(String(sourceNode?.type)) && sh === "lr_schedule";
        }
        if (th === "mup_lr_schedule") {
          return sourceNode?.type === "mup_lr_schedule" && sh === "mup_lr_schedule";
        }
        if (th === "optimizer_lr_schedule") {
          return (
            (PLAIN_LR_SCHEDULE_SOURCE_TYPES.has(String(sourceNode?.type)) && sh === "lr_schedule") ||
            (sourceNode?.type === "mup_lr_schedule" && sh === "mup_lr_schedule")
          );
        }
      }

      /* Outgoing model handle: only valid target is Trainer `model`, handled above. */
      if (sourceNode?.type === "model_checkpoint") {
        return false;
      }

      const vizCommentSources = VIZ_COMMENT_SOURCE_TYPES;

      // 后 target=comment 已由 ports verdict 提前接管——本块仅为
      // comment-源(canvas_comment_source × sh=comment)到非 comment 目标的
      // return-style 尾巴一律为 false；comment 路径实际不再到达。
      if (sh === "comment" && sourceNode != null && vizCommentSources.has(String(sourceNode.type))) {
        return targetNode?.type === "comment" && th === "comment";
      }

      if (targetNode?.type === "regressor") {
        if (th !== "tensor") return false;
        if (sourceNode?.type === "tensor_selector" && isTensorSelectorSourceHandle(sh)) return true;
        if (sourceNode?.type === "training_visualization" && sh === "out_tensor_list") return true;
        if (isScalarAnalysisTensorSource(sourceNode, sh)) return true;
        if (
          (sourceNode?.type === "observable_viz" || sourceNode?.type === "observable_accuracy") &&
          sh === "out_tensor"
        ) {
          return true;
        }
        if (sourceNode?.type === "statistics2" && sh === "tensor") return true;
        if (isTensorMultiInputSource(sourceNode, sh)) return true;
        if (isTensorConstantOrGeneratedSource(sourceNode, sh)) return true;
        if (isTrainerLikeCanvasType(sourceNode?.type) && (sh === "loss_results" || sh === "observable_results"))
          return true;
        if (sourceNode?.type === "table_viz" && sh === "tensor") {
          return sourceNode.id ? tableVizTensorConnectable(nodes, edges, sourceNode.id) : false;
        }
        if (sourceNode?.type === "dimension_permutator" && sh === "tensor_out") return true;
        if ((sourceNode?.type === "tensor_slicing" || sourceNode?.type === "elementwise_transform") && sh === "tensor") return true;
        if (atomicOrChainStripTensorSourceOut(sourceNode, sh)) return true;
        if (fullModelTensorIoSource(sourceNode, sh)) return true;
        if (combinedModelModelModeOut(sourceNode, sh)) return true;
        if (combinedModelTensorIoSource(sourceNode, sh)) return true;
        if (tensorVizTypes.has(String(sourceNode?.type)) && sh === "out_tensor") {
          return false;
        }
        return false;
      }

      if (
        targetNode &&
        atomicLayerTypes.has(String(targetNode.type)) &&
        isLayerStripTargetHandle(th) &&
        !atomicLayerTensorIoTarget(targetNode, th)
      ) {
        return false;
      }
      if (
        targetNode &&
        fullModelCanvasTypes.has(String(targetNode.type)) &&
        isLayerStripTargetHandle(th) &&
        !fullModelTensorIoTarget(targetNode, th)
      ) {
        return false;
      }
      if (
        targetNode?.type === "combined_model" &&
        isLayerStripTargetHandle(th) &&
        !combinedModelTensorIoTarget(targetNode, th)
      ) {
        return false;
      }
      if (
        targetNode?.type === "combined_model" &&
        th === COMBINED_MODEL_RETURN_TARGET_HANDLE &&
        !combinedModelTensorReturnTarget(targetNode, th)
      ) {
        return false;
      }
      if (
        sourceNode &&
        fullModelCanvasTypes.has(String(sourceNode.type)) &&
        sh === "model" &&
        !fullModelModelSocket(sourceNode, sh)
      ) {
        return false;
      }
      if (
        sourceNode &&
        fullModelCanvasTypes.has(String(sourceNode.type)) &&
        isLayerStripSourceHandle(sh) &&
        !fullModelTensorIoSource(sourceNode, sh)
      ) {
        return false;
      }
      if (
        sourceNode &&
        atomicLayerTypes.has(String(sourceNode.type)) &&
        isLayerStripSourceHandle(sh) &&
        !atomicLayerTensorSourceOut(sourceNode, sh)
      ) {
        return false;
      }
      if (
        sourceNode?.type === "combined_model" &&
        isLayerStripSourceHandle(sh) &&
        !combinedModelTensorIoSource(sourceNode, sh) &&
        !combinedModelModelModeOut(sourceNode, sh)
      ) {
        return false;
      }

      if (targetNode?.type === "protein_structure_displayer") {
        return (th ?? "").trim() === "coords";
      }
      if (targetNode?.type === "protein_structure_comparison_viz") {
        return (th ?? "").trim() === "pred_coords" || (th ?? "").trim() === "true_coords";
      }

      return true;
}

export function planAutoConnectCanvas(nodes: Node[], edges: Edge[]): Connection[] {
    const trainerNodes = nodes.filter((n) => isTrainerLikeCanvasType(String(n.type)));
    if (!trainerNodes.length) return [];
    const datasetNodeTypes = AUTOCONNECT_DATASET_TYPES;
    const optimizerNodeTypes = OPTIMIZER_CANVAS_CONN_TYPES;
    const lossNodeTypes = PRIMARY_LOSS_CONN_TYPES;
    const weightRegLossNodeTypes = LOSS_SOCKET_AUX_CONN_TYPES;
    const modelNodeTypes = AUTOCONNECT_MODEL_TYPES;

    const edgeExists = (
      source: string,
      target: string,
      sourceHandle: string | null,
      targetHandle: string | null,
      pool: Edge[],
    ) =>
      pool.some(
        (e) =>
          e.source === source &&
          e.target === target &&
          (e.sourceHandle ?? null) === sourceHandle &&
          (e.targetHandle ?? null) === targetHandle,
      );

    const distanceSq = (a: Node, b: Node) => {
      const dx = a.position.x - b.position.x;
      const dy = a.position.y - b.position.y;
      return dx * dx + dy * dy;
    };

    const chooseNearestConnectable = (
      trainer: Node,
      sourceHandles: string[],
      targetHandle: string,
      nodeFilter?: (node: Node) => boolean,
    ): { node: Node; sourceHandle: string } | null => {
      let best: { node: Node; sourceHandle: string } | null = null;
      let bestDist = Number.POSITIVE_INFINITY;
      for (const candidate of nodes) {
        if (candidate.id === trainer.id) continue;
        if (nodeFilter && !nodeFilter(candidate)) continue;
        for (const sh of sourceHandles) {
          const valid = isValidCanvasConnection({
            source: candidate.id,
            target: trainer.id,
            sourceHandle: sh,
            targetHandle,
          }, nodes, edges);
          if (!valid) continue;
          if (edgeExists(candidate.id, trainer.id, sh, targetHandle, edges)) continue;
          const d = distanceSq(candidate, trainer);
          if (d < bestDist) {
            bestDist = d;
            best = { node: candidate, sourceHandle: sh };
          }
        }
      }
      return best;
    };

    const plannedConnections: Connection[] = [];
    const connectionPlanned = (
      source: string,
      target: string,
      sourceHandle: string | null,
      targetHandle: string | null,
    ) =>
      edgeExists(source, target, sourceHandle, targetHandle, edges) ||
      plannedConnections.some(
        (c) =>
          c.source === source &&
          c.target === target &&
          (c.sourceHandle ?? null) === sourceHandle &&
          (c.targetHandle ?? null) === targetHandle,
      );

    for (const trainer of trainerNodes) {
      const trainerId = trainer.id;
      const isCrl = trainer.type === "crl_trainer";
      const hasIncoming = (targetHandle: string) =>
        edges.some((e) => e.target === trainerId && (e.targetHandle ?? "") === targetHandle);
      const hasPrimaryLossIncoming = () =>
        edges.some((e) => {
          if (e.target !== trainerId || (e.targetHandle ?? "") !== "loss") return false;
          const src = nodes.find((n) => n.id === e.source);
          return src != null && lossNodeTypes.has(String(src.type));
        }) ||
        plannedConnections.some((c) => {
          if (c.target !== trainerId || (c.targetHandle ?? "") !== "loss") return false;
          const src = nodes.find((n) => n.id === c.source);
          return src != null && lossNodeTypes.has(String(src.type));
        });

      if (!hasIncoming("model")) {
        const pick = chooseNearestConnectable(
          trainer,
          ["model"],
          "model",
          isCrl ? (n) => n.type === "crl_residual_mlp" : (n) => modelNodeTypes.has(String(n.type)),
        );
        if (pick) {
          plannedConnections.push({
            source: pick.node.id,
            target: trainerId,
            sourceHandle: pick.sourceHandle,
            targetHandle: "model",
          });
        }
      }
      if (!hasIncoming("optimizer")) {
        const pick = chooseNearestConnectable(
          trainer,
          ["optimizer"],
          "optimizer",
          (n) => optimizerNodeTypes.has(String(n.type)),
        );
        if (pick) {
          plannedConnections.push({
            source: pick.node.id,
            target: trainerId,
            sourceHandle: pick.sourceHandle,
            targetHandle: "optimizer",
          });
        }
      }
      if (isCrl) {
        if (!hasIncoming("env")) {
          const pick = chooseNearestConnectable(trainer, ["env"], "env", (n) => n.type === "crl_env_config");
          if (pick) {
            plannedConnections.push({
              source: pick.node.id,
              target: trainerId,
              sourceHandle: pick.sourceHandle,
              targetHandle: "env",
            });
          }
        }
      } else {
        if (!hasPrimaryLossIncoming()) {
          const pick = chooseNearestConnectable(
            trainer,
            ["loss"],
            "loss",
            (n) => lossNodeTypes.has(String(n.type)),
          );
          if (pick) {
            plannedConnections.push({
              source: pick.node.id,
              target: trainerId,
              sourceHandle: pick.sourceHandle,
              targetHandle: "loss",
            });
          }
        }
        if (!hasIncoming("dataset")) {
          const pick = chooseNearestConnectable(
            trainer,
            ["dataset"],
            "dataset",
            (n) => datasetNodeTypes.has(String(n.type)),
          );
          if (pick) {
            plannedConnections.push({
              source: pick.node.id,
              target: trainerId,
              sourceHandle: pick.sourceHandle,
              targetHandle: "dataset",
            });
          }
        }
      }
    }

    for (const regNode of nodes) {
      if (!weightRegLossNodeTypes.has(String(regNode.type))) continue;
      let nearestTrainer: Node | null = null;
      let nearestDist = Number.POSITIVE_INFINITY;
      for (const trainer of trainerNodes) {
        if (trainer.type === "crl_trainer") continue;
        const d = distanceSq(regNode, trainer);
        if (d < nearestDist) {
          nearestDist = d;
          nearestTrainer = trainer;
        }
      }
      if (!nearestTrainer) continue;
      const targetTrainerId = nearestTrainer.id;
      if (connectionPlanned(regNode.id, targetTrainerId, "loss", "loss")) continue;
      const valid = isValidCanvasConnection({
        source: regNode.id,
        target: targetTrainerId,
        sourceHandle: "loss",
        targetHandle: "loss",
      }, nodes, edges);
      if (!valid) continue;
      plannedConnections.push({
        source: regNode.id,
        target: targetTrainerId,
        sourceHandle: "loss",
        targetHandle: "loss",
      });
    }

    for (const obsNode of nodes) {
      if (!String(obsNode.type).startsWith("observable_")) continue;
      if (obsNode.type === "observable_viz") continue;
      let nearestTrainer: Node | null = null;
      let nearestDist = Number.POSITIVE_INFINITY;
      for (const trainer of trainerNodes) {
        const d = distanceSq(obsNode, trainer);
        if (d < nearestDist) {
          nearestDist = d;
          nearestTrainer = trainer;
        }
      }
      if (!nearestTrainer) continue;
      const targetTrainerId = nearestTrainer.id;
      if (edges.some((e) => e.source === obsNode.id && e.target === targetTrainerId && e.targetHandle === "observables")) {
        continue;
      }
      // Trainer accepts either handle name in validation, but the edge must use the node's real
      // source handle id or React Flow will not draw the wire (`ObservableSourceStrip` default is
      // `observables`; `observable_user` uses `observable` only).
      const userObs = obsNode.type === "observable_user";
      const sourceHandle = userObs
        ? isValidCanvasConnection({
              source: obsNode.id,
              target: targetTrainerId,
              sourceHandle: "observable",
              targetHandle: "observables",
            }, nodes, edges)
          ? "observable"
          : null
        : isValidCanvasConnection({
              source: obsNode.id,
              target: targetTrainerId,
              sourceHandle: "observables",
              targetHandle: "observables",
            }, nodes, edges)
          ? "observables"
          : isValidCanvasConnection({
                source: obsNode.id,
                target: targetTrainerId,
                sourceHandle: "observable",
                targetHandle: "observables",
              }, nodes, edges)
            ? "observable"
            : null;
      if (!sourceHandle) continue;
      plannedConnections.push({
        source: obsNode.id,
        target: targetTrainerId,
        sourceHandle,
        targetHandle: "observables",
      });
    }

    return plannedConnections;
}
