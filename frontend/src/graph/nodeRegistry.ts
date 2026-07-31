import type { ComponentType } from "react";
import type { NodeProps, NodeTypes } from "@xyflow/react";
import { GENERATED_NODE_SPECS } from "../generated/generatedNodeSpecs";
import { GenericObservableNode } from "../components/nodes/GenericObservableNode";
import { GenericDatasetNode } from "../components/nodes/GenericDatasetNode";
import { ActivationNode } from "../components/nodes/ActivationNode";
import { SchemaNode } from "../components/nodes/SchemaNode";
import { LrScheduleNode } from "../components/nodes/LrScheduleNode";
import { MupLrScheduleNode } from "../components/nodes/MupLrScheduleNode";
import { CyclicLrScheduleNode } from "../components/nodes/CyclicLrScheduleNode";
import { CyclicBatchScheduleNode } from "../components/nodes/CyclicBatchScheduleNode";
import { KeskarCnnModelNode } from "../components/nodes/KeskarCnnModelNode";
import { Vgg11CifarModelNode } from "../components/nodes/Vgg11CifarModelNode";
import { CurveSeriesTableNode } from "../components/nodes/CurveSeriesTableNode";
import { CurveSeriesVizNode } from "../components/nodes/CurveSeriesVizNode";
import { MetricCompareNode } from "../components/nodes/MetricCompareNode";
import { ParametricPathSamplerNode } from "../components/nodes/ParametricPathSamplerNode";
import { LinearInterpolationBarrierNode } from "../components/nodes/LinearInterpolationBarrierNode";
import { BezierModeConnectivityNode } from "../components/nodes/BezierModeConnectivityNode";
import { MupInitializationNode } from "../components/nodes/MupInitializationNode";
import { IdnnsInitializationNode } from "../components/nodes/IdnnsInitializationNode";
import { SaxeInitializationNode } from "../components/nodes/SaxeInitializationNode";
import { SymmetrizedMlpInitNode } from "../components/nodes/SymmetrizedMlpInitNode";
import { NeuronTrajectory2dVizNode } from "../components/nodes/NeuronTrajectory2dVizNode";
import { LinearDatasetNode } from "../components/nodes/LinearDatasetNode";
import { SymbolicFuncDatasetNode } from "../components/nodes/SymbolicFuncDatasetNode";
import { MlpModelNode } from "../components/nodes/MlpModelNode";
import { GatedMlpModelNode } from "../components/nodes/GatedMlpModelNode";
import { MoeMlpModelNode } from "../components/nodes/MoeMlpModelNode";
import { KanModelNode } from "../components/nodes/KanModelNode";
import { VisualizeKanNode } from "../components/nodes/VisualizeKanNode";
import { LinearLayerNode } from "../components/nodes/LinearLayerNode";
import { ActivationLayerNode } from "../components/nodes/ActivationLayerNode";
import { LayerNormLayerNode } from "../components/nodes/LayerNormLayerNode";
import { RmsNormLayerNode } from "../components/nodes/RmsNormLayerNode";
import { EmbeddingLayerNode } from "../components/nodes/EmbeddingLayerNode";
import { UnembeddingLayerNode } from "../components/nodes/UnembeddingLayerNode";
import { AbsolutePosEmbedLayerNode } from "../components/nodes/AbsolutePosEmbedLayerNode";
import { RotaryEmbedLayerNode } from "../components/nodes/RotaryEmbedLayerNode";
import { LocalMixingLayerNode } from "../components/nodes/LocalMixingLayerNode";
import { ResidualLnModelNode } from "../components/nodes/ResidualLnModelNode";
import { AttentionOnlyModelNode } from "../components/nodes/AttentionOnlyModelNode";
import { AlternativeArchTokenLmNode } from "../components/nodes/AlternativeArchTokenLmNode";
import { DiffusionScoreModelNode } from "../components/nodes/DiffusionScoreModelNode";
import { UnetDdpmModelNode } from "../components/nodes/UnetDdpmModelNode";
import {
  DeterministicDiffusionSamplerNode,
  NearestTrainGlNode,
  PairedGenerationSimilarityNode,
  RpScoreSscdNode,
} from "../components/nodes/DiffusionReproNodes";
import { DiffusionMseLossNode } from "../components/nodes/DiffusionMseLossNode";
import { CombinedModelNode } from "../components/nodes/CombinedModelNode";
import { TokenPredictionDatasetNode } from "../components/nodes/TokenPredictionDatasetNode";
import { CircleRandomWalkDatasetNode } from "../components/nodes/CircleRandomWalkDatasetNode";
import { CircularMotionDatasetNode } from "../components/nodes/CircularMotionDatasetNode";
import { Kepler2dDatasetNode } from "../components/nodes/Kepler2dDatasetNode";
import { UnigramDatasetNode } from "../components/nodes/UnigramDatasetNode";
import { BigramLowRankDatasetNode } from "../components/nodes/BigramLowRankDatasetNode";
import { RandomInputDistributionNode } from "../components/nodes/RandomInputDistributionNode";
import { InputSamplerNode } from "../components/nodes/InputSamplerNode";
import { TeacherDatasetNode } from "../components/nodes/TeacherDatasetNode";
import { InContextAssociativeRecallDatasetNode } from "../components/nodes/InContextAssociativeRecallDatasetNode";
import { UniformLinearMotionDatasetNode } from "../components/nodes/UniformLinearMotionDatasetNode";
import {
  AdvectionDatasetNode,
  DiffusionPdeDatasetNode,
  ReactionDiffusionDatasetNode,
} from "../components/nodes/PdeFieldDatasetNode";
import { ModularAdditionDatasetNode } from "../components/nodes/ModularAdditionDatasetNode";
import { DatasetMixerNode } from "../components/nodes/DatasetMixerNode";
import { DatasetMixerBNode } from "../components/nodes/DatasetMixerBNode";
import { ToyLanguageLmDatasetNode } from "../components/nodes/ToyLanguageLmDatasetNode";
import { VisionDatasetNode } from "../components/nodes/VisionDatasetNode";
import { ResnetModelNode } from "../components/nodes/ResnetModelNode";
import { VitModelNode } from "../components/nodes/VitModelNode";
import { MlpTokenModelNode } from "../components/nodes/MlpTokenModelNode";
import { NumericTransformerModelNode } from "../components/nodes/NumericTransformerModelNode";
import { NumericHyenaModelNode } from "../components/nodes/NumericHyenaModelNode";
import { MppSpatiotemporalModelNode } from "../components/nodes/MppSpatiotemporalModelNode";
import { AfnoLiteSpatiotemporalModelNode } from "../components/nodes/AfnoLiteSpatiotemporalModelNode";
import { AfnoAtomicLayerNode } from "../components/nodes/AfnoAtomicLayerNode";
import { TransformerMultiTokenModelNode } from "../components/nodes/TransformerMultiTokenModelNode";
import { TransformerTokenModelNode } from "../components/nodes/TransformerTokenModelNode";
import { CrossEntropyLossNode } from "../components/nodes/CrossEntropyLossNode";
import { BinaryCrossEntropyWithLogitsLossNode } from "../components/nodes/BinaryCrossEntropyWithLogitsLossNode";
import { PcaNode } from "../components/nodes/PcaNode";
import { SvdNode } from "../components/nodes/SvdNode";
import { StatisticsNode } from "../components/nodes/StatisticsNode";
import { Statistics2Node } from "../components/nodes/Statistics2Node";
import { TensorAddNode } from "../components/nodes/TensorAddNode";
import { TensorStackNode } from "../components/nodes/TensorStackNode";
import { TensorConcatNode } from "../components/nodes/TensorConcatNode";
import { BasicCalculatorNode } from "../components/nodes/BasicCalculatorNode";
import { TensorConstantNode } from "../components/nodes/TensorConstantNode";
import { TensorLinspaceNode } from "../components/nodes/TensorLinspaceNode";
import { ElementwiseTransformNode } from "../components/nodes/ElementwiseTransformNode";
import { FakeTensorNode } from "../components/nodes/FakeTensorNode";
import { ModelCheckpointNode } from "../components/nodes/ModelCheckpointNode";
import { KanRegNode } from "../components/nodes/KanRegNode";
import { L1RegNode } from "../components/nodes/L1RegNode";
import { L2RegNode } from "../components/nodes/L2RegNode";
import { L2ProjectionNode } from "../components/nodes/L2ProjectionNode";
import { MseLossNode } from "../components/nodes/MseLossNode";
import { CrlEnvConfigNode } from "../components/nodes/CrlEnvConfigNode";
import { CrlResidualMlpNode } from "../components/nodes/CrlResidualMlpNode";
import { CrlTrainerNode } from "../components/nodes/CrlTrainerNode";
import { TrainerNode } from "../components/nodes/TrainerNode";
import { ObservableVizNode } from "../components/nodes/ObservableVizNode";
import { TrainingVisualizationNode } from "../components/nodes/TrainingVisualizationNode";
import { ImageDatasetDisplayerNode } from "../components/nodes/ImageDatasetDisplayerNode";
import { ProteinStructureDisplayerNode } from "../components/nodes/ProteinStructureDisplayerNode";
import { ProteinStructureComparisonVizNode } from "../components/nodes/ProteinStructureComparisonVizNode";
import { ObservableUserNode } from "../components/nodes/ObservableUserNode";
import { WeightL2ObservableNode } from "../components/nodes/WeightL2ObservableNode";
import { HessianEigenvaluesObservableNode } from "../components/nodes/HessianEigenvaluesObservableNode";
import { GradientNormObservableNode } from "../components/nodes/GradientNormObservableNode";
import { FourierComponentObservableNode } from "../components/nodes/FourierComponentObservableNode";
import { InformationPlaneObservableNode } from "../components/nodes/InformationPlaneObservableNode";
import { LayerSpectralNormObservableNode } from "../components/nodes/LayerSpectralNormObservableNode";
import { ActivationStatsObservableNode } from "../components/nodes/ActivationStatsObservableNode";
import { AttentionMapObservableNode } from "../components/nodes/AttentionMapObservableNode";
import { EmbeddingEffectiveRankObservableNode } from "../components/nodes/EmbeddingEffectiveRankObservableNode";
import { EmbeddingFeatureDriftObservableNode } from "../components/nodes/EmbeddingFeatureDriftObservableNode";
import { SinkAttentionMassObservableNode } from "../components/nodes/SinkAttentionMassObservableNode";
import { AttentionEntropyMeanObservableNode } from "../components/nodes/AttentionEntropyMeanObservableNode";
import { AttentionMaxWeightMeanObservableNode } from "../components/nodes/AttentionMaxWeightMeanObservableNode";
import { AttentionHeadSinkMaxObservableNode } from "../components/nodes/AttentionHeadSinkMaxObservableNode";
import { AttentionPositionBiasRatioObservableNode } from "../components/nodes/AttentionPositionBiasRatioObservableNode";
import { AttentionRelationScoreObservableNode } from "../components/nodes/AttentionRelationScoreObservableNode";
import { ActivationNormMeanObservableNode } from "../components/nodes/ActivationNormMeanObservableNode";
import { ActivationOutlierRatioObservableNode } from "../components/nodes/ActivationOutlierRatioObservableNode";
import { ModelWeightTensorsNode } from "../components/nodes/ModelWeightTensorsNode";
import { EffectiveRankNode } from "../components/nodes/EffectiveRankNode";
import { SeriesEndpointGapNode } from "../components/nodes/SeriesEndpointGapNode";
import { SmoothingCurveNode } from "../components/nodes/SmoothingCurveNode";
import { DerivativeCurveNode } from "../components/nodes/DerivativeCurveNode";
import { PredictionNode } from "../components/nodes/PredictionNode";
import { TensorSelectorNode } from "../components/nodes/TensorSelectorNode";
import { DimensionPermutatorNode } from "../components/nodes/DimensionPermutatorNode";
import { TensorSlicingNode } from "../components/nodes/TensorSlicingNode";
import { TensorSplitterNode } from "../components/nodes/TensorSplitterNode";
import { ReshapeNode } from "../components/nodes/ReshapeNode";
import { FlattenNode } from "../components/nodes/FlattenNode";
import { EinsumNode } from "../components/nodes/EinsumNode";
import { SoftmaxNode } from "../components/nodes/SoftmaxNode";
import { CausalMaskNode } from "../components/nodes/CausalMaskNode";
import { TensorViz0dNode } from "../components/nodes/TensorViz0dNode";
import { TensorScatterVizNode } from "../components/nodes/TensorScatterVizNode";
import { SweepDataTableNode } from "../components/nodes/SweepDataTableNode";
import { TableVizNode } from "../components/nodes/TableVizNode";
import { makeTensorVizComponent } from "../components/nodes/TensorVizNode";
import { CommentNode } from "../components/nodes/CommentNode";
import { UrlNode } from "../components/nodes/UrlNode";
import { RegressorNode } from "../components/nodes/RegressorNode";
import { CurveAnnotatorNode } from "../components/nodes/CurveAnnotatorNode";
import { GraphAssistFailureOverlayNode } from "../components/nodes/GraphAssistFailureOverlayNode";
import { ShapeCheckerNode } from "../components/nodes/ShapeCheckerNode";
import { TensorReaderNode } from "../components/nodes/TensorReaderNode";
import { withNodeResize } from "../components/nodes/withNodeResize";
import {
  NODE_SPEC_REGISTRY,
  nodeRegistryCodegen,
  nodeRegistryDefaults,
  nodeRegistryHint,
  nodeRegistryTitle,
  type NodeCodegenArgs,
  type NodeFamily,
  type NodeRegistrySpec,
  type RegisteredNodeKind,
} from "./nodeRegistrySpec";

export {
  NODE_SPEC_REGISTRY,
  nodeRegistryCodegen,
  nodeRegistryDefaults,
  nodeRegistryHint,
  nodeRegistryTitle,
  type NodeCodegenArgs,
  type NodeFamily,
  type RegisteredNodeKind,
};

export interface NodeSpec<D = Record<string, unknown>> extends NodeRegistrySpec<D> {
  component: ComponentType<NodeProps>;
}

// Components are resolved from generated specs through componentKey.
function nodeSpec<D = Record<string, unknown>>(
  type: string,
  component: ComponentType<NodeProps>,
): NodeSpec<D> {
  const spec = NODE_SPEC_REGISTRY[type as RegisteredNodeKind] as NodeRegistrySpec<D> | undefined;
  return { type, component, ...spec };
}

/** componentKey → component adapters(adapter,不是 registry)。Custom-TSX 的 adapter
 * 条目=在此加一行;条目只做组件映射,绝不携带 schema facts(核心句)。 */
const GENERATED_COMPONENT_ADAPTERS: Record<string, ComponentType<NodeProps>> = {
  GenericObservableNode,
  GenericDatasetNode,
  AdvectionDatasetNode,
  DiffusionPdeDatasetNode,
  ReactionDiffusionDatasetNode,
  LinearDatasetNode,
  WeightL2ObservableNode,
  SinkAttentionMassObservableNode,
  AttentionHeadSinkMaxObservableNode,
  AttentionEntropyMeanObservableNode,
  AttentionMaxWeightMeanObservableNode,
  AttentionPositionBiasRatioObservableNode,
  AttentionRelationScoreObservableNode,
  ActivationNormMeanObservableNode,
  ActivationOutlierRatioObservableNode,
  EmbeddingEffectiveRankObservableNode,
  EmbeddingFeatureDriftObservableNode,
  HessianEigenvaluesObservableNode,
  GradientNormObservableNode,
  FourierComponentObservableNode,
  InformationPlaneObservableNode,
  LayerSpectralNormObservableNode,
  ActivationStatsObservableNode,
  AttentionMapObservableNode,
  ObservableUserNode,
  ToyLanguageLmDatasetNode,
  VisionDatasetNode,
  MlpModelNode,
  GatedMlpModelNode,
  MoeMlpModelNode,
  MlpTokenModelNode,
  AttentionOnlyModelNode,
  AlternativeArchTokenLmNode,
  TransformerTokenModelNode,
  TransformerMultiTokenModelNode,
  NumericTransformerModelNode,
  NumericHyenaModelNode,
  MppSpatiotemporalModelNode,
  AfnoLiteSpatiotemporalModelNode,
  KanModelNode,
  DiffusionScoreModelNode,
  UnetDdpmModelNode,
  ResidualLnModelNode,
  CrlResidualMlpNode,
  LinearLayerNode,
  ActivationLayerNode,
  LayerNormLayerNode,
  RmsNormLayerNode,
  EmbeddingLayerNode,
  UnembeddingLayerNode,
  AbsolutePosEmbedLayerNode,
  RotaryEmbedLayerNode,
  LocalMixingLayerNode,
  AfnoAtomicLayerNode,
  CombinedModelNode,
  TensorAddNode,
  TensorStackNode,
  TensorConcatNode,
  TensorConstantNode,
  TensorLinspaceNode,
  ElementwiseTransformNode,
  FakeTensorNode,
  TensorSplitterNode,
  ReshapeNode,
  FlattenNode,
  EinsumNode,
  SoftmaxNode,
  CausalMaskNode,
  SchemaNode,
  LrScheduleNode,
  MupLrScheduleNode,
  CyclicLrScheduleNode,
  CyclicBatchScheduleNode,
  MseLossNode,
  L1RegNode,
  L2RegNode,
  L2ProjectionNode,
  DiffusionMseLossNode,
  CrossEntropyLossNode,
  BinaryCrossEntropyWithLogitsLossNode,
  KanRegNode,
  MupInitializationNode,
  IdnnsInitializationNode,
  SaxeInitializationNode,
  SymmetrizedMlpInitNode,
  TrainerNode,
  CrlTrainerNode,
  ResnetModelNode,
  VitModelNode,
  KeskarCnnModelNode,
  Vgg11CifarModelNode,
  DatasetMixerBNode,
  DatasetMixerNode,
  InputSamplerNode,
  RandomInputDistributionNode,
  TeacherDatasetNode,
  BigramLowRankDatasetNode,
  CircleRandomWalkDatasetNode,
  CircularMotionDatasetNode,
  CrlEnvConfigNode,
  InContextAssociativeRecallDatasetNode,
  Kepler2dDatasetNode,
  ModularAdditionDatasetNode,
  SymbolicFuncDatasetNode,
  TokenPredictionDatasetNode,
  UniformLinearMotionDatasetNode,
  UnigramDatasetNode,
  CommentNode,
  UrlNode,
  GraphAssistFailureOverlayNode,
  PcaNode,
  SvdNode,
  StatisticsNode,
  Statistics2Node,
  BasicCalculatorNode,
  ModelWeightTensorsNode,
  EffectiveRankNode,
  SeriesEndpointGapNode,
  SmoothingCurveNode,
  DerivativeCurveNode,
  PredictionNode,
  ShapeCheckerNode,
  TensorReaderNode,
  TensorSelectorNode,
  DimensionPermutatorNode,
  TensorSlicingNode,
  SweepDataTableNode,
  TableVizNode,
  CurveSeriesTableNode,
  CurveSeriesVizNode,
  MetricCompareNode,
  ParametricPathSamplerNode,
  LinearInterpolationBarrierNode,
  BezierModeConnectivityNode,
  RegressorNode,
  CurveAnnotatorNode,
  TensorVizGeneralNode: makeTensorVizComponent("general"),
  TensorViz1dNode: makeTensorVizComponent("1d"),
  TensorViz2dNode: makeTensorVizComponent("2d"),
  TensorViz0dNode,
  TensorScatterVizNode,
  TrainingVisualizationNode,
  ImageDatasetDisplayerNode,
  VisualizeKanNode,
  ProteinStructureDisplayerNode,
  ProteinStructureComparisonVizNode,
  ObservableVizNode,
  NeuronTrajectory2dVizNode,
  ModelCheckpointNode,
  ActivationNode,
  DeterministicDiffusionSamplerNode,
  PairedGenerationSimilarityNode,
  RpScoreSscdNode,
  NearestTrainGlNode,
};

export function resolveGeneratedComponent(type: string, componentKey: string | undefined): ComponentType<NodeProps> {
  // componentKey 缺省 → Generic；显式声明但 adapter 表缺失时直接 throw。
  // 会重新制造"半接线"——漏 adapter 必须在启动/测试期爆)。
  if (componentKey == null || componentKey === "") return GenericObservableNode;
  const component = GENERATED_COMPONENT_ADAPTERS[componentKey];
  if (!component) {
    throw new Error(
      `generated node "${type}" declares componentKey "${componentKey}" but GENERATED_COMPONENT_ADAPTERS has no such entry — add the adapter line in nodeRegistry.ts`,
    );
  }
  return component;
}

const GENERATED_NODE_REGISTRY = Object.fromEntries(
  Object.entries(GENERATED_NODE_SPECS).map(([type, g]) => [
    type,
    nodeSpec(type, resolveGeneratedComponent(type, g.frontend?.componentKey)),
  ]),
) as Record<string, NodeSpec>;

export const NODE_REGISTRY = GENERATED_NODE_REGISTRY;

export function buildNodeTypesFromRegistry(registry: Record<string, NodeSpec> = NODE_REGISTRY): NodeTypes {
  return Object.fromEntries(
    Object.entries(registry).map(([type, spec]) => [
      type,
      spec.resize === false ? spec.component : withNodeResize(spec.component),
    ]),
  ) as NodeTypes;
}

export const registeredResearchNodeTypes = buildNodeTypesFromRegistry();
