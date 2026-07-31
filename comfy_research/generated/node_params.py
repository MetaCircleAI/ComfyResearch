from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


# Generated from frontend/src/graph/nodeRegistrySpec.ts.
# Do not edit by hand; run `npm run generate:node-manifest` from frontend/.
class NodeParamsBase(BaseModel):
    model_config = ConfigDict(extra="allow")


class AbsolutePosEmbedLayerParams(NodeParamsBase):
    maxSeqLen: int | list[int] = 512
    embeddingDim: int | list[int] = 64
    seed: int | list[int] = 0

class ActivationParams(NodeParamsBase):
    pass

class ActivationLayerParams(NodeParamsBase):
    activation: str | list[str] = "relu"
    leakyP: float | list[float] = 0

class AdamOptimizerParams(NodeParamsBase):
    learningRate: float | list[float] = 0.001
    beta1: float | list[float] = 0.9
    beta2: float | list[float] = 0.999
    epsilon: float | list[float] = 1e-8
    weightDecay: float | list[float] = 0

class AdamwOptimizerParams(NodeParamsBase):
    learningRate: float | list[float] = 0.001
    beta1: float | list[float] = 0.9
    beta2: float | list[float] = 0.999
    epsilon: float | list[float] = 1e-8
    weightDecay: float | list[float] = 0.01

class AdvectionDatasetParams(NodeParamsBase):
    contextFrames: int | list[int] = 4
    channels: int | list[int] = 1
    gridSize: int | list[int] = 16
    trainSize: int | list[int] = 512
    testSize: int | list[int] = 128
    warmupSteps: int | list[int] = 40
    dt: float | list[float] = 0.05
    diffusionCoeff: float | list[float] = 0.2
    reactionRate: float | list[float] = 1
    velocityX: float | list[float] = 0.5
    velocityY: float | list[float] = 0.2
    icScale: float | list[float] = 0.5
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class AfnoEncoderBlockLayerParams(NodeParamsBase):
    contextFrames: int | list[int] = 4
    channels: int | list[int] = 1
    gridSize: int | list[int] = 16
    inputDim: int | list[int] = 1024
    outputDim: int | list[int] = 1024
    patchSize: int | list[int] = 4
    embedDim: int | list[int] = 64
    numHeads: int | list[int] = 4
    ffRatio: float | list[float] = 2
    dropout: float | list[float] = 0
    numSpectralBlocks: int | list[int] = 1
    maxFrequencyModes: int | list[int] = 4
    spectralShrinkFactor: float | list[float] = 1
    seed: int | list[int] = 0

class AfnoLiteSpatiotemporalModelParams(NodeParamsBase):
    contextFrames: int | list[int] = 4
    channels: int | list[int] = 1
    gridSize: int | list[int] = 16
    inputDim: int | list[int] = 1024
    outputDim: int | list[int] = 1024
    patchSize: int | list[int] = 4
    embedDim: int | list[int] = 64
    depth: int | list[int] = 2
    numHeads: int | list[int] = 4
    ffRatio: float | list[float] = 2
    dropout: float | list[float] = 0
    numSpectralBlocks: int | list[int] = 1
    maxFrequencyModes: int | list[int] = 4
    spectralShrinkFactor: float | list[float] = 1
    seed: int | list[int] = 0

class AfnoPatchDecodeLayerParams(NodeParamsBase):
    contextFrames: int | list[int] = 4
    channels: int | list[int] = 1
    gridSize: int | list[int] = 16
    inputDim: int | list[int] = 1024
    outputDim: int | list[int] = 1024
    patchSize: int | list[int] = 4
    embedDim: int | list[int] = 64
    numHeads: int | list[int] = 4
    ffRatio: float | list[float] = 2
    dropout: float | list[float] = 0
    numSpectralBlocks: int | list[int] = 1
    maxFrequencyModes: int | list[int] = 4
    spectralShrinkFactor: float | list[float] = 1
    seed: int | list[int] = 0

class AfnoPatchEmbedLayerParams(NodeParamsBase):
    contextFrames: int | list[int] = 4
    channels: int | list[int] = 1
    gridSize: int | list[int] = 16
    inputDim: int | list[int] = 1024
    outputDim: int | list[int] = 1024
    patchSize: int | list[int] = 4
    embedDim: int | list[int] = 64
    numHeads: int | list[int] = 4
    ffRatio: float | list[float] = 2
    dropout: float | list[float] = 0
    numSpectralBlocks: int | list[int] = 1
    maxFrequencyModes: int | list[int] = 4
    spectralShrinkFactor: float | list[float] = 1
    seed: int | list[int] = 0

class AfnoSpectralMixerLayerParams(NodeParamsBase):
    contextFrames: int | list[int] = 4
    channels: int | list[int] = 1
    gridSize: int | list[int] = 16
    inputDim: int | list[int] = 1024
    outputDim: int | list[int] = 1024
    patchSize: int | list[int] = 4
    embedDim: int | list[int] = 64
    numHeads: int | list[int] = 4
    ffRatio: float | list[float] = 2
    dropout: float | list[float] = 0
    numSpectralBlocks: int | list[int] = 1
    maxFrequencyModes: int | list[int] = 4
    spectralShrinkFactor: float | list[float] = 1
    seed: int | list[int] = 0

class AgentTraceVizParams(NodeParamsBase):
    logScaleX: bool | list[bool] = False
    logScaleY: bool | list[bool] = False

class AttentionOnlyModelParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    embedDim: int | list[int] = 32
    numHeads: int | list[int] = 4
    contextLength: int | list[int] = 4
    causalAttention: str | list[str] = "yes"
    localMixingKernel: int | list[int] = 0
    qkNorm: str | list[str] = "no"
    attnTemperature: float | list[float] = 1
    attnLogitCap: float | list[float] = 0
    attnDropout: float | list[float] = 0
    seed: int | list[int] = 0

class BasicCalculatorParams(NodeParamsBase):
    inputCount: int | list[int] = 2
    equationLatex: str | list[str] = "x_1 + x_2"

class BigramLowRankDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    rank: int | list[int] = 20
    logitScale: float | list[float] = 1
    corruptRatio: float | list[float] = 0
    corruptScale: float | list[float] = 5
    decayType: str | list[str] = "power_law"
    alpha: float | list[float] = 0
    trainSize: int | list[int] = 1200
    testSize: int | list[int] = 300
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class BinaryCrossEntropyWithLogitsLossParams(NodeParamsBase):
    lossScale: float | list[float] = 1

class BiographyLmDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 64
    contextLength: int | list[int] = 32
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"
    biographyAugmentation: str | list[str] = "template"
    slotNoiseProb: float | list[float] = 0

class CausalMaskParams(NodeParamsBase):
    diagonalOffset: int | list[int] = 1
    ioMode: str | list[str] = "input-output"
    levelMode: str | list[str] = "high"

class Cifar10DatasetParams(NodeParamsBase):
    trainSize: int | list[int] = 2048
    testSize: int | list[int] = 512
    initSeed: int | list[int] = 0
    seed: int | list[int] = 0
    subsetSeed: int | list[int] = 0
    classBalanced: bool | list[bool] = True
    normalize: str | list[str] = "zero_one"
    inputTransform: str | list[str] = "none"
    preprocessing: str | list[str] = "none"
    labelCorruption: float | list[float] = 0
    trainingRecipe: str | list[str] = "standard"
    flattenOutput: bool | list[bool] = False
    samplingMode: str | list[str] = "fixed"
    specCodeName: str | list[str] = "cifar10_datasetSpec"
    downloadCacheDir: str | list[str] = ""

class CircleRandomWalkDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 10
    contextLength: int | list[int] = 1
    rightStepProb: float | list[float] = 0.5
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class CircularMotionDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 128
    contextLength: int | list[int] = 20
    radiusMin: float | list[float] = 0.15
    radiusMax: float | list[float] = 0.35
    angularVelocity: float | list[float] = 0.5
    trainSize: int | list[int] = 4000
    testSize: int | list[int] = 1000
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class CogsDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 128
    contextLength: int | list[int] = 32
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"

class CombinedModelParams(NodeParamsBase):
    pass

class CommentParams(NodeParamsBase):
    text: str | list[str] = ""
    url: str | list[str] = ""

class CrlEnvConfigParams(NodeParamsBase):
    preset: str | list[str] = "point_u4_maze"
    numEnvs: int | list[int] = 8
    episodeLength: int | list[int] = 200
    mazeSizeScaling: int | list[int] = 4
    seed: int | list[int] = 0

class CrlResidualMlpParams(NodeParamsBase):
    stateDim: int | list[int] = 4
    actionDim: int | list[int] = 2
    goalDim: int | list[int] = 2
    actorWidth: int | list[int] = 128
    criticWidth: int | list[int] = 128
    actorDepth: int | list[int] = 4
    criticDepth: int | list[int] = 4
    embedDim: int | list[int] = 64
    activation: str | list[str] = "silu"
    seed: int | list[int] = 0

class CrlTrainerParams(NodeParamsBase):
    trainingSteps: int | list[int] = 40
    logFrequency: int | list[int] = 5
    computeDevice: str | list[str] = "cpu"
    batchSize: int | list[int] = 32
    unrollLength: int | list[int] = 24
    sgdStepsPerTrainStep: int | list[int] = 4
    gamma: float | list[float] = 0.99
    logsumexpPenaltyCoeff: float | list[float] = 0.1
    entropyParam: float | list[float] = 0.5
    disableEntropy: bool | list[bool] = False
    maxReplayChunks: int | list[int] = 200
    seed: int | list[int] = 0

class CrossEntropyLossParams(NodeParamsBase):
    lossScale: float | list[float] = 1
    labelSmoothing: float | list[float] = 0
    lossMaskContextLength: int | list[int] = 1
    lossMaskMode: str | list[str] = "all"
    lossMaskCustom: str | list[str] = ""

class CurveAnnotatorParams(NodeParamsBase):
    pass

class CurveSeriesTableParams(NodeParamsBase):
    pass

class CurveSeriesVizParams(NodeParamsBase):
    dualAxis: bool | list[bool] = True
    meanByRun: bool | list[bool] = False
    plotXMode: str | list[str] = "progress"
    plotXKey: str | list[str] = "step"

class CyclicBatchScheduleParams(NodeParamsBase):
    batchMin: int | list[int] = 128
    batchMax: int | list[int] = 640
    cycleLengthEpochs: int | list[int] = 10
    refBatchSize: int | list[int] = 128
    cycleLengthSteps: int | list[int] = 0
    scheduleMode: str | list[str] = "discrete_epoch"

class CyclicLrScheduleParams(NodeParamsBase):
    lrMin: float | list[float] = 0.001
    lrMax: float | list[float] = 0.005
    cycleLengthEpochs: int | list[int] = 10
    refBatchSize: int | list[int] = 128
    cycleLengthSteps: int | list[int] = 0
    scheduleMode: str | list[str] = "discrete_epoch"

class DatasetMixerParams(NodeParamsBase):
    trainTotalSamples: int | list[int] = 800
    testTotalSamples: int | list[int] = 0
    proportionA: float | list[float] = 0.5
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class DatasetMixerBParams(NodeParamsBase):
    interpolationLambda: float | list[float] = 0.5

class DerivativeCurveParams(NodeParamsBase):
    order: str | list[str] = "1"
    logScaleX: bool | list[bool] = False
    logScaleY: bool | list[bool] = False

class DeterministicDiffusionSamplerParams(NodeParamsBase):
    noiseSeed: int | list[int] = 0
    sampleCount: int | list[int] = 64
    numSteps: int | list[int] = 50

class DiagonalSsmTokenModelParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    embedDim: int | list[int] = 32
    contextLength: int | list[int] = 8
    seed: int | list[int] = 0
    localMixingKernel: int | list[int] = 0
    numLayers: int | list[int] = 2

class DiffusionMseLossParams(NodeParamsBase):
    lossScale: float | list[float] = 1

class DiffusionPdeDatasetParams(NodeParamsBase):
    contextFrames: int | list[int] = 4
    channels: int | list[int] = 1
    gridSize: int | list[int] = 16
    trainSize: int | list[int] = 512
    testSize: int | list[int] = 128
    warmupSteps: int | list[int] = 40
    dt: float | list[float] = 0.05
    diffusionCoeff: float | list[float] = 0.2
    reactionRate: float | list[float] = 1
    velocityX: float | list[float] = 0.5
    velocityY: float | list[float] = 0.2
    icScale: float | list[float] = 0.5
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class DiffusionScoreModelParams(NodeParamsBase):
    inputDim: int | list[int] = 8
    hiddenDim: int | list[int] = 128
    depth: int | list[int] = 3
    timeEmbedDim: int | list[int] = 64
    diffusionTimesteps: int | list[int] = 100
    seed: int | list[int] = 0

class DimensionPermutatorParams(NodeParamsBase):
    pass

class DistanceContactLayerParams(NodeParamsBase):
    inFeatures: int | list[int] = 64
    outFeatures: int | list[int] = 6
    bias: int | list[int] = 1
    seed: int | list[int] = 11

class DockingPoseVizParams(NodeParamsBase):
    pass

class DyckDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 2
    contextLength: int | list[int] = 16
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"
    numBracketTypes: int | list[int] = 1
    maxNestingDepth: int | list[int] = 0

class EffectiveRankParams(NodeParamsBase):
    pass

class EinsumParams(NodeParamsBase):
    equation: str | list[str] = "b h t d, b h s d -> b h t s"
    ioMode: str | list[str] = "input-output"
    levelMode: str | list[str] = "high"

class ElementwiseTransformParams(NodeParamsBase):
    ruleLatex: str | list[str] = "x^2"

class EmbeddingLayerParams(NodeParamsBase):
    numEmbeddings: int | list[int] = 4096
    embeddingDim: int | list[int] = 64
    numIndexColumns: int | list[int] = 1
    paddingIdx: int | list[int] = -1
    scaleGradByFreq: int | list[int] = 0
    seed: int | list[int] = 0

class EnergyReadoutLayerParams(NodeParamsBase):
    inFeatures: int | list[int] = 48
    outFeatures: int | list[int] = 4
    bias: int | list[int] = 1
    seed: int | list[int] = 7

class EquivariantMessageLayerParams(NodeParamsBase):
    inFeatures: int | list[int] = 48
    outFeatures: int | list[int] = 48
    bias: int | list[int] = 1
    seed: int | list[int] = 7

class FakeTensorParams(NodeParamsBase):
    dtype: str | list[str] = "float"

class FlattenParams(NodeParamsBase):
    ioMode: str | list[str] = "input-output"
    levelMode: str | list[str] = "high"

class FormalLanguageSuiteDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 8
    contextLength: int | list[int] = 16
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"
    languageType: str | list[str] = "anbn"

class GatedMlpModelParams(NodeParamsBase):
    inputDim: int | list[int] = 10
    outputDim: int | list[int] = 1
    depth: int | list[int] = 2
    width: int | list[int] = 64
    activation: str | list[str] = "silu"
    seed: int | list[int] = 0

class GatedMlpTokenModelParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    embedDim: int | list[int] = 64
    tokensPerInput: int | list[int] = 1
    depth: int | list[int] = 2
    width: int | list[int] = 64
    numExperts: int | list[int] = 4
    activation: str | list[str] = "relu"
    tieWeights: str | list[str] = "yes"
    seed: int | list[int] = 0

class GaussianBlobDatasetParams(NodeParamsBase):
    trainSize: int | list[int] = 2048
    testSize: int | list[int] = 512
    initSeed: int | list[int] = 0
    seed: int | list[int] = 0
    flattenOutput: bool | list[bool] = False
    samplingMode: str | list[str] = "fixed"
    specCodeName: str | list[str] = "gaussian_blob_datasetSpec"
    numClasses: int | list[int] = 10
    imageSize: int | list[int] = 28
    noiseLevel: float | list[float] = 0.15

class GraphAssistFailureOverlayParams(NodeParamsBase):
    pass

class HoleCountingDatasetParams(NodeParamsBase):
    trainSize: int | list[int] = 2048
    testSize: int | list[int] = 512
    initSeed: int | list[int] = 0
    seed: int | list[int] = 0
    flattenOutput: bool | list[bool] = False
    samplingMode: str | list[str] = "fixed"
    specCodeName: str | list[str] = "hole_counting_datasetSpec"
    imageSize: int | list[int] = 48
    maxHoles: int | list[int] = 3

class HyenaLikeConvModelParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    embedDim: int | list[int] = 32
    contextLength: int | list[int] = 8
    seed: int | list[int] = 0
    localMixingKernel: int | list[int] = 0
    depth: int | list[int] = 2
    convKernel: int | list[int] = 7
    ffMult: float | list[float] = 2

class HypothesisParams(NodeParamsBase):
    text: str | list[str] = "Hypothesis: "
    url: str | list[str] = ""

class IdnnsInitializationParams(NodeParamsBase):
    seed: int | list[int] = 0

class ImageDatasetDisplayerParams(NodeParamsBase):
    split: str | list[str] = "train"
    indexRange: str | list[str] = "0-9"
    columnsPerRow: int | list[int] = 5

class InContextAssociativeRecallDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 64
    numPairs: int | list[int] = 32
    inContextRepeat: int | list[int] = 1
    crossSampleRepeatProb: float | list[float] = 0
    repeatedTokenCount: int | list[int] = 2
    trainSize: int | list[int] = 10000
    testSize: int | list[int] = 2000
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class InformationBottleneckDatasetParams(NodeParamsBase):
    inputDim: int | list[int] = 12
    outputDim: int | list[int] = 2
    trainSize: int | list[int] = 3482
    testSize: int | list[int] = 4096
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class InputSamplerParams(NodeParamsBase):
    numSamples: int | list[int] = 800

class InteratomicEvalVizParams(NodeParamsBase):
    pass

class KanModelParams(NodeParamsBase):
    inputDim: int | list[int] = 10
    outputDim: int | list[int] = 1
    depth: int | list[int] = 2
    width: int | list[int] = 5
    grid: int | list[int] = 3
    k: int | list[int] = 3
    baseFun: str | list[str] = "silu"
    seed: int | list[int] = 0

class KanRegParams(NodeParamsBase):
    regMetric: str | list[str] = "edge_forward_spline_n"
    lamb: float | list[float] = 0.01
    lambL1: float | list[float] = 1
    lambEntropy: float | list[float] = 2
    lambCoef: float | list[float] = 0
    lambCoefDiff: float | list[float] = 0

class Kepler2dDatasetParams(NodeParamsBase):
    contextLength: int | list[int] = 8
    trainSize: int | list[int] = 1600
    testSize: int | list[int] = 400
    semiMajorAxisMin: float | list[float] = 0.7
    semiMajorAxisMax: float | list[float] = 1.3
    eccentricityMin: int | list[int] = 0
    eccentricityMax: float | list[float] = 0.55
    meanMotion: float | list[float] = 0.4
    outputDistribution: str | list[str] = "deterministic"
    noiseLevel: float | list[float] = 0
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class KeskarC1C2CnnModelParams(NodeParamsBase):
    architecture: str | list[str] = "c1"
    seed: int | list[int] = 0
    specCodeName: str | list[str] = "keskarCnnModelSpec"

class L1RegParams(NodeParamsBase):
    lossScale: float | list[float] = 1

class L2ProjectionParams(NodeParamsBase):
    targetNorm: float | list[float] = 1

class L2RegParams(NodeParamsBase):
    lossScale: float | list[float] = 1

class LayerNormLayerParams(NodeParamsBase):
    normalizedShape: int | list[int] = 64
    eps: float | list[float] = 0.00001
    elementwiseAffine: int | list[int] = 1

class LinearAttentionModelParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    embedDim: int | list[int] = 32
    contextLength: int | list[int] = 8
    seed: int | list[int] = 0
    localMixingKernel: int | list[int] = 0
    numHeads: int | list[int] = 4
    causalAttention: str | list[str] = "yes"

class LinearDatasetParams(NodeParamsBase):
    inputDim: int | list[int] = 10
    outputDim: int | list[int] = 1
    inputDistribution: str | list[str] = "standard_normal"
    outputDistribution: str | list[str] = "additive_gaussian"
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    noiseLevel: float | list[float] = 0.25
    alpha: float | list[float] = 1
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class LinearLayerParams(NodeParamsBase):
    inFeatures: int | list[int] = 10
    outFeatures: int | list[int] = 10
    bias: int | list[int] = 1
    seed: int | list[int] = 0

class ListopsDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 64
    contextLength: int | list[int] = 48
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"

class LocalMixingLayerParams(NodeParamsBase):
    modelDim: int | list[int] = 64
    kernelSize: int | list[int] = 5
    seed: int | list[int] = 0

class LrScheduleParams(NodeParamsBase):
    lrWarmupSteps: int | list[int] = 0
    lrSchedule: str | list[str] = "constant"
    cosineLrMinFraction: float | list[float] = 0
    exponentialDecayFactor: float | list[float] = 0.95
    exponentialDecayEpochs: int | list[int] = 1

class MemorizationADatasetParams(NodeParamsBase):
    inputDim: int | list[int] = 40
    outputDim: int | list[int] = 40
    inputDistribution: str | list[str] = "standard_normal"
    outputDistribution: str | list[str] = "uniform_class_probs"
    trainSize: int | list[int] = 160
    testSize: int | list[int] = 0
    noiseLevel: float | list[float] = 0
    alpha: float | list[float] = 1
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    specCodeName: str | list[str] = "Memorization_A_Dataset"

class MemorizationBDatasetParams(NodeParamsBase):
    inputDim: int | list[int] = 40
    outputDim: int | list[int] = 40
    vocabSize: int | list[int] = 40
    inputDistribution: str | list[str] = "standard_normal"
    outputDistribution: str | list[str] = "uniform_class_probs"
    trainSize: int | list[int] = 160
    testSize: int | list[int] = 0
    noiseLevel: float | list[float] = 0
    alpha: float | list[float] = 1
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    specCodeName: str | list[str] = "Memorization_B_Dataset"

class MetricCompareParams(NodeParamsBase):
    layout: str | list[str] = "horizontal"

class MlpModelParams(NodeParamsBase):
    inputDim: int | list[int] = 10
    outputDim: int | list[int] = 1
    depth: int | list[int] = 2
    width: int | list[int] = 64
    activation: str | list[str] = "relu"
    outputScale: float | list[float] = 1
    seed: int | list[int] = 0

class MlpTokenModelParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    embedDim: int | list[int] = 64
    tokensPerInput: int | list[int] = 1
    depth: int | list[int] = 2
    width: int | list[int] = 64
    numExperts: int | list[int] = 4
    activation: str | list[str] = "relu"
    tieWeights: str | list[str] = "yes"
    seed: int | list[int] = 0

class MnistDatasetParams(NodeParamsBase):
    trainSize: int | list[int] = 2048
    testSize: int | list[int] = 512
    initSeed: int | list[int] = 0
    seed: int | list[int] = 0
    flattenOutput: bool | list[bool] = False
    samplingMode: str | list[str] = "fixed"
    specCodeName: str | list[str] = "mnist_datasetSpec"
    downloadCacheDir: str | list[str] = ""

class ModelCheckpointParams(NodeParamsBase):
    checkpoint_b64: str | list[str] = ""
    memoryCheckpoint_b64: str | list[str] = ""
    checkpointSource: str | list[str] = "memory"
    checkpointFileName: str | list[str] = ""

class ModelWeightTensorsParams(NodeParamsBase):
    pass

class ModularAdditionDatasetParams(NodeParamsBase):
    modulus: int | list[int] = 59
    trainFraction: float | list[float] = 0.3
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class MoeMlpModelParams(NodeParamsBase):
    inputDim: int | list[int] = 10
    outputDim: int | list[int] = 1
    depth: int | list[int] = 2
    width: int | list[int] = 32
    numExperts: int | list[int] = 4
    activation: str | list[str] = "silu"
    seed: int | list[int] = 0

class MoeMlpTokenModelParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    embedDim: int | list[int] = 64
    tokensPerInput: int | list[int] = 1
    depth: int | list[int] = 2
    width: int | list[int] = 64
    numExperts: int | list[int] = 4
    activation: str | list[str] = "relu"
    tieWeights: str | list[str] = "yes"
    seed: int | list[int] = 0

class MppSpatiotemporalModelParams(NodeParamsBase):
    contextFrames: int | list[int] = 4
    channels: int | list[int] = 1
    gridSize: int | list[int] = 16
    inputDim: int | list[int] = 1024
    outputDim: int | list[int] = 1024
    patchSize: int | list[int] = 4
    embedDim: int | list[int] = 128
    depth: int | list[int] = 4
    numHeads: int | list[int] = 4
    ffRatio: float | list[float] = 4
    dropout: float | list[float] = 0
    seed: int | list[int] = 0

class MseLossParams(NodeParamsBase):
    lossScale: float | list[float] = 1
    lossMaskContextLength: int | list[int] = 1
    lossMaskMode: str | list[str] = "all"

class MultiHopFactChainDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 96
    contextLength: int | list[int] = 48
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"
    chainHops: int | list[int] = 3

class MuonOptimizerParams(NodeParamsBase):
    learningRate: float | list[float] = 0.003
    momentum: float | list[float] = 0.95

class MupInitializationParams(NodeParamsBase):
    pass

class MupLrScheduleParams(NodeParamsBase):
    mupEmbedLrMult: float | list[float] = 1
    mupHiddenLrMult: float | list[float] = 1
    mupOutputLrMult: float | list[float] = 1

class NgramLanguageDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 32
    contextLength: int | list[int] = 16
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"
    orderN: int | list[int] = 3
    dirichletAlpha: float | list[float] = 1

class NumericHyenaModelParams(NodeParamsBase):
    contextLength: int | list[int] = 8
    inputDim: int | list[int] = 2
    outputDim: int | list[int] = 2
    modelDim: int | list[int] = 64
    depth: int | list[int] = 2
    convKernel: int | list[int] = 7
    ffMult: float | list[float] = 2
    localMixingKernel: int | list[int] = 0
    seed: int | list[int] = 0

class NumericTransformerModelParams(NodeParamsBase):
    contextLength: int | list[int] = 2
    inputDim: int | list[int] = 1
    outputDim: int | list[int] = 1
    modelDim: int | list[int] = 32
    numHeads: int | list[int] = 1
    numLayers: int | list[int] = 1
    ffDim: int | list[int] = 64
    activation: str | list[str] = "gelu"
    encoderBackend: str | list[str] = "pytorch"
    encoderDropout: float | list[float] = 0
    spectralNormLinears: str | list[str] = "no"
    stableQkNorm: str | list[str] = "no"
    stableAttnTemperature: float | list[float] = 1
    stableAttnLogitCap: float | list[float] = 0
    stableAttnDropout: float | list[float] = 0
    causalAttention: str | list[str] = "yes"
    seed: int | list[int] = 0

class ObservableAccuracyParams(NodeParamsBase):
    pass

class ObservableActivationNormMeanParams(NodeParamsBase):
    pass

class ObservableActivationOutlierRatioParams(NodeParamsBase):
    pass

class ObservableActivationStatsParams(NodeParamsBase):
    pass

class ObservableAttentionEntropyMeanParams(NodeParamsBase):
    pass

class ObservableAttentionHeadSinkMaxParams(NodeParamsBase):
    sinkTokenIndex: int | list[int] = 0

class ObservableAttentionMapParams(NodeParamsBase):
    attentionLayerIndices: int | list[int] = 0
    attentionBatchIndices: int | list[int] = 0
    attentionHeadIndices: int | list[int] = 0

class ObservableAttentionMaxWeightMeanParams(NodeParamsBase):
    pass

class ObservableAttentionPositionBiasRatioParams(NodeParamsBase):
    pass

class ObservableAttentionRelationScoreParams(NodeParamsBase):
    keyRelation: str | list[str] = "pos(k) == pos(q) - 1"
    queryFilter: str | list[str] = ""
    keyReduction: str | list[str] = "mean"
    layerIndex: int | list[int] = 0
    headIndex: int | list[int] = 0

class ObservableBezierModeConnectivityParams(NodeParamsBase):
    alphaSteps: int | list[int] = 21
    curveOptimizationSteps: int | list[int] = 500
    curveSamplesPerStep: int | list[int] = 4
    curveBatchSize: int | list[int] = 256
    curveLearningRate: float | list[float] = 0.01
    showTrainCurve: bool | list[bool] = True
    showTestCurve: bool | list[bool] = True
    recomputeBnStats: bool | list[bool] = True
    bnCalibrationBatches: int | list[int] = 100
    evalBatchSize: int | list[int] = 256

class ObservableCapacityParams(NodeParamsBase):
    pass

class ObservableEmbeddingEffectiveRankParams(NodeParamsBase):
    pass

class ObservableEmbeddingEvolutionParams(NodeParamsBase):
    label: str | list[str] = "Embedding evolution"

class ObservableEmbeddingFeatureDriftParams(NodeParamsBase):
    pass

class ObservableEmbeddingTrajectoryParams(NodeParamsBase):
    label: str | list[str] = "Embedding trajectory"

class ObservableFourierComponentParams(NodeParamsBase):
    frequency: float | list[float] = 1
    metric: str | list[str] = "relative_projection_mse"
    inputAxis: int | list[int] = 0
    outputIndex: int | list[int] = 0

class ObservableGradientNormParams(NodeParamsBase):
    normAggregation: str | list[str] = "global"
    gradientNormNormalized: bool | list[bool] = True

class ObservableHessianEigenvaluesParams(NodeParamsBase):
    topK: int | list[int] = 5
    order: str | list[str] = "descending"

class ObservableInformationPlaneParams(NodeParamsBase):
    bins: int | list[int] = 30
    maxSamples: int | list[int] = 512
    includeOutput: bool | list[bool] = True
    binning: str | list[str] = "uniform_intervals"
    outputMapping: str | list[str] = "tanh"

class ObservableLastLayerWeightNormParams(NodeParamsBase):
    pass

class ObservableLayerSpectralNormParams(NodeParamsBase):
    estimator: str | list[str] = "singular_value"
    powerIterations: int | list[int] = 10
    startVector: str | list[str] = "deterministic"
    seed: int | list[int] = 0

class ObservableLinearInterpolationBarrierParams(NodeParamsBase):
    alphaMin: float | list[float] = 0
    alphaMax: float | list[float] = 1
    alphaSteps: int | list[int] = 21
    showTrainCurve: bool | list[bool] = True
    showTestCurve: bool | list[bool] = True
    recomputeBnStats: bool | list[bool] = False
    bnCalibrationBatches: int | list[int] = 100
    evalBatchSize: int | list[int] = 256

class ObservableNearestTrainGlParams(NodeParamsBase):
    glThreshold: float | list[float] = 0.95

class ObservableNeuronTrajectory2dParams(NodeParamsBase):
    pass

class ObservablePairedGenerationSimilarityParams(NodeParamsBase):
    pass

class ObservableReluNonlinearCountParams(NodeParamsBase):
    hiddenLayerIndex: int | list[int] = 1

class ObservableRepresentationAlphaReqParams(NodeParamsBase):
    representationId: str | list[str] = "0::output"
    tokenPositionsAsSamples: bool | list[bool] = False

class ObservableRepresentationRankmeParams(NodeParamsBase):
    representationId: str | list[str] = "0::output"
    captureTrajectories: bool | list[bool] = False
    tokenPositionsAsSamples: bool | list[bool] = False

class ObservableRpScoreSscdParams(NodeParamsBase):
    threshold: float | list[float] = 0.95

class ObservableSinkAttentionMassParams(NodeParamsBase):
    sinkTokenIndex: int | list[int] = 0

class ObservableTrainTestGapParams(NodeParamsBase):
    pass

class ObservableUserParams(NodeParamsBase):
    userObservableId: str | list[str] = ""
    label: str | list[str] = "User observable"
    tensorVizNodeId: str | list[str] = ""
    tensorSelectorNodeId: str | list[str] = ""

class ObservableVizParams(NodeParamsBase):
    observableName: str | list[str] = "User observable"
    logScaleX: bool | list[bool] = False
    logScaleY: bool | list[bool] = False
    showSeries: bool | list[bool] = True
    showTrainCurve: bool | list[bool] = True
    showTestCurve: bool | list[bool] = True
    vizVariant: str | list[str] = "user"

class ObservableVizNeuronTrajectory2dParams(NodeParamsBase):
    pass

class ObservableWeightDisplacementParams(NodeParamsBase):
    pass

class ObservableWeightL1Params(NodeParamsBase):
    pass

class ObservableWeightL2Params(NodeParamsBase):
    normAggregation: str | list[str] = "global"

class ObservableWeightProductSvParams(NodeParamsBase):
    topK: int | list[int] = 3

class PairwiseRbfLayerParams(NodeParamsBase):
    inFeatures: int | list[int] = 18
    outFeatures: int | list[int] = 48
    bias: int | list[int] = 1
    seed: int | list[int] = 7

class PaperClassificationDatasetParams(NodeParamsBase):
    experimentMode: str | list[str] = "rank_collapse_skewed"
    inputDim: int | list[int] = 16
    outputDim: int | list[int] = 16
    trainSize: int | list[int] = 1029
    testSize: int | list[int] = 0
    frequencyRatio: float | list[float] = 2
    classSeparation: float | list[float] = 1
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class ParametricPathSamplerParams(NodeParamsBase):
    alphaMin: int | list[int] = -1
    alphaMax: int | list[int] = 2
    alphaSteps: int | list[int] = 50
    metric: str | list[str] = "loss"
    split: str | list[str] = "test"
    computeDevice: str | list[str] = "auto"
    remoteGpu: bool | list[bool] = False
    seriesLabel: str | list[str] = "parametric path"

class PcaParams(NodeParamsBase):
    representationId: str | list[str] = ""
    nComponents: int | list[int] = 2

class PcfgDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 32
    contextLength: int | list[int] = 16
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"
    pcfgGenMode: str | list[str] = "binary_tree"
    pcfgGrammarId: str | list[str] = "world_model"
    pcfgMaxDepth: int | list[int] = 8
    pcfgTermProb: float | list[float] = 0.35

class Phi1StyleDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 256
    contextLength: int | list[int] = 96
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"
    vocabCap: float | list[float] = 256
    tokenizerMode: str | list[str] = "char"
    seqLen: int | list[int] = 600
    stride: int | list[int] = 96
    domainMix: str | list[str] = "mixed"

class PredictionParams(NodeParamsBase):
    split: str | list[str] = "both"

class ProteinStructureComparisonVizParams(NodeParamsBase):
    predCoordsFlat: str | list[str] = ""
    trueCoordsFlat: str | list[str] = ""
    sampleIndex: int | list[int] = 0

class ProteinStructureDisplayerParams(NodeParamsBase):
    coordsFlat: str | list[str] = ""
    resolvedCoordsFlat: str | list[str] = ""
    showPolyline: bool | list[bool] = True
    sampleIndex: int | list[int] = 0

class RandomInputDistributionParams(NodeParamsBase):
    inputDim: int | list[int] = 10
    inputDistribution: str | list[str] = "standard_normal"
    noiseDistribution: str | list[str] = "deterministic"
    noiseLevel: float | list[float] = 0
    seed: int | list[int] = 0

class RandomNoiseDatasetParams(NodeParamsBase):
    inputDim: int | list[int] = 10
    outputDim: int | list[int] = 1
    inputDistribution: str | list[str] = "standard_normal"
    outputDistribution: str | list[str] = "deterministic"
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    noiseLevel: float | list[float] = 0
    alpha: float | list[float] = 1
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class RankAlignedInitializationParams(NodeParamsBase):
    basisMode: str | list[str] = "random_orthogonal"
    structuredProfile: str | list[str] = "robust_v1"
    amplitude: float | list[float] = 0.0001
    scale: float | list[float] = 1.2
    singularRatio: float | list[float] = 1.3
    frequencyRatio: float | list[float] = 2
    perturbationScale: float | list[float] = 0.05
    inputRotationAngleRadians: float | list[float] = 3
    inputRotationPlane: int | list[int] = 1
    outputRotationAngleRadians: float | list[float] = 1.7
    outputRotationPlane: int | list[int] = 2
    seed: int | list[int] = 0

class ReactionDiffusionDatasetParams(NodeParamsBase):
    contextFrames: int | list[int] = 4
    channels: int | list[int] = 1
    gridSize: int | list[int] = 16
    trainSize: int | list[int] = 512
    testSize: int | list[int] = 128
    warmupSteps: int | list[int] = 40
    dt: float | list[float] = 0.05
    diffusionCoeff: float | list[float] = 0.2
    reactionRate: float | list[float] = 1
    velocityX: float | list[float] = 0.5
    velocityY: float | list[float] = 0.2
    icScale: float | list[float] = 0.5
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class RegressorParams(NodeParamsBase):
    fitNonce: int | list[int] = 0

class RelationTupleDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 64
    contextLength: int | list[int] = 24
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"
    relationMode: str | list[str] = "forward"

class RelativePoseEncoderLayerParams(NodeParamsBase):
    inFeatures: int | list[int] = 24
    outFeatures: int | list[int] = 64
    bias: int | list[int] = 1
    seed: int | list[int] = 11

class ReshapeParams(NodeParamsBase):
    reshapeRule: str | list[str] = "b t d -> b t d"
    shapeHint: str | list[str] = "split heads"
    ioMode: str | list[str] = "input-output"
    levelMode: str | list[str] = "high"

class ResidualLnModelParams(NodeParamsBase):
    dim: int | list[int] = 256
    depth: int | list[int] = 100
    alpha: float | list[float] = 1
    lnMode: str | list[str] = "pre_ln"
    activation: str | list[str] = "relu"
    seed: int | list[int] = 0

class ResnetModelParams(NodeParamsBase):
    variant: str | list[str] = "resnet18"
    baseChannels: int | list[int] = 32
    blocksStage1: int | list[int] = 2
    blocksStage2: int | list[int] = 2
    blocksStage3: int | list[int] = 2
    blocksStage4: int | list[int] = 2
    kernelSize: int | list[int] = 3
    seed: int | list[int] = 0
    specCodeName: str | list[str] = "resnetModelSpec"

class RmsNormLayerParams(NodeParamsBase):
    normalizedShape: int | list[int] = 64
    eps: float | list[float] = 0.000001
    elementwiseAffine: int | list[int] = 1

class RotaryEmbedLayerParams(NodeParamsBase):
    rotaryDim: int | list[int] = 64
    thetaBase: int | list[int] = 10000
    seed: int | list[int] = 0

class RwkvTimeMixTokenModelParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    embedDim: int | list[int] = 32
    contextLength: int | list[int] = 8
    seed: int | list[int] = 0
    localMixingKernel: int | list[int] = 0
    depth: int | list[int] = 2

class SaxeInitializationParams(NodeParamsBase):
    amplitude: float | list[float] = 0.01

class ScanDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 64
    contextLength: int | list[int] = 24
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"

class SeriesEndpointGapParams(NodeParamsBase):
    pass

class SgdOptimizerParams(NodeParamsBase):
    learningRate: float | list[float] = 0.01
    momentum: float | list[float] = 0
    weightDecay: float | list[float] = 0

class ShampooOptimizerParams(NodeParamsBase):
    learningRate: float | list[float] = 0.01
    momentum: float | list[float] = 0
    epsilon: float | list[float] = 1e-8
    weightDecay: float | list[float] = 0
    preconditionFrequency: int | list[int] = 10
    maxPreconditionerDim: int | list[int] = 1024

class ShapeCheckerParams(NodeParamsBase):
    pass

class ShapeWorldDatasetParams(NodeParamsBase):
    trainSize: int | list[int] = 2048
    testSize: int | list[int] = 512
    initSeed: int | list[int] = 0
    seed: int | list[int] = 0
    flattenOutput: bool | list[bool] = False
    samplingMode: str | list[str] = "fixed"
    specCodeName: str | list[str] = "shape_world_datasetSpec"
    imageSize: int | list[int] = 32
    noiseLevel: float | list[float] = 0.04

class SignsgdOptimizerParams(NodeParamsBase):
    learningRate: float | list[float] = 0.001
    weightDecay: float | list[float] = 0

class SlotAttentionTokenModelParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    embedDim: int | list[int] = 32
    contextLength: int | list[int] = 8
    seed: int | list[int] = 0
    localMixingKernel: int | list[int] = 0
    numSlots: int | list[int] = 4
    slotIters: int | list[int] = 3

class SmallInceptionCifarModelParams(NodeParamsBase):
    seed: int | list[int] = 0
    specCodeName: str | list[str] = "smallInceptionCifarModelSpec"

class SmoothingCurveParams(NodeParamsBase):
    sigma: int | list[int] = 1
    logScaleX: bool | list[bool] = False
    logScaleY: bool | list[bool] = False

class SoapOptimizerParams(NodeParamsBase):
    learningRate: float | list[float] = 0.0003
    beta1: float | list[float] = 0.9
    beta2: float | list[float] = 0.95
    epsilon: float | list[float] = 1e-8
    weightDecay: float | list[float] = 0
    preconditionFrequency: int | list[int] = 10
    maxPreconditionerDim: int | list[int] = 1024

class SoftmaxParams(NodeParamsBase):
    dimension: int | list[int] = -1
    ioMode: str | list[str] = "input-output"
    levelMode: str | list[str] = "high"

class StatisticsParams(NodeParamsBase):
    einsumSubscripts: str | list[str] = "ab->b"
    reductionOp: str | list[str] = "mean"

class Statistics2Params(NodeParamsBase):
    einsumSubscripts: str | list[str] = "ij,ik->jk"
    pairReduction: str | list[str] = "dot"

class SvdParams(NodeParamsBase):
    representationId: str | list[str] = ""
    removeMean: bool | list[bool] = False

class SweepDataTableParams(NodeParamsBase):
    pass

class SymbolicFuncDatasetParams(NodeParamsBase):
    equationLatex: str | list[str] = "\\exp(\\sin(\\pi x_1) + x_2^2)"
    inputDim: int | list[int] = 2
    outputDim: int | list[int] = 1
    inputDistribution: str | list[str] = "standard_normal"
    evaluationPrecision: str | list[str] = "input"
    outputDistribution: str | list[str] = "deterministic"
    trainSize: int | list[int] = 500
    testSize: int | list[int] = 0
    noiseLevel: float | list[float] = 0
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class SymmetrizedMlpInitParams(NodeParamsBase):
    tau: float | list[float] = 1

class SyntheticPlaygroundDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 64
    contextLength: int | list[int] = 32
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"
    playgroundFamily: str | list[str] = "depo"
    depoWindow: int | list[int] = 4
    manoModulus: int | list[int] = 17
    lanoNestingDepth: int | list[int] = 4

class TableVizParams(NodeParamsBase):
    logScaleX: bool | list[bool] = False
    logScaleY: bool | list[bool] = False

class TeacherDatasetParams(NodeParamsBase):
    samplingMode: str | list[str] = "fixed"

class TensorAddParams(NodeParamsBase):
    pass

class TensorConcatParams(NodeParamsBase):
    inputCount: int | list[int] = 2
    concatDimension: int | list[int] = 0

class TensorConstantParams(NodeParamsBase):
    init: str | list[str] = "zero"
    initSeed: int | list[int] = 0

class TensorLinspaceParams(NodeParamsBase):
    start: int | list[int] = 0
    end: int | list[int] = 1
    numPoints: int | list[int] = 8
    space: str | list[str] = "linear"

class TensorReaderParams(NodeParamsBase):
    pass

class TensorSelectorParams(NodeParamsBase):
    selectedTensorKey: str | list[str] = ""
    tensorSelectorSweeping: bool | list[bool] = False
    tensorSelectorSweepSeq: int | list[int] = 0

class TensorSlicingParams(NodeParamsBase):
    pass

class TensorSplitterParams(NodeParamsBase):
    splitDimension: int | list[int] = -1
    numParts: int | list[int] = 3
    ioMode: str | list[str] = "input-output"
    levelMode: str | list[str] = "high"

class TensorStackParams(NodeParamsBase):
    inputCount: int | list[int] = 2

class TensorViz0dParams(NodeParamsBase):
    pass

class TensorViz1dParams(NodeParamsBase):
    plot1dStyle: str | list[str] = "line"
    plot1dLineSort: str | list[str] = "original"
    histBins: int | list[int] = 20
    plot2dStyle: str | list[str] = "scatter"
    plot2dScatterAxis: int | list[int] = 1
    plot2dScatterI1: int | list[int] = 0
    plot2dScatterI2: int | list[int] = 1

class TensorViz2dParams(NodeParamsBase):
    plot1dStyle: str | list[str] = "line"
    plot1dLineSort: str | list[str] = "original"
    histBins: int | list[int] = 20
    plot2dStyle: str | list[str] = "scatter"
    plot2dScatterAxis: int | list[int] = 1
    plot2dScatterI1: int | list[int] = 0
    plot2dScatterI2: int | list[int] = 1

class TensorVizGeneralParams(NodeParamsBase):
    plot1dStyle: str | list[str] = "line"
    plot1dLineSort: str | list[str] = "original"
    histBins: int | list[int] = 20
    plot2dStyle: str | list[str] = "scatter"
    plot2dScatterAxis: int | list[int] = 1
    plot2dScatterI1: int | list[int] = 0
    plot2dScatterI2: int | list[int] = 1

class TensorVizScatterParams(NodeParamsBase):
    pass

class TinyshakespeareLmDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 256
    contextLength: int | list[int] = 32
    trainSize: int | list[int] = 4000
    testSize: int | list[int] = 0
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    stride: int | list[int] = 1

class TinystoriesDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 256
    contextLength: int | list[int] = 64
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    initSeed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"
    dataSource: str | list[str] = "synthetic"
    cacheDir: str | list[str] = ""
    inspectFormat: str | list[str] = "id"
    vocabCap: float | list[float] = 256
    tokenizerMode: str | list[str] = "char"
    seqLen: int | list[int] = 512
    stride: int | list[int] = 64

class TokenPredictionDatasetParams(NodeParamsBase):
    retrievalMode: str | list[str] = "position"
    vocabSize: int | list[int] = 4
    contextLength: int | list[int] = 4
    whichToken: int | list[int] = -1
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class TrainerParams(NodeParamsBase):
    trainingSteps: int | list[int] = 1000
    logFrequency: int | list[int] = 10
    logSchedule: str | list[str] = "fixed_interval"
    logSamples: int | list[int] = 1800
    logAggregation: str | list[str] = "last_batch"
    logTiming: str | list[str] = "post_update"
    testEvaluation: str | list[str] = "log_ticks"
    trainSeed: int | list[int] = -1
    computeDevice: str | list[str] = "cpu"
    batchSize: int | list[int] = -1
    minibatchSampling: str | list[str] = "independent_step"
    minibatchSeed: int | list[int] = -1
    gradClipMaxNorm: float | list[float] = 0
    disableExtraObservables: bool | list[bool] = False

class TrainingVisualizationParams(NodeParamsBase):
    logScaleX: bool | list[bool] = False
    logScaleY: bool | list[bool] = False
    showTrainCurve: bool | list[bool] = True
    showTestCurve: bool | list[bool] = True
    yPlotMetric: str | list[str] = "loss"
    lossView: str | list[str] = "loss"

class TransformerMultiTokenModelParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    contextLength: int | list[int] = 4
    tokensPerPosition: int | list[int] = 2
    modelDim: int | list[int] = 32
    numHeads: int | list[int] = 1
    numLayers: int | list[int] = 1
    ffDim: int | list[int] = 64
    encoderBackend: str | list[str] = "pytorch"
    encoderDropout: float | list[float] = 0
    spectralNormLinears: str | list[str] = "no"
    lmLogitScale: float | list[float] = 1
    stableQkNorm: str | list[str] = "no"
    stableAttnTemperature: float | list[float] = 1
    stableAttnLogitCap: float | list[float] = 0
    stableAttnDropout: float | list[float] = 0
    tieEmbeddingLmHead: str | list[str] = "no"
    causalAttention: str | list[str] = "yes"
    seed: int | list[int] = 0

class TransformerTokenModelParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    contextLength: int | list[int] = 4
    modelDim: int | list[int] = 32
    numHeads: int | list[int] = 1
    numLayers: int | list[int] = 1
    ffDim: int | list[int] = 64
    activation: str | list[str] = "gelu"
    encoderBackend: str | list[str] = "pytorch"
    encoderDropout: float | list[float] = 0
    spectralNormLinears: str | list[str] = "no"
    lmLogitScale: float | list[float] = 1
    stableQkNorm: str | list[str] = "no"
    stableAttnTemperature: float | list[float] = 1
    stableAttnLogitCap: float | list[float] = 0
    stableAttnDropout: float | list[float] = 0
    tieEmbeddingLmHead: str | list[str] = "yes"
    causalAttention: str | list[str] = "yes"
    localMixingKernel: int | list[int] = 0
    seed: int | list[int] = 0

class UnembeddingLayerParams(NodeParamsBase):
    inFeatures: int | list[int] = 64
    outFeatures: int | list[int] = 4096
    bias: int | list[int] = 1
    seed: int | list[int] = 0

class UnetDdpmModelParams(NodeParamsBase):
    inChannels: int | list[int] = 3
    baseChannels: int | list[int] = 64
    channelMult: str | list[str] = "1,2,2"
    timeEmbedDim: int | list[int] = 128
    diffusionTimesteps: int | list[int] = 1000
    imageSize: int | list[int] = 32
    seed: int | list[int] = 0

class UniformLinearMotionDatasetParams(NodeParamsBase):
    contextLength: int | list[int] = 2
    positionDim: int | list[int] = 1
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    positionDistribution: str | list[str] = "standard_normal"
    velocityDistribution: str | list[str] = "standard_normal"
    velocityScale: float | list[float] = 1
    outputDistribution: str | list[str] = "deterministic"
    noiseLevel: float | list[float] = 0
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class UnigramDatasetParams(NodeParamsBase):
    vocabSize: int | list[int] = 100
    outputDistribution: str | list[str] = "power_law_class_probs"
    alpha: float | list[float] = 1
    contextLength: int | list[int] = 1
    trainSize: int | list[int] = 800
    testSize: int | list[int] = 200
    seed: int | list[int] = 0
    samplingMode: str | list[str] = "fixed"

class UrlNodeParams(NodeParamsBase):
    url: str | list[str] = ""

class Vgg11CifarModelParams(NodeParamsBase):
    seed: int | list[int] = 0
    specCodeName: str | list[str] = "vgg11CifarModelSpec"

class VisualizeKanParams(NodeParamsBase):
    plotPngBase64: str | list[str] = ""
    datasetSampleSplit: str | list[str] = "train"
    sampleCount: int | list[int] = 256
    plotScale: float | list[float] = 0.35
    plotMetric: str | list[str] = "backward"

class VitModelParams(NodeParamsBase):
    variant: str | list[str] = "tiny"
    patchSize: int | list[int] = 4
    hiddenDim: int | list[int] = 128
    depth: int | list[int] = 3
    numHeads: int | list[int] = 4
    seed: int | list[int] = 0
    specCodeName: str | list[str] = "vitModelSpec"

NODE_PARAM_MODELS: dict[str, type[NodeParamsBase]] = {
    "absolute_pos_embed_layer": AbsolutePosEmbedLayerParams,
    "activation": ActivationParams,
    "activation_layer": ActivationLayerParams,
    "adam_optimizer": AdamOptimizerParams,
    "adamw_optimizer": AdamwOptimizerParams,
    "advection_dataset": AdvectionDatasetParams,
    "afno_encoder_block_layer": AfnoEncoderBlockLayerParams,
    "afno_lite_spatiotemporal_model": AfnoLiteSpatiotemporalModelParams,
    "afno_patch_decode_layer": AfnoPatchDecodeLayerParams,
    "afno_patch_embed_layer": AfnoPatchEmbedLayerParams,
    "afno_spectral_mixer_layer": AfnoSpectralMixerLayerParams,
    "agent_trace_viz": AgentTraceVizParams,
    "attention_only_model": AttentionOnlyModelParams,
    "basic_calculator": BasicCalculatorParams,
    "bigram_low_rank_dataset": BigramLowRankDatasetParams,
    "binary_cross_entropy_with_logits_loss": BinaryCrossEntropyWithLogitsLossParams,
    "biography_lm_dataset": BiographyLmDatasetParams,
    "causal_mask": CausalMaskParams,
    "cifar10_dataset": Cifar10DatasetParams,
    "circle_random_walk_dataset": CircleRandomWalkDatasetParams,
    "circular_motion_dataset": CircularMotionDatasetParams,
    "cogs_dataset": CogsDatasetParams,
    "combined_model": CombinedModelParams,
    "comment": CommentParams,
    "crl_env_config": CrlEnvConfigParams,
    "crl_residual_mlp": CrlResidualMlpParams,
    "crl_trainer": CrlTrainerParams,
    "cross_entropy_loss": CrossEntropyLossParams,
    "curve_annotator": CurveAnnotatorParams,
    "curve_series_table": CurveSeriesTableParams,
    "curve_series_viz": CurveSeriesVizParams,
    "cyclic_batch_schedule": CyclicBatchScheduleParams,
    "cyclic_lr_schedule": CyclicLrScheduleParams,
    "dataset_mixer": DatasetMixerParams,
    "dataset_mixer_b": DatasetMixerBParams,
    "derivative_curve": DerivativeCurveParams,
    "deterministic_diffusion_sampler": DeterministicDiffusionSamplerParams,
    "diagonal_ssm_token_model": DiagonalSsmTokenModelParams,
    "diffusion_mse_loss": DiffusionMseLossParams,
    "diffusion_pde_dataset": DiffusionPdeDatasetParams,
    "diffusion_score_model": DiffusionScoreModelParams,
    "dimension_permutator": DimensionPermutatorParams,
    "distance_contact_layer": DistanceContactLayerParams,
    "docking_pose_viz": DockingPoseVizParams,
    "dyck_dataset": DyckDatasetParams,
    "effective_rank": EffectiveRankParams,
    "einsum": EinsumParams,
    "elementwise_transform": ElementwiseTransformParams,
    "embedding_layer": EmbeddingLayerParams,
    "energy_readout_layer": EnergyReadoutLayerParams,
    "equivariant_message_layer": EquivariantMessageLayerParams,
    "fake_tensor": FakeTensorParams,
    "flatten": FlattenParams,
    "formal_language_suite_dataset": FormalLanguageSuiteDatasetParams,
    "gated_mlp_model": GatedMlpModelParams,
    "gated_mlp_token_model": GatedMlpTokenModelParams,
    "gaussian_blob_dataset": GaussianBlobDatasetParams,
    "graph_assist_failure_overlay": GraphAssistFailureOverlayParams,
    "hole_counting_dataset": HoleCountingDatasetParams,
    "hyena_like_conv_model": HyenaLikeConvModelParams,
    "hypothesis": HypothesisParams,
    "idnns_initialization": IdnnsInitializationParams,
    "image_dataset_displayer": ImageDatasetDisplayerParams,
    "in_context_associative_recall_dataset": InContextAssociativeRecallDatasetParams,
    "information_bottleneck_dataset": InformationBottleneckDatasetParams,
    "input_sampler": InputSamplerParams,
    "interatomic_eval_viz": InteratomicEvalVizParams,
    "kan_model": KanModelParams,
    "kan_reg": KanRegParams,
    "kepler_2d_dataset": Kepler2dDatasetParams,
    "keskar_c1_c2_cnn_model": KeskarC1C2CnnModelParams,
    "l1_reg": L1RegParams,
    "l2_projection": L2ProjectionParams,
    "l2_reg": L2RegParams,
    "layer_norm_layer": LayerNormLayerParams,
    "linear_attention_model": LinearAttentionModelParams,
    "linear_dataset": LinearDatasetParams,
    "linear_layer": LinearLayerParams,
    "listops_dataset": ListopsDatasetParams,
    "local_mixing_layer": LocalMixingLayerParams,
    "lr_schedule": LrScheduleParams,
    "memorization_a_dataset": MemorizationADatasetParams,
    "memorization_b_dataset": MemorizationBDatasetParams,
    "metric_compare": MetricCompareParams,
    "mlp_model": MlpModelParams,
    "mlp_token_model": MlpTokenModelParams,
    "mnist_dataset": MnistDatasetParams,
    "model_checkpoint": ModelCheckpointParams,
    "model_weight_tensors": ModelWeightTensorsParams,
    "modular_addition_dataset": ModularAdditionDatasetParams,
    "moe_mlp_model": MoeMlpModelParams,
    "moe_mlp_token_model": MoeMlpTokenModelParams,
    "mpp_spatiotemporal_model": MppSpatiotemporalModelParams,
    "mse_loss": MseLossParams,
    "multi_hop_fact_chain_dataset": MultiHopFactChainDatasetParams,
    "muon_optimizer": MuonOptimizerParams,
    "mup_initialization": MupInitializationParams,
    "mup_lr_schedule": MupLrScheduleParams,
    "ngram_language_dataset": NgramLanguageDatasetParams,
    "numeric_hyena_model": NumericHyenaModelParams,
    "numeric_transformer_model": NumericTransformerModelParams,
    "observable_accuracy": ObservableAccuracyParams,
    "observable_activation_norm_mean": ObservableActivationNormMeanParams,
    "observable_activation_outlier_ratio": ObservableActivationOutlierRatioParams,
    "observable_activation_stats": ObservableActivationStatsParams,
    "observable_attention_entropy_mean": ObservableAttentionEntropyMeanParams,
    "observable_attention_head_sink_max": ObservableAttentionHeadSinkMaxParams,
    "observable_attention_map": ObservableAttentionMapParams,
    "observable_attention_max_weight_mean": ObservableAttentionMaxWeightMeanParams,
    "observable_attention_position_bias_ratio": ObservableAttentionPositionBiasRatioParams,
    "observable_attention_relation_score": ObservableAttentionRelationScoreParams,
    "observable_bezier_mode_connectivity": ObservableBezierModeConnectivityParams,
    "observable_capacity": ObservableCapacityParams,
    "observable_embedding_effective_rank": ObservableEmbeddingEffectiveRankParams,
    "observable_embedding_evolution": ObservableEmbeddingEvolutionParams,
    "observable_embedding_feature_drift": ObservableEmbeddingFeatureDriftParams,
    "observable_embedding_trajectory": ObservableEmbeddingTrajectoryParams,
    "observable_fourier_component": ObservableFourierComponentParams,
    "observable_gradient_norm": ObservableGradientNormParams,
    "observable_hessian_eigenvalues": ObservableHessianEigenvaluesParams,
    "observable_information_plane": ObservableInformationPlaneParams,
    "observable_last_layer_weight_norm": ObservableLastLayerWeightNormParams,
    "observable_layer_spectral_norm": ObservableLayerSpectralNormParams,
    "observable_linear_interpolation_barrier": ObservableLinearInterpolationBarrierParams,
    "observable_nearest_train_gl": ObservableNearestTrainGlParams,
    "observable_neuron_trajectory_2d": ObservableNeuronTrajectory2dParams,
    "observable_paired_generation_similarity": ObservablePairedGenerationSimilarityParams,
    "observable_relu_nonlinear_count": ObservableReluNonlinearCountParams,
    "observable_representation_alpha_req": ObservableRepresentationAlphaReqParams,
    "observable_representation_rankme": ObservableRepresentationRankmeParams,
    "observable_rp_score_sscd": ObservableRpScoreSscdParams,
    "observable_sink_attention_mass": ObservableSinkAttentionMassParams,
    "observable_train_test_gap": ObservableTrainTestGapParams,
    "observable_user": ObservableUserParams,
    "observable_viz": ObservableVizParams,
    "observable_viz_neuron_trajectory_2d": ObservableVizNeuronTrajectory2dParams,
    "observable_weight_displacement": ObservableWeightDisplacementParams,
    "observable_weight_l1": ObservableWeightL1Params,
    "observable_weight_l2": ObservableWeightL2Params,
    "observable_weight_product_sv": ObservableWeightProductSvParams,
    "pairwise_rbf_layer": PairwiseRbfLayerParams,
    "paper_classification_dataset": PaperClassificationDatasetParams,
    "parametric_path_sampler": ParametricPathSamplerParams,
    "pca": PcaParams,
    "pcfg_dataset": PcfgDatasetParams,
    "phi1_style_dataset": Phi1StyleDatasetParams,
    "prediction": PredictionParams,
    "protein_structure_comparison_viz": ProteinStructureComparisonVizParams,
    "protein_structure_displayer": ProteinStructureDisplayerParams,
    "random_input_distribution": RandomInputDistributionParams,
    "random_noise_dataset": RandomNoiseDatasetParams,
    "rank_aligned_initialization": RankAlignedInitializationParams,
    "reaction_diffusion_dataset": ReactionDiffusionDatasetParams,
    "regressor": RegressorParams,
    "relation_tuple_dataset": RelationTupleDatasetParams,
    "relative_pose_encoder_layer": RelativePoseEncoderLayerParams,
    "reshape": ReshapeParams,
    "residual_ln_model": ResidualLnModelParams,
    "resnet_model": ResnetModelParams,
    "rms_norm_layer": RmsNormLayerParams,
    "rotary_embed_layer": RotaryEmbedLayerParams,
    "rwkv_time_mix_token_model": RwkvTimeMixTokenModelParams,
    "saxe_initialization": SaxeInitializationParams,
    "scan_dataset": ScanDatasetParams,
    "series_endpoint_gap": SeriesEndpointGapParams,
    "sgd_optimizer": SgdOptimizerParams,
    "shampoo_optimizer": ShampooOptimizerParams,
    "shape_checker": ShapeCheckerParams,
    "shape_world_dataset": ShapeWorldDatasetParams,
    "signsgd_optimizer": SignsgdOptimizerParams,
    "slot_attention_token_model": SlotAttentionTokenModelParams,
    "small_inception_cifar_model": SmallInceptionCifarModelParams,
    "smoothing_curve": SmoothingCurveParams,
    "soap_optimizer": SoapOptimizerParams,
    "softmax": SoftmaxParams,
    "statistics": StatisticsParams,
    "statistics2": Statistics2Params,
    "svd": SvdParams,
    "sweep_data_table": SweepDataTableParams,
    "symbolic_func_dataset": SymbolicFuncDatasetParams,
    "symmetrized_mlp_init": SymmetrizedMlpInitParams,
    "synthetic_playground_dataset": SyntheticPlaygroundDatasetParams,
    "table_viz": TableVizParams,
    "teacher_dataset": TeacherDatasetParams,
    "tensor_add": TensorAddParams,
    "tensor_concat": TensorConcatParams,
    "tensor_constant": TensorConstantParams,
    "tensor_linspace": TensorLinspaceParams,
    "tensor_reader": TensorReaderParams,
    "tensor_selector": TensorSelectorParams,
    "tensor_slicing": TensorSlicingParams,
    "tensor_splitter": TensorSplitterParams,
    "tensor_stack": TensorStackParams,
    "tensor_viz_0d": TensorViz0dParams,
    "tensor_viz_1d": TensorViz1dParams,
    "tensor_viz_2d": TensorViz2dParams,
    "tensor_viz_general": TensorVizGeneralParams,
    "tensor_viz_scatter": TensorVizScatterParams,
    "tinyshakespeare_lm_dataset": TinyshakespeareLmDatasetParams,
    "tinystories_dataset": TinystoriesDatasetParams,
    "token_prediction_dataset": TokenPredictionDatasetParams,
    "trainer": TrainerParams,
    "training_visualization": TrainingVisualizationParams,
    "transformer_multi_token_model": TransformerMultiTokenModelParams,
    "transformer_token_model": TransformerTokenModelParams,
    "unembedding_layer": UnembeddingLayerParams,
    "unet_ddpm_model": UnetDdpmModelParams,
    "uniform_linear_motion_dataset": UniformLinearMotionDatasetParams,
    "unigram_dataset": UnigramDatasetParams,
    "url_node": UrlNodeParams,
    "vgg11_cifar_model": Vgg11CifarModelParams,
    "visualize_kan": VisualizeKanParams,
    "vit_model": VitModelParams,
}


def param_model_for(node_type: str) -> type[NodeParamsBase]:
    return NODE_PARAM_MODELS.get(str(node_type), NodeParamsBase)


def validate_node_params(node_type: str, data: dict[str, Any] | None) -> NodeParamsBase:
    return param_model_for(str(node_type)).model_validate(data or {})
