/**
 * Code-tab Python for dataset nodes beyond linear / vision / unigram / modular.
 * Reuses the same ``generate*SpecCode`` bodies as **View/edit code** where they return numpy dicts or arrays.
 */
import type { Node as RFNode } from "@xyflow/react";
import { defaultBigramLowRankDatasetData, type BigramLowRankDatasetNodeData } from "../components/nodes/bigramLowRankDatasetDefaults";
import { defaultCircleRandomWalkDatasetData, type CircleRandomWalkDatasetNodeData } from "../components/nodes/circleRandomWalkDatasetDefaults";
import { defaultCircularMotionDatasetData, type CircularMotionDatasetNodeData } from "../components/nodes/circularMotionDatasetDefaults";
import { defaultKepler2dDatasetData, type Kepler2dDatasetNodeData } from "../components/nodes/kepler2dDatasetDefaults";
import { defaultDatasetMixerData, type DatasetMixerNodeData } from "../components/nodes/datasetMixerDefaults";
import { defaultInContextAssociativeRecallDatasetData, type InContextAssociativeRecallDatasetNodeData } from "../components/nodes/inContextAssociativeRecallDatasetDefaults";
import { defaultInputSamplerData, type InputSamplerNodeData } from "../components/nodes/inputSamplerDefaults";
import {
  defaultLinearDatasetData,
  type InputDistributionId,
  type LinearDatasetNodeData,
  type OutputDistributionId,
} from "../components/nodes/linearDatasetDefaults";
import { defaultModularAdditionDatasetData, type ModularAdditionDatasetNodeData } from "../components/nodes/modularAdditionDatasetDefaults";
import { defaultPdeFieldDatasetData, type PdeFieldDatasetKind, type PdeFieldDatasetNodeData } from "../components/nodes/pdeFieldDatasetDefaults";
import { defaultRandomInputDistributionData, type RandomInputDistributionNodeData } from "../components/nodes/randomInputDistributionDefaults";
import { defaultSymbolicFuncDatasetData, type SymbolicFuncDatasetNodeData } from "../components/nodes/symbolicFuncDatasetDefaults";
import { defaultTeacherDatasetData, type TeacherDatasetNodeData } from "../components/nodes/teacherDatasetDefaults";
import { defaultTokenPredictionDatasetData, type TokenPredictionDatasetNodeData } from "../components/nodes/tokenPredictionDatasetDefaults";
import { defaultUnigramDatasetData, type UnigramDatasetNodeData } from "../components/nodes/unigramDatasetDefaults";
import {
  defaultToyLanguageDatasetData,
  TOY_LANGUAGE_DATASET_KINDS,
  type ToyLanguageDatasetKind,
  type ToyLanguageDatasetNodeData,
} from "../components/nodes/toyLanguageDatasetDefaults";
import { defaultUniformLinearMotionDatasetData, type UniformLinearMotionDatasetNodeData } from "../components/nodes/uniformLinearMotionDatasetDefaults";
import type { CodegenContext } from "./codegenContext";
import { pySlugForNode } from "./codegenContext";
import { generateBigramLowRankDatasetSpecCode } from "./specCode/bigramLowRankDatasetSpecCode";
import { generateCircleRandomWalkDatasetSpecCode } from "./specCode/circleRandomWalkDatasetSpecCode";
import { generateCircularMotionDatasetSpecCode } from "./specCode/circularMotionDatasetSpecCode";
import { generateKepler2dDatasetSpecCode } from "./specCode/kepler2dDatasetSpecCode";
import { generateInContextAssociativeRecallDatasetSpecCode } from "./specCode/inContextAssociativeRecallDatasetSpecCode";
import { generatePdeFieldDatasetSpecCode, defaultParamOrderForPdeKind, DEFAULT_PDE_FIELD_DATASET_SPEC_NAME } from "./specCode/pdeFieldDatasetSpecCode";
import { generateRandomInputDistributionSpecCode } from "./specCode/randomInputDistributionSpecCode";
import { generateTeacherDatasetSpecCode } from "./specCode/teacherDatasetSpecCode";
import { generateTokenPredictionDatasetSpecCode } from "./specCode/tokenPredictionDatasetSpecCode";
import { generateToyLanguageDatasetSpecCode } from "./specCode/toyLanguageDatasetSpecCode";
import { generateUniformLinearMotionDatasetSpecCode } from "./specCode/uniformLinearMotionDatasetSpecCode";
import { wrapDictReturningSpecAsLoaders } from "./specDictPackToLoaders";

function firstScalar<T>(v: T | T[] | undefined | null, fallback: T): T {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return (v.length ? v[0]! : fallback) as T;
  return v as T;
}

function safeSpecFn(pySym: string): string {
  const t = pySym.replace(/[^a-zA-Z0-9_]+/g, "_");
  return t.match(/^[0-9]/) ? `spec_${t}` : `spec_${t}`;
}

function upstreamSource(ctx: CodegenContext, targetId: string, targetHandle: string): RFNode | undefined {
  const e = ctx.edges.find((x) => x.target === targetId && (x.targetHandle ?? "") === targetHandle);
  if (!e) return undefined;
  return ctx.nodes.find((n) => n.id === e.source);
}

const TOKEN_LONG = { xTrain: "int64", yTrain: "int64", xTest: "int64", yTest: "int64" } as const;
const FLOAT_PACK = { xTrain: "float32", yTrain: "float32", xTest: "float32", yTest: "float32" } as const;

function toPythonDictLiteral(obj: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(obj)) {
    const pk = JSON.stringify(k);
    if (typeof v === "string") parts.push(`${pk}: ${JSON.stringify(v)}`);
    else if (typeof v === "number") parts.push(`${pk}: ${Number.isFinite(v) ? String(v) : "float('nan')"}`);
    else if (typeof v === "boolean") parts.push(`${pk}: ${v ? "True" : "False"}`);
    else parts.push(`${pk}: ${JSON.stringify(v)}`);
  }
  return `{\n    ${parts.join(",\n    ")},\n}`;
}

export function buildLinearLikeDatasetTorch(
  pySym: string,
  title: string,
  nodeType: string,
  raw: Record<string, unknown>,
): string {
  const defs = defaultLinearDatasetData();
  const d = { ...defs, ...(raw as Partial<LinearDatasetNodeData>) } as LinearDatasetNodeData;
  const inputDim = firstScalar(d.inputDim, defs.inputDim as number);
  const outputDim = firstScalar(d.outputDim, defs.outputDim as number);
  const trainSize = firstScalar(d.trainSize, defs.trainSize as number);
  const testSize = firstScalar(d.testSize, defs.testSize as number);
  const noiseLevel = firstScalar(d.noiseLevel, defs.noiseLevel as number);
  const seed = firstScalar(d.seed, defs.seed as number);
  const inputDistribution = firstScalar(
    d.inputDistribution,
    defs.inputDistribution as InputDistributionId,
  ) as string;
  const outputDistribution = firstScalar(
    d.outputDistribution,
    defs.outputDistribution as OutputDistributionId,
  ) as string;
  const alpha = firstScalar(d.alpha, defs.alpha as number);
  const vocabSize = firstScalar(d.vocabSize ?? d.outputDim, defs.outputDim as number);

  const isMem = nodeType === "memorization_a_dataset";
  const isMemB = nodeType === "memorization_b_dataset";
  const isRandomNoise = nodeType === "random_noise_dataset";

  if (isMemB) {
    return `# === ${title} (${nodeType}) ===
# Runnable toy classification: independent random input and output classes (same prior family on ranks); x is one-hot.
import torch
from torch.utils.data import DataLoader, TensorDataset


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(${JSON.stringify(seed)})
    vocab = ${vocabSize}
    d_in = vocab
    classes = vocab
    n_train = ${trainSize}
    n_test = ${testSize}
    out_dist = ${JSON.stringify(outputDistribution)}
    alpha = float(${JSON.stringify(alpha)})

    def class_probs(k: int) -> torch.Tensor:
        idx = torch.arange(1, k + 1, dtype=torch.float32, device=device)
        if out_dist == "power_law_class_probs":
            p = idx.pow(-max(alpha, 1e-8))
        elif out_dist == "exponential_class_probs":
            p = torch.exp(-max(alpha, 1e-8) * idx)
        else:
            p = torch.ones(k, dtype=torch.float32, device=device)
        return p / p.sum()

    def sample_one_hot(n: int, k: int) -> torch.Tensor:
        if n <= 0:
            return torch.zeros(0, k, device=device)
        p = class_probs(k)
        c = torch.multinomial(p, n, replacement=True, generator=g)
        oh = torch.zeros(n, k, device=device)
        oh[torch.arange(n, device=device), c] = 1.0
        return oh

    x_train = sample_one_hot(n_train, d_in)
    y_train = torch.multinomial(class_probs(classes), n_train, replacement=True, generator=g).to(torch.long)
    x_test = sample_one_hot(n_test, d_in) if n_test > 0 else torch.zeros(0, d_in, device=device)
    y_test = (
        torch.multinomial(class_probs(classes), n_test, replacement=True, generator=g).to(torch.long)
        if n_test > 0
        else torch.zeros(0, dtype=torch.long, device=device)
    )

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test) if n_test > 0 else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds else None
    return train_loader, test_loader
`;
  }

  if (isMem) {
    return `# === ${title} (${nodeType}) ===
# Runnable toy classification data (labels are independent of x, sampled from a class prior).
import torch
from torch.utils.data import DataLoader, TensorDataset


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(${JSON.stringify(seed)})
    d_in = ${inputDim}
    classes = ${outputDim}
    n_train = ${trainSize}
    n_test = ${testSize}
    out_dist = ${JSON.stringify(outputDistribution)}
    alpha = float(${JSON.stringify(alpha)})

    def sample_x(n: int) -> torch.Tensor:
        if n <= 0:
            return torch.zeros(0, d_in)
        if ${JSON.stringify(inputDistribution)} == "uniform_neg1_1":
            return torch.empty(n, d_in, generator=g, device="cpu").uniform_(-1.0, 1.0).to(device)
        if ${JSON.stringify(inputDistribution)} == "uniform_0_1":
            return torch.empty(n, d_in, generator=g, device="cpu").uniform_(0.0, 1.0).to(device)
        return torch.randn(n, d_in, generator=g, device=device)

    def class_probs() -> torch.Tensor:
        idx = torch.arange(1, classes + 1, dtype=torch.float32, device=device)
        if out_dist == "power_law_class_probs":
            p = idx.pow(-max(alpha, 1e-8))
        elif out_dist == "exponential_class_probs":
            p = torch.exp(-max(alpha, 1e-8) * idx)
        else:
            p = torch.ones(classes, dtype=torch.float32, device=device)
        return p / p.sum()

    x_train = sample_x(n_train)
    probs = class_probs()
    y_train = torch.multinomial(probs, n_train, replacement=True, generator=g).to(torch.long)
    x_test = sample_x(n_test) if n_test > 0 else torch.zeros(0, d_in, device=device)
    y_test = (
        torch.multinomial(probs, n_test, replacement=True, generator=g).to(torch.long)
        if n_test > 0
        else torch.zeros(0, dtype=torch.long, device=device)
    )

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test) if n_test > 0 else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds else None
    return train_loader, test_loader
`;
  }

  if (isRandomNoise) {
    return `# === ${title} (${nodeType}) ===
# Random-noise vectors: x and y are sampled independently from the same base input distribution.
import torch
from torch.utils.data import DataLoader, TensorDataset


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(${JSON.stringify(seed)})
    d_in = ${inputDim}
    d_out = ${outputDim}
    n_train = ${trainSize}
    n_test = ${testSize}
    in_dist = ${JSON.stringify(inputDistribution)}

    def sample_vec(n: int, d: int) -> torch.Tensor:
        if n <= 0:
            return torch.zeros(0, d, device=device)
        if in_dist == "uniform_neg1_1":
            return torch.empty(n, d, generator=g, device=device).uniform_(-1.0, 1.0)
        if in_dist == "uniform_0_1":
            return torch.empty(n, d, generator=g, device=device).uniform_(0.0, 1.0)
        return torch.randn(n, d, generator=g, device=device)

    x_train = sample_vec(n_train, d_in)
    y_train = sample_vec(n_train, d_out)

    if n_test > 0:
        x_test = sample_vec(n_test, d_in)
        y_test = sample_vec(n_test, d_out)
    else:
        x_test = torch.zeros(0, d_in, device=device)
        y_test = torch.zeros(0, d_out, device=device)

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test) if n_test > 0 else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds else None
    return train_loader, test_loader
`;
  }




  return `# === ${title} (${nodeType}) ===
# Matches ComfyResearch linear dataset: y = x @ W.T + σ·ε with W_ij ~ N(0, 1/√d_in), ε ~ N(0,1) if additive Gaussian.
import torch
from torch.utils.data import DataLoader, TensorDataset


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(${JSON.stringify(seed)})
    d_in = ${inputDim}
    d_out = ${outputDim}
    n_train = ${trainSize}
    n_test = ${testSize}
    sigma = float(${noiseLevel})
    in_dist = ${JSON.stringify(inputDistribution)}
    out_dist = ${JSON.stringify(outputDistribution)}

    W = torch.randn(d_out, d_in, generator=g, device=device) / (max(d_in, 1) ** 0.5)

    def sample_x(n: int) -> torch.Tensor:
        if n <= 0:
            return torch.zeros(0, d_in, device=device)
        if in_dist == "uniform_neg1_1":
            return torch.empty(n, d_in, generator=g, device=device).uniform_(-1.0, 1.0)
        if in_dist == "uniform_0_1":
            return torch.empty(n, d_in, generator=g, device=device).uniform_(0.0, 1.0)
        return torch.randn(n, d_in, generator=g, device=device)

    x_train = sample_x(n_train)
    y_train = x_train @ W.T
    if out_dist == "additive_gaussian" and sigma > 0:
        y_train = y_train + sigma * torch.randn_like(y_train, generator=g)

    if n_test > 0:
        x_test = sample_x(n_test)
        y_test = x_test @ W.T
        if out_dist == "additive_gaussian" and sigma > 0:
            y_test = y_test + sigma * torch.randn_like(y_test, generator=g)
    else:
        x_test = torch.zeros(0, d_in, device=device)
        y_test = torch.zeros(0, d_out, device=device)

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test) if n_test > 0 else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds else None
    return train_loader, test_loader
`;
}

export function buildModularAdditionDatasetTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultModularAdditionDatasetData();
  const d = { ...defs, ...(raw as Partial<ModularAdditionDatasetNodeData>) } as ModularAdditionDatasetNodeData;
  const modulus = Math.max(2, Math.floor(firstScalar(d.modulus, defs.modulus as number)));
  const trainFraction = firstScalar(d.trainFraction, defs.trainFraction as number);
  const seed = Math.floor(firstScalar(d.seed, defs.seed as number));

  return `# === ${title} (modular_addition_dataset) ===
# All ordered pairs (i, j) with i, j in {{0,…,p-1}}; target y = (i + j) mod p. Total p^2 samples; shuffle then split.
# Matches ComfyResearch /api/train: train rows are the first n_train after permutation; test is the remaining p^2 - n_train rows.
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    p = int(${modulus})
    frac = float(${JSON.stringify(trainFraction)})
    seed = int(${seed})
    if p < 2:
        raise ValueError("modulus must be >= 2")
    if not (0.0 < frac < 1.0):
        raise ValueError("train_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    a, b = np.meshgrid(np.arange(p, dtype=np.int64), np.arange(p, dtype=np.int64), indexing="ij")
    x_all = np.stack([a.reshape(-1), b.reshape(-1)], axis=1)
    y_all = ((x_all[:, 0] + x_all[:, 1]) % p).astype(np.int64)
    perm = rng.permutation(x_all.shape[0])
    x_all = x_all[perm]
    y_all = y_all[perm]
    n_train = int(round(frac * x_all.shape[0]))
    n_train = min(max(n_train, 1), x_all.shape[0])
    n_test = int(x_all.shape[0] - n_train)
    x_train = x_all[:n_train]
    y_train = y_all[:n_train]
    if n_test <= 0:
        x_test = np.zeros((0, 2), dtype=np.int64)
        y_test = np.zeros((0,), dtype=np.int64)
    else:
        x_test = x_all[n_train:]
        y_test = y_all[n_train:]

    x_train_t = torch.as_tensor(x_train, dtype=torch.long, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    train_ds = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if n_test <= 0:
        test_loader = None
    else:
        x_test_t = torch.as_tensor(x_test, dtype=torch.long, device=device)
        y_test_t = torch.as_tensor(y_test, dtype=torch.long, device=device)
        test_ds = TensorDataset(x_test_t, y_test_t)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, test_loader
`;
}

export function buildUnigramDatasetTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultUnigramDatasetData();
  const d = { ...defs, ...(raw as Partial<UnigramDatasetNodeData>) } as UnigramDatasetNodeData;
  const vocabSize = firstScalar(d.vocabSize, defs.vocabSize as number);
  const outputDistribution = firstScalar(d.outputDistribution, defs.outputDistribution as string);
  const alpha = firstScalar(d.alpha, defs.alpha as number);
  const contextLength = firstScalar(d.contextLength, defs.contextLength as number);
  const trainSize = firstScalar(d.trainSize, defs.trainSize as number);
  const testSize = firstScalar(d.testSize, defs.testSize as number);
  const seed = firstScalar(d.seed, defs.seed as number);

  return `# === ${title} (unigram_dataset) ===
# Matches ComfyResearch unigram dataset: i.i.d. token contexts and labels sampled from one rank-based class prior.
import torch
from torch.utils.data import DataLoader, TensorDataset


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(${JSON.stringify(seed)})
    vocab = int(${JSON.stringify(vocabSize)})
    ctx_len = int(${JSON.stringify(contextLength)})
    n_train = int(${JSON.stringify(trainSize)})
    n_test = int(${JSON.stringify(testSize)})
    out_dist = ${JSON.stringify(outputDistribution)}
    alpha = float(${JSON.stringify(alpha)})

    if vocab < 2:
        raise ValueError("vocab_size must be >= 2")
    if ctx_len < 1:
        raise ValueError("context_length must be >= 1")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    def token_probs() -> torch.Tensor:
        idx = torch.arange(1, vocab + 1, dtype=torch.float32, device="cpu")
        if out_dist == "power_law_class_probs":
            p = idx.pow(-max(alpha, 1e-8))
        elif out_dist == "exponential_class_probs":
            p = torch.exp(-max(alpha, 1e-8) * idx)
        elif out_dist == "uniform_class_probs":
            p = torch.ones(vocab, dtype=torch.float32, device="cpu")
        else:
            raise ValueError("outputDistribution must be uniform_class_probs, power_law_class_probs, or exponential_class_probs")
        return p / p.sum()

    probs = token_probs()
    x_train = torch.multinomial(probs, n_train * ctx_len, replacement=True, generator=g).reshape(n_train, ctx_len).to(torch.long)
    y_train = torch.multinomial(probs, n_train, replacement=True, generator=g).to(torch.long)
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    if n_test > 0:
        x_test = torch.multinomial(probs, n_test * ctx_len, replacement=True, generator=g).reshape(n_test, ctx_len).to(torch.long)
        y_test = torch.multinomial(probs, n_test, replacement=True, generator=g).to(torch.long)
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        test_ds = TensorDataset(x_test, y_test)
    else:
        test_ds = None

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds else None
    return train_loader, test_loader
`;
}

export function buildTokenPredictionDatasetLoaders(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultTokenPredictionDatasetData();
  const d = { ...defs, ...(raw as Partial<TokenPredictionDatasetNodeData>) } as TokenPredictionDatasetNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [];
  const fn = safeSpecFn(pySym);
  const body = generateTokenPredictionDatasetSpecCode(d, order, fn);
  return wrapDictReturningSpecAsLoaders({ title, nodeType: "token_prediction_dataset", pySym, specBody: body, specFnName: fn, plan: TOKEN_LONG });
}

export function buildCircleRandomWalkDatasetLoaders(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultCircleRandomWalkDatasetData();
  const d = { ...defs, ...(raw as Partial<CircleRandomWalkDatasetNodeData>) } as CircleRandomWalkDatasetNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [];
  const fn = safeSpecFn(pySym);
  const body = generateCircleRandomWalkDatasetSpecCode(d, order, fn);
  return wrapDictReturningSpecAsLoaders({ title, nodeType: "circle_random_walk_dataset", pySym, specBody: body, specFnName: fn, plan: TOKEN_LONG });
}

export function buildCircularMotionDatasetLoaders(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultCircularMotionDatasetData();
  const d = { ...defs, ...(raw as Partial<CircularMotionDatasetNodeData>) } as CircularMotionDatasetNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [];
  const fn = safeSpecFn(pySym);
  const body = generateCircularMotionDatasetSpecCode(d, order, fn);
  return wrapDictReturningSpecAsLoaders({ title, nodeType: "circular_motion_dataset", pySym, specBody: body, specFnName: fn, plan: TOKEN_LONG });
}

export function buildKepler2dDatasetLoaders(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultKepler2dDatasetData();
  const d = { ...defs, ...(raw as Partial<Kepler2dDatasetNodeData>) } as Kepler2dDatasetNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [];
  const fn = safeSpecFn(pySym);
  const body = generateKepler2dDatasetSpecCode(d, order, fn);
  return wrapDictReturningSpecAsLoaders({ title, nodeType: "kepler_2d_dataset", pySym, specBody: body, specFnName: fn, plan: FLOAT_PACK });
}

export function buildUniformLinearMotionDatasetLoaders(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultUniformLinearMotionDatasetData();
  const d = { ...defs, ...(raw as Partial<UniformLinearMotionDatasetNodeData>) } as UniformLinearMotionDatasetNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [];
  const fn = safeSpecFn(pySym);
  const body = generateUniformLinearMotionDatasetSpecCode(d, order, fn);
  return wrapDictReturningSpecAsLoaders({ title, nodeType: "uniform_linear_motion_dataset", pySym, specBody: body, specFnName: fn, plan: FLOAT_PACK });
}

export function buildInContextAssociativeRecallDatasetLoaders(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultInContextAssociativeRecallDatasetData();
  const d = { ...defs, ...(raw as Partial<InContextAssociativeRecallDatasetNodeData>) } as InContextAssociativeRecallDatasetNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [];
  const fn = safeSpecFn(pySym);
  const body = generateInContextAssociativeRecallDatasetSpecCode(d, order, fn);
  return wrapDictReturningSpecAsLoaders({ title, nodeType: "in_context_associative_recall_dataset", pySym, specBody: body, specFnName: fn, plan: TOKEN_LONG });
}

export function buildBigramLowRankDatasetLoaders(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultBigramLowRankDatasetData();
  const d = { ...defs, ...(raw as Partial<BigramLowRankDatasetNodeData>) } as BigramLowRankDatasetNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [];
  const fn = safeSpecFn(pySym);
  const body = generateBigramLowRankDatasetSpecCode(d, order, fn);
  return wrapDictReturningSpecAsLoaders({ title, nodeType: "bigram_low_rank_dataset", pySym, specBody: body, specFnName: fn, plan: TOKEN_LONG });
}

export function buildTeacherDatasetLoaders(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultTeacherDatasetData();
  const d = { ...defs, ...(raw as Partial<TeacherDatasetNodeData>) } as TeacherDatasetNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [];
  const fn = safeSpecFn(pySym);
  const body = generateTeacherDatasetSpecCode(d, order, fn);
  return wrapDictReturningSpecAsLoaders({
    title,
    nodeType: "teacher_dataset",
    pySym,
    specBody: body,
    specFnName: fn,
    plan: FLOAT_PACK,
    extraHeaderLines: ["Same stub template as View/edit code (teacher_labels_numpy demo)."],
  });
}

export function buildPdeFieldDatasetLoaders(pySym: string, title: string, kind: PdeFieldDatasetKind, raw: Record<string, unknown>): string {
  const defs = defaultPdeFieldDatasetData(kind);
  const d = { ...defs, ...(raw as Partial<PdeFieldDatasetNodeData>) } as PdeFieldDatasetNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : defaultParamOrderForPdeKind(kind);
  const specNm = safeSpecFn(pySym);
  const body = generatePdeFieldDatasetSpecCode(kind, d, order, specNm);
  const sampleBase = specNm.trim() || DEFAULT_PDE_FIELD_DATASET_SPEC_NAME;
  const sampleFn = `sample_${sampleBase}_arrays`;
  const tr = Math.floor(firstScalar(d.trainSize, defs.trainSize as number));
  const te = Math.floor(firstScalar(d.testSize, defs.testSize as number));
  const iseed = Math.floor(firstScalar(d.initSeed ?? d.seed, 0));
  return `# === ${title} (${kind}) ===
# Uses \`build_pde_field_arrays\` exactly like the spec cell. Repo root on PYTHONPATH.
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

${body}

def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    x_tr, y_tr, x_te, y_te = ${sampleFn}(int(${JSON.stringify(tr)}), int(${JSON.stringify(te)}), int(${JSON.stringify(iseed)}))
    x_train_t = torch.as_tensor(x_tr, device=device, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_tr, device=device, dtype=torch.float32)
    if x_te is not None and y_te is not None and int(np.asarray(x_te).shape[0]) > 0:
        x_test_t = torch.as_tensor(x_te, device=device, dtype=torch.float32)
        y_test_t = torch.as_tensor(y_te, device=device, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size)
    else:
        test_loader = None
    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
`;
}

export function buildToyLanguageDatasetLoaders(
  kind: ToyLanguageDatasetKind,
  pySym: string,
  title: string,
  raw: Record<string, unknown>,
): string {
  const defs = defaultToyLanguageDatasetData(kind);
  const d0 = { ...defs, ...(raw as Partial<ToyLanguageDatasetNodeData>) } as ToyLanguageDatasetNodeData;
  const specId = `Cr_tl_${pySym.replace(/[^a-zA-Z0-9_]+/g, "_")}`;
  const d = { ...d0, specCodeName: specId };
  const body = generateToyLanguageDatasetSpecCode(kind, d, specId);
  return wrapDictReturningSpecAsLoaders({
    title,
    nodeType: kind,
    pySym,
    specBody: body,
    specFnName: specId,
    plan: TOKEN_LONG,
    extraHeaderLines: [
      "Inlined generator matches ``comfy_research/engine/toy_language_*.py`` (see spec docstring). Repo root on PYTHONPATH.",
    ],
  });
}

export function buildSymbolicFuncDatasetLoaders(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultSymbolicFuncDatasetData();
  const d = { ...defs, ...(raw as Partial<SymbolicFuncDatasetNodeData>) } as SymbolicFuncDatasetNodeData;
  const payload: Record<string, unknown> = {
    equationLatex: firstScalar(d.equationLatex, defs.equationLatex),
    inputDim: firstScalar(d.inputDim, defs.inputDim as number),
    outputDim: firstScalar(d.outputDim, defs.outputDim as number),
    inputDistribution: firstScalar(d.inputDistribution, defs.inputDistribution as string),
    outputDistribution: firstScalar(d.outputDistribution, defs.outputDistribution as string),
    trainSize: firstScalar(d.trainSize, defs.trainSize as number),
    testSize: firstScalar(d.testSize, defs.testSize as number),
    noiseLevel: firstScalar(d.noiseLevel, defs.noiseLevel as number),
    seed: firstScalar(d.seed, defs.seed as number),
  };
  const pyDict = toPythonDictLiteral(payload);
  return `# === ${title} (symbolic_func_dataset) ===
# y(x) from LaTeX via the same compiler as \`/api/train\`.
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from comfy_research.engine.linear_dataset_sampling import sample_inputs
from comfy_research.engine.symbolic_dataset_compile import build_y_numpy_fn

_DATA = ${pyDict}

def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    rng = np.random.default_rng(int(_DATA.get("seed", 0)))
    d_in = int(_DATA["inputDim"])
    n_tr = int(_DATA["trainSize"])
    n_te = int(_DATA["testSize"])
    in_dist = str(_DATA.get("inputDistribution", "standard_normal"))
    out_dist = str(_DATA.get("outputDistribution", "deterministic"))
    sigma = float(_DATA.get("noiseLevel", 0.0) or 0.0)
    additive = out_dist == "additive_gaussian"
    y_fn = build_y_numpy_fn(_DATA)
    x_train = sample_inputs(rng, n_tr, d_in, in_dist)
    y_train = y_fn(x_train)
    if additive and sigma > 0:
        y_train = y_train + sigma * rng.standard_normal(y_train.shape).astype(np.float32)
    if n_te > 0:
        x_test = sample_inputs(rng, n_te, d_in, in_dist)
        y_test = y_fn(x_test)
        if additive and sigma > 0:
            y_test = y_test + sigma * rng.standard_normal(y_test.shape).astype(np.float32)
    else:
        x_test = y_test = None
    xt = torch.as_tensor(x_train, device=device, dtype=torch.float32)
    yt = torch.as_tensor(y_train, device=device, dtype=torch.float32)
    if x_test is not None and y_test is not None:
        xte = torch.as_tensor(x_test, device=device, dtype=torch.float32)
        yte = torch.as_tensor(y_test, device=device, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(xte, yte), batch_size=batch_size)
    else:
        test_loader = None
    train_loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
`;
}


/** Code-tab loader for the versioned information-bottleneck input table.
 * Keep verification and sampling in Python so exports use the same checked input. */
export function buildInformationBottleneckDatasetLoaders(
  pySym: string,
  title: string,
  raw: Record<string, unknown>,
): string {
  const trainSize = firstScalar(raw.trainSize, 3482);
  const testSize = firstScalar(raw.testSize, 4096);
  const seed = firstScalar(raw.seed, 0);
  return `# === ${title} (information_bottleneck_dataset) ===
# Uses ComfyResearch's bundled, SHA-256-verified F/y input table.
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from comfy_research.engine.information_bottleneck_dataset import build_information_bottleneck_arrays


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    x_train, y_train, x_test, y_test = build_information_bottleneck_arrays(
        np.random.default_rng(${JSON.stringify(seed)}),
        train_size=${JSON.stringify(trainSize)},
        test_size=${JSON.stringify(testSize)},
    )
    train_ds = TensorDataset(torch.from_numpy(x_train).to(device), torch.from_numpy(y_train).to(device))
    test_ds = (
        TensorDataset(torch.from_numpy(x_test).to(device), torch.from_numpy(y_test).to(device))
        if x_test is not None and y_test is not None
        else None
    )
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), (DataLoader(test_ds, batch_size=batch_size) if test_ds else None)
`;
}

export function buildRandomInputDistributionLoaders(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultRandomInputDistributionData();
  const d = { ...defs, ...(raw as Partial<RandomInputDistributionNodeData>) } as RandomInputDistributionNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [];
  const fn = safeSpecFn(pySym);
  const body = generateRandomInputDistributionSpecCode(d, order, fn);
  const trainN = 800;
  const testN = 200;
  return `# === ${title} (random_input_distribution) ===
# x-only sampling (labels are zeros). Default train/test counts for notebook use.
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from comfy_research.engine.random_input_distribution_runtime import (
    rng_from_random_input_distribution_data,
    sample_x_from_random_input_dict,
)

${body}

def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    dd = {
        "inputDim": int(${JSON.stringify(firstScalar(d.inputDim, defs.inputDim as number))}),
        "inputDistribution": ${JSON.stringify(String(firstScalar(d.inputDistribution, defs.inputDistribution as string)))},
        "noiseDistribution": ${JSON.stringify(String(firstScalar(d.noiseDistribution, defs.noiseDistribution as string)))},
        "noiseLevel": float(${JSON.stringify(firstScalar(d.noiseLevel, defs.noiseLevel as number))}),
        "seed": int(${JSON.stringify(firstScalar(d.seed, defs.seed as number))}),
    }
    rng = rng_from_random_input_distribution_data(dd)
    x_train = sample_x_from_random_input_dict(dd, int(${trainN}), rng)
    x_test = sample_x_from_random_input_dict(dd, int(${testN}), rng)
    y_train = torch.zeros((int(x_train.shape[0]), 1), device=device, dtype=torch.float32)
    y_test = torch.zeros((int(x_test.shape[0]), 1), device=device, dtype=torch.float32)
    xt = torch.as_tensor(x_train, device=device, dtype=torch.float32)
    xte = torch.as_tensor(x_test, device=device, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(xt, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(xte, y_test), batch_size=batch_size)
    return train_loader, test_loader
`;
}

export function buildInputSamplerLoaders(
  pySym: string,
  title: string,
  raw: Record<string, unknown>,
  ctx: CodegenContext | undefined,
  nodeId: string,
): string {
  const defs = defaultInputSamplerData();
  const d = { ...defs, ...(raw as Partial<InputSamplerNodeData>) } as InputSamplerNodeData;
  const numSamples = Math.max(1, Math.floor(firstScalar(d.numSamples, defs.numSamples as number)));
  const ridNode = ctx ? upstreamSource(ctx, nodeId, "distribution") : undefined;
  const ridRaw = (ridNode?.data ?? {}) as Record<string, unknown>;
  const rid = { ...defaultRandomInputDistributionData(), ...ridRaw } as RandomInputDistributionNodeData;
  const ridOrder = rid.paramOrder?.length ? rid.paramOrder : [];
  const ridFn = `Cr_rid_${pySym.replace(/[^a-zA-Z0-9_]+/g, "_")}`;
  const ridBody = generateRandomInputDistributionSpecCode(rid, ridOrder, ridFn);
  return `# === ${title} (input_sampler) ===
# \`sample_x_from_sampler_dict\` with upstream random-input dict (wire **distribution** on the canvas).
import torch
from torch.utils.data import DataLoader, TensorDataset
from comfy_research.engine.random_input_distribution_runtime import (
    rng_from_random_input_distribution_data,
    sample_x_from_sampler_dict,
)

${ridBody}

def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    dd = {
        "inputDim": int(${JSON.stringify(firstScalar(rid.inputDim, 10))}),
        "inputDistribution": ${JSON.stringify(String(firstScalar(rid.inputDistribution, "standard_normal")))},
        "noiseDistribution": ${JSON.stringify(String(firstScalar(rid.noiseDistribution, "deterministic")))},
        "noiseLevel": float(${JSON.stringify(firstScalar(rid.noiseLevel, 0))}),
        "seed": int(${JSON.stringify(firstScalar(rid.seed, 0))}),
    }
    rng = rng_from_random_input_distribution_data(dd)
    sampler = {"numSamples": int(${JSON.stringify(numSamples)})}
    x = sample_x_from_sampler_dict(sampler, dd)
    xt = torch.as_tensor(x, device=device, dtype=torch.float32)
    y = torch.zeros((xt.shape[0], 1), device=device, dtype=torch.float32)
    return DataLoader(TensorDataset(xt, y), batch_size=batch_size, shuffle=True), None
`;
}

export function buildDatasetMixerLoaders(
  pySym: string,
  title: string,
  raw: Record<string, unknown>,
  ctx: CodegenContext | undefined,
  nodeId: string,
): string {
  const defs = defaultDatasetMixerData();
  const d = { ...defs, ...(raw as Partial<DatasetMixerNodeData>) } as DatasetMixerNodeData;
  const pA = Number(firstScalar(d.proportionA, defs.proportionA as number));
  const initSeed = Math.floor(firstScalar(d.initSeed, 0));
  const trainTotal = Math.floor(firstScalar(d.trainTotalSamples, defs.trainTotalSamples as number));
  const testTotal = Math.floor(firstScalar(d.testTotalSamples, defs.testTotalSamples as number));
  if (!ctx) return buildDatasetMixerUnwired(pySym, title, trainTotal);
  const a = upstreamSource(ctx, nodeId, "dataset_a");
  const b = upstreamSource(ctx, nodeId, "dataset_b");
  if (!a || !b) return buildDatasetMixerUnwired(pySym, title, trainTotal);
  const symA = pySlugForNode(a.id, ctx.nodes);
  const symB = pySlugForNode(b.id, ctx.nodes);
  return `# === ${title} (dataset_mixer) ===
# Concatenate n_a=round(train_total*proportion_a) synthetic draws from each branch loader, then shuffle (see trainer_run).
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    la, _ta = fn_${symA}_loaders(batch_size=max(1, int(batch_size)), device=device)
    lb, _tb = fn_${symB}_loaders(batch_size=max(1, int(batch_size)), device=device)
    xaf, yaf = la.dataset.tensors
    xbf, ybf = lb.dataset.tensors
    rng = np.random.default_rng(int(${JSON.stringify(initSeed)}))
    n_train = int(${JSON.stringify(trainTotal)})
    n_test = int(${JSON.stringify(testTotal)})
    p_a = float(${JSON.stringify(pA)})

    def split_counts(n_total: int) -> tuple[int, int]:
        n_a = int(round(n_total * p_a))
        n_a = max(0, min(n_total, n_a))
        return n_a, n_total - n_a

    n_a_tr, n_b_tr = split_counts(n_train)
    n_a_te, n_b_te = split_counts(n_test)

    def take_rows(x: torch.Tensor, y: torch.Tensor, n_want: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_want = int(max(0, n_want))
        if n_want <= 0 or int(x.shape[0]) == 0:
            zx = torch.zeros((0,) + tuple(x.shape[1:]), device=device, dtype=x.dtype)
            zy = torch.zeros((0,) + tuple(y.shape[1:]), device=device, dtype=y.dtype)
            return zx, zy
        idx = torch.as_tensor(rng.integers(0, int(x.shape[0]), size=(n_want,)), device=device, dtype=torch.long)
        return x.index_select(0, idx), y.index_select(0, idx)

    xa_tr, ya_tr = take_rows(xaf, yaf, n_a_tr)
    xb_tr, yb_tr = take_rows(xbf, ybf, n_b_tr)
    x_train = torch.cat([xa_tr, xb_tr], dim=0)
    y_train = torch.cat([ya_tr, yb_tr], dim=0)
    if int(x_train.shape[0]) > 1:
        perm = torch.as_tensor(rng.permutation(int(x_train.shape[0])), device=device, dtype=torch.long)
        x_train = x_train.index_select(0, perm)
        y_train = y_train.index_select(0, perm)
    if n_test > 0:
        xa_te, ya_te = take_rows(xaf, yaf, n_a_te)
        xb_te, yb_te = take_rows(xbf, ybf, n_b_te)
        x_test = torch.cat([xa_te, xb_te], dim=0)
        y_test = torch.cat([ya_te, yb_te], dim=0)
        if int(x_test.shape[0]) > 1:
            perm2 = torch.as_tensor(rng.permutation(int(x_test.shape[0])), device=device, dtype=torch.long)
            x_test = x_test.index_select(0, perm2)
            y_test = y_test.index_select(0, perm2)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)
    else:
        test_loader = None
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
`;
}

function buildDatasetMixerUnwired(pySym: string, title: string, trainTotal: number): string {
  return `# === ${title} (dataset_mixer) ===
# Wire **dataset_a** and **dataset_b**, then refresh Code so this cell can call both upstream \`fn_*_loaders\`.
import torch
from torch.utils.data import DataLoader, TensorDataset

def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    n = int(max(1, ${JSON.stringify(trainTotal)}))
    x = torch.zeros(n, 1, device=device, dtype=torch.float32)
    y = torch.zeros(n, 1, device=device, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True), None
`;
}

export function buildDatasetMixerBPlaceholder(pySym: string, title: string): string {
  return `# === ${title} (dataset_mixer_b) ===
# Synchronized interpolation between branches is implemented in \`comfy_research/engine/trainer_run.py\`.
# This placeholder keeps batch shapes runnable; use Train on the canvas for engine-faithful mixing.
import torch
from torch.utils.data import DataLoader, TensorDataset

def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    x = torch.randn(64, 8, device=device)
    y = torch.randn(64, 1, device=device)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True), None
`;
}

export { TOY_LANGUAGE_DATASET_KINDS };


// 自 nodeRegistrySpec.ts 移入(打破 generatedNodeSpecTypes ↔ nodeRegistrySpec
// 的循环 import TDZ;crl 系 spec 条目与 CODEGEN_ADAPTERS 共用)。
export function crlServerSideStub(pySym: string, title: string, nodeType: string): string {
  return `# === ${title} (${nodeType}) ===
# This node runs on the ComfyResearch server (see comfy_research/engine/crl_run.py).
# Use the canvas Train button; there is no standalone local export for Code mode yet.


def fn_${pySym}_info():
    print("CRL / env wiring is executed via POST /api/train — not in this notebook cell.")
`;
}
