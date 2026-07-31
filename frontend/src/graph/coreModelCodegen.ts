import {
  defaultAttentionOnlyModelData,
  type AttentionOnlyModelNodeData,
} from "../components/nodes/attentionOnlyModelDefaults";
import { defaultMlpModelData, type MlpModelNodeData } from "../components/nodes/mlpModelDefaults";
import { defaultMlpTokenModelData, type MlpTokenModelNodeData } from "../components/nodes/mlpTokenModelDefaults";
import {
  defaultTransformerTokenModelData,
  type TransformerTokenModelNodeData,
} from "../components/nodes/transformerTokenModelDefaults";
import {
  DEFAULT_ATTENTION_ONLY_PARAM_ORDER,
  generateAttentionOnlyModelSpecCode,
} from "./specCode/attentionOnlyModelSpecCode";
import { generateMlpModelSpecCode, DEFAULT_MLP_PARAM_ORDER } from "./specCode/mlpModelSpecCode";
import {
  generateMlpTokenModelVariantSpecCode,
  DEFAULT_MLP_TOKEN_MODEL_PARAM_ORDER,
} from "./specCode/mlpTokenModelSpecCode";
import {
  readmeAttentionOnlyCore,
  readmeMlpToken,
  readmeMlpVector,
  readmeTransformerToken,
} from "./specCode/notebookImplementationReadme";
import {
  DEFAULT_TRANSFORMER_TOKEN_MODEL_PARAM_ORDER,
  generateTransformerTokenModelSpecCode,
} from "./specCode/transformerTokenModelSpecCode";

function firstScalar<T>(v: T | T[] | undefined | null, fallback: T): T {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return (v.length ? v[0]! : fallback) as T;
  return v as T;
}

export function buildMlpTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultMlpModelData();
  const d = { ...defs, ...(raw as Partial<MlpModelNodeData>) } as MlpModelNodeData;
  // outputScale is mlp_model-only; don't let the defaults merge re-add it for
  // node types that never carried it.
  if (!raw || !("outputScale" in raw)) delete d.outputScale;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_MLP_PARAM_ORDER];
  const className = `CrModel_${pySym}`;
  const specBody = generateMlpModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  const inputDim = Math.max(1, Math.floor(Number(firstScalar(d.inputDim, defs.inputDim as number))));
  const outputDim = Math.max(1, Math.floor(Number(firstScalar(d.outputDim, defs.outputDim as number))));
  const depth = Math.max(0, Math.floor(Number(firstScalar(d.depth, defs.depth as number))));
  const readme = readmeMlpVector(inputDim, outputDim, depth);
  return `# === ${title} (mlp_model) ===
${readme}
${specBody}


def fn_${pySym}_model() -> ${className}:
    import torch
    torch.manual_seed(${seed})
    return ${className}()
`;
}

export function buildMlpTokenTorch(
  pySym: string,
  title: string,
  nodeType: "mlp_token_model" | "gated_mlp_token_model" | "moe_mlp_token_model",
  raw: Record<string, unknown>,
): string {
  const defs = defaultMlpTokenModelData();
  const d = { ...defs, ...(raw as Partial<MlpTokenModelNodeData>) } as MlpTokenModelNodeData;
  const variant = nodeType === "gated_mlp_token_model" ? "gated" : nodeType === "moe_mlp_token_model" ? "moe" : "plain";
  const order = (d.paramOrder?.length ? d.paramOrder : [...DEFAULT_MLP_TOKEN_MODEL_PARAM_ORDER]).filter((k) =>
    variant === "moe" ? true : k !== "numExperts",
  );
  const className = `CrModel_${pySym}`;
  const specBody = generateMlpTokenModelVariantSpecCode(d, order, className, variant);
  const seed = firstScalar(d.seed, defs.seed as number);
  const label =
    nodeType === "gated_mlp_token_model"
      ? "Gated MLP token LM"
      : nodeType === "moe_mlp_token_model"
        ? "MoE MLP token LM"
        : "MLP token LM";
  const vocab = Math.max(2, Math.floor(Number(firstScalar(d.vocabSize, defs.vocabSize as number))));
  const embed = Math.max(1, Math.floor(Number(firstScalar(d.embedDim, defs.embedDim as number))));
  const tokensPerInput = Math.max(1, Math.floor(Number(firstScalar(d.tokensPerInput, defs.tokensPerInput as number))));
  const readme = readmeMlpToken(label, vocab, embed, tokensPerInput);
  return `# === ${title} (${nodeType}) ===
${readme}
${specBody}


def fn_${pySym}_model() -> ${className}:
    import torch
    torch.manual_seed(${seed})
    return ${className}()
`;
}

function transformerTokenModelCodegenOrder(d: TransformerTokenModelNodeData): string[] {
  const base = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_TRANSFORMER_TOKEN_MODEL_PARAM_ORDER];
  const leading = (["contextLength", "vocabSize"] as const).filter((k) => !base.includes(k));
  return [...leading, ...base];
}

export function buildTransformerTokenTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultTransformerTokenModelData();
  const d = { ...defs, ...(raw as Partial<TransformerTokenModelNodeData>) } as TransformerTokenModelNodeData;
  const order = transformerTokenModelCodegenOrder(d);
  const className = `CrModel_${pySym}`;
  const specBody = generateTransformerTokenModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  const vocab = Math.max(2, Math.floor(Number(firstScalar(d.vocabSize, defs.vocabSize as number))));
  const dim = Math.max(1, Math.floor(Number(firstScalar(d.modelDim, defs.modelDim as number))));
  const layers = Math.max(1, Math.floor(Number(firstScalar(d.numLayers, defs.numLayers as number))));
  const ctx = Math.max(1, Math.floor(Number(firstScalar(d.contextLength, defs.contextLength as number))));
  const readme = readmeTransformerToken(vocab, dim, layers, ctx);
  return `# === ${title} (transformer_token_model) ===
${readme}
${specBody}


def fn_${pySym}_model() -> ${className}:
    import torch
    torch.manual_seed(${seed})
    return ${className}()
`;
}

export function buildAttentionOnlyTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultAttentionOnlyModelData();
  const d = { ...defs, ...(raw as Partial<AttentionOnlyModelNodeData>) } as AttentionOnlyModelNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_ATTENTION_ONLY_PARAM_ORDER];
  const className = `CrModel_${pySym}`;
  const specBody = generateAttentionOnlyModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  const embedDim = Math.max(1, Math.floor(Number(firstScalar(d.embedDim, defs.embedDim as number))));
  const contextLength = Math.max(1, Math.floor(Number(firstScalar(d.contextLength, defs.contextLength as number))));
  const numHeads = Math.max(1, Math.floor(Number(firstScalar(d.numHeads, defs.numHeads as number))));
  const readme = readmeAttentionOnlyCore(embedDim, contextLength, numHeads);
  return `# === ${title} (attention_only_model) ===
${readme}
${specBody}


def fn_${pySym}_model() -> ${className}:
    import torch
    torch.manual_seed(${seed})
    return ${className}()
`;
}
