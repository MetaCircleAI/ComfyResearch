import { defaultCombinedModelData, type CombinedModelNodeData } from "../components/nodes/combinedModelDefaults";
import { defaultGatedMlpModelData, type GatedMlpModelNodeData } from "../components/nodes/gatedMlpModelDefaults";
import { defaultKanModelData, type KanModelNodeData } from "../components/nodes/kanModelDefaults";
import { defaultMoeMlpModelData, type MoeMlpModelNodeData } from "../components/nodes/moeMlpModelDefaults";
import { defaultMppSpatiotemporalModelData, type MppSpatiotemporalModelNodeData } from "../components/nodes/mppSpatiotemporalModelDefaults";
import {
  defaultAfnoLiteSpatiotemporalModelData,
  type AfnoLiteSpatiotemporalModelNodeData,
} from "../components/nodes/afnoLiteSpatiotemporalModelDefaults";
import { defaultNumericTransformerModelData, type NumericTransformerModelNodeData } from "../components/nodes/numericTransformerModelDefaults";
import { defaultNumericHyenaModelData, type NumericHyenaModelNodeData } from "../components/nodes/numericHyenaModelDefaults";
import { defaultResidualLnModelData, type ResidualLnModelNodeData } from "../components/nodes/residualLnModelDefaults";
import { defaultTransformerMultiTokenModelData, type TransformerMultiTokenModelNodeData } from "../components/nodes/transformerMultiTokenModelDefaults";
import { generateCombinedModelSpecCode } from "./specCode/combinedModelSpecCode";
import { generateGatedMlpModelSpecCode, DEFAULT_GATED_MLP_PARAM_ORDER } from "./specCode/gatedMlpModelSpecCode";
import { generateKanModelSpecCode, DEFAULT_KAN_PARAM_ORDER } from "./specCode/kanModelSpecCode";
import { generateMoeMlpModelSpecCode, DEFAULT_MOE_MLP_PARAM_ORDER } from "./specCode/moeMlpModelSpecCode";
import { generateMppSpatiotemporalModelSpecCode, DEFAULT_MPP_SPATIOTEMPORAL_PARAM_ORDER } from "./specCode/mppSpatiotemporalModelSpecCode";
import {
  generateAfnoLiteSpatiotemporalModelSpecCode,
  DEFAULT_AFNO_LITE_SPATIOTEMPORAL_PARAM_ORDER,
} from "./specCode/afnoLiteSpatiotemporalModelSpecCode";
import {
  generateNumericTransformerModelSpecCode,
  DEFAULT_NUMERIC_TRANSFORMER_MODEL_PARAM_ORDER,
} from "./specCode/numericTransformerModelSpecCode";
import {
  generateNumericHyenaModelSpecCode,
  DEFAULT_NUMERIC_HYENA_MODEL_PARAM_ORDER,
} from "./specCode/numericHyenaModelSpecCode";
import { generateResidualLnModelSpecCode, DEFAULT_RESIDUAL_LN_PARAM_ORDER } from "./specCode/residualLnModelSpecCode";
import {
  generateTransformerMultiTokenModelSpecCode,
  DEFAULT_TRANSFORMER_MULTI_TOKEN_MODEL_PARAM_ORDER,
} from "./specCode/transformerMultiTokenModelSpecCode";

function firstScalar<T>(v: T | T[] | undefined | null, fallback: T): T {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return (v.length ? v[0]! : fallback) as T;
  return v as T;
}

function wrapModelSpec(title: string, nodeType: string, pySym: string, className: string, specBody: string, seed: number): string {
  return `# === ${title} (${nodeType}) ===
${specBody}


def fn_${pySym}_model() -> ${className}:
    import torch
    torch.manual_seed(int(${JSON.stringify(seed)}))
    return ${className}()
`;
}

export function buildGatedMlpModelTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultGatedMlpModelData();
  const d = { ...defs, ...(raw as Partial<GatedMlpModelNodeData>) } as GatedMlpModelNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_GATED_MLP_PARAM_ORDER];
  const className = `CrModel_${pySym}`;
  const body = generateGatedMlpModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapModelSpec(title, "gated_mlp_model", pySym, className, body, seed);
}

export function buildMoeMlpModelTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultMoeMlpModelData();
  const d = { ...defs, ...(raw as Partial<MoeMlpModelNodeData>) } as MoeMlpModelNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_MOE_MLP_PARAM_ORDER];
  const className = `CrModel_${pySym}`;
  const body = generateMoeMlpModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapModelSpec(title, "moe_mlp_model", pySym, className, body, seed);
}

export function buildKanModelTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultKanModelData();
  const d = { ...defs, ...(raw as Partial<KanModelNodeData>) } as KanModelNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_KAN_PARAM_ORDER];
  const className = `CrModel_${pySym}`;
  const body = generateKanModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapModelSpec(title, "kan_model", pySym, className, body, seed);
}

export function buildCombinedModelTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultCombinedModelData();
  const d = { ...defs, ...(raw as Partial<CombinedModelNodeData>) } as CombinedModelNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : ["displayName"];
  const className = `CrModel_${pySym}`;
  const body = generateCombinedModelSpecCode(d, order, className);
  return wrapModelSpec(title, "combined_model", pySym, className, body, 0);
}

export function buildResidualLnModelTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultResidualLnModelData();
  const d = { ...defs, ...(raw as Partial<ResidualLnModelNodeData>) } as ResidualLnModelNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_RESIDUAL_LN_PARAM_ORDER];
  const className = `CrModel_${pySym}`;
  const body = generateResidualLnModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapModelSpec(title, "residual_ln_model", pySym, className, body, seed);
}

export function buildNumericTransformerModelTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultNumericTransformerModelData();
  const d = { ...defs, ...(raw as Partial<NumericTransformerModelNodeData>) } as NumericTransformerModelNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_NUMERIC_TRANSFORMER_MODEL_PARAM_ORDER];
  const className = `CrModel_${pySym}`;
  const body = generateNumericTransformerModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapModelSpec(title, "numeric_transformer_model", pySym, className, body, seed);
}

export function buildNumericHyenaModelTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultNumericHyenaModelData();
  const d = { ...defs, ...(raw as Partial<NumericHyenaModelNodeData>) } as NumericHyenaModelNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_NUMERIC_HYENA_MODEL_PARAM_ORDER];
  const className = `CrModel_${pySym}`;
  const body = generateNumericHyenaModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapModelSpec(title, "numeric_hyena_model", pySym, className, body, seed);
}

export function buildMppSpatiotemporalModelTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultMppSpatiotemporalModelData();
  const d = { ...defs, ...(raw as Partial<MppSpatiotemporalModelNodeData>) } as MppSpatiotemporalModelNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_MPP_SPATIOTEMPORAL_PARAM_ORDER];
  const className = `CrModel_${pySym}`;
  const body = generateMppSpatiotemporalModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapModelSpec(title, "mpp_spatiotemporal_model", pySym, className, body, seed);
}

export function buildAfnoLiteSpatiotemporalModelTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultAfnoLiteSpatiotemporalModelData();
  const d = { ...defs, ...(raw as Partial<AfnoLiteSpatiotemporalModelNodeData>) } as AfnoLiteSpatiotemporalModelNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_AFNO_LITE_SPATIOTEMPORAL_PARAM_ORDER];
  const className = `CrModel_${pySym}`;
  const body = generateAfnoLiteSpatiotemporalModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapModelSpec(title, "afno_lite_spatiotemporal_model", pySym, className, body, seed);
}

export function buildTransformerMultiTokenModelTorch(pySym: string, title: string, raw: Record<string, unknown>): string {
  const defs = defaultTransformerMultiTokenModelData();
  const d = { ...defs, ...(raw as Partial<TransformerMultiTokenModelNodeData>) } as TransformerMultiTokenModelNodeData;
  const order = d.paramOrder?.length ? d.paramOrder : [...DEFAULT_TRANSFORMER_MULTI_TOKEN_MODEL_PARAM_ORDER];
  const className = `CrModel_${pySym}`;
  const body = generateTransformerMultiTokenModelSpecCode(d, order, className);
  const seed = firstScalar(d.seed, defs.seed as number);
  return wrapModelSpec(title, "transformer_multi_token_model", pySym, className, body, seed);
}
