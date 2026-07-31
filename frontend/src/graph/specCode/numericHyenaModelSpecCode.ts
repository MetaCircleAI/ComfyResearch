import type { NumericHyenaModelNodeData } from "../../components/nodes/numericHyenaModelDefaults";
import { defaultNumericHyenaModelData } from "../../components/nodes/numericHyenaModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "contextLength",
  "inputDim",
  "outputDim",
  "modelDim",
  "depth",
  "convKernel",
  "ffMult",
  "localMixingKernel",
  "seed",
]);

export const DEFAULT_NUMERIC_HYENA_MODEL_SPEC_NAME = "NumericHyenaModel";
export const DEFAULT_NUMERIC_HYENA_MODEL_PARAM_ORDER: (keyof NumericHyenaModelNodeData)[] = [
  "contextLength",
  "inputDim",
  "outputDim",
  "modelDim",
  "depth",
  "convKernel",
  "ffMult",
  "localMixingKernel",
  "seed",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(v: unknown): string {
  const s = firstScalar(v);
  return typeof s === "number" ? String(s) : JSON.stringify(s);
}

export function generateNumericHyenaModelSpecCode(
  d: NumericHyenaModelNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_NUMERIC_HYENA_MODEL_SPEC_NAME;
  const merged = { ...defaultNumericHyenaModelData(), ...d };
  const keys = (order.length ? order : DEFAULT_NUMERIC_HYENA_MODEL_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  const lines: string[] = [
    `import torch`,
    `from comfy_research.engine.hyena_like_numeric_model import NumericHyenaModel`,
    ``,
    `class ${name}(NumericHyenaModel):`,
    `    def __init__(`,
    `        self,`,
  ];
  for (const key of keys) {
    lines.push(`        ${camelToSnakeCase(key)}: int = ${formatPyDefault(merged[key as keyof NumericHyenaModelNodeData])},`);
  }
  lines.push(`    ):`);
  lines.push(`        super().__init__(`);
  lines.push(`            context_length=int(context_length),`);
  lines.push(`            token_dim=int(input_dim),`);
  lines.push(`            output_token_dim=int(output_dim),`);
  lines.push(`            model_dim=int(model_dim),`);
  lines.push(`            depth=int(depth),`);
  lines.push(`            kernel_size=int(conv_kernel),`);
  lines.push(`            ff_mult=int(ff_mult),`);
  lines.push(`            local_mixing_kernel=int(local_mixing_kernel),`);
  lines.push(`        )`);
  lines.push(`        self.seed = int(seed)`);
  lines.push(``);
  lines.push(`    def forward(self, x: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        return super().forward(x)`);
  return lines.join("\n");
}

export function parseNumericHyenaModelSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<NumericHyenaModelNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<NumericHyenaModelNodeData> = {};
  const extras: Record<string, string | number | boolean> = {};
  const paramOrder: string[] = [];
  for (const row of p.params) {
    const camel = snakeToCamelCase(row.snakeName);
    const val = parsePythonDefault(row.rawValue);
    if (!KNOWN_KEYS.has(camel)) {
      if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") extras[camel] = val;
      continue;
    }
    paramOrder.push(camel);
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isFinite(n) || !Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[camel as keyof NumericHyenaModelNodeData] = n as never;
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
