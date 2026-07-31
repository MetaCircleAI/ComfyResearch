import type { ResidualLnModelNodeData, ResidualLnMode } from "../../components/nodes/residualLnModelDefaults";
import { defaultResidualLnModelData } from "../../components/nodes/residualLnModelDefaults";
import type { MlpActivationId } from "../../components/nodes/mlpModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set(["dim", "depth", "alpha", "lnMode", "activation", "seed"]);

export const DEFAULT_RESIDUAL_LN_SPEC_NAME = "ResidualLnModel";

export const DEFAULT_RESIDUAL_LN_PARAM_ORDER: (keyof ResidualLnModelNodeData)[] = [
  "dim",
  "depth",
  "alpha",
  "lnMode",
  "activation",
  "seed",
];

const LN_MODES = new Set(["pre_ln", "post_ln"]);

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(key: keyof ResidualLnModelNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (key === "lnMode" || key === "activation") return JSON.stringify(String(s));
  if (typeof s === "number" && !Number.isInteger(s)) return String(s);
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

function pyTypeForKey(key: keyof ResidualLnModelNodeData): string {
  if (key === "lnMode" || key === "activation") return "str";
  if (key === "alpha") return "float";
  return "int";
}

function parseClassInitHeader(code: string): { className: string; paramsChunk: string; error?: string } {
  const cm = code.match(/^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*torch\.nn\.Module\s*\)\s*:/m);
  if (!cm) return { className: "", paramsChunk: "", error: "Expected class Name(torch.nn.Module):" };
  const className = cm[1]!;
  const after = code.slice(code.indexOf(cm[0]) + cm[0].length);
  const im = after.match(/def\s+__init__\s*\(\s*self\s*,([\s\S]*?)\)\s*:/m);
  if (!im) return { className, paramsChunk: "", error: "Expected __init__(self, ...) in class." };
  return { className, paramsChunk: im[1] ?? "" };
}

export function generateResidualLnModelSpecCode(
  d: ResidualLnModelNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_RESIDUAL_LN_SPEC_NAME;
  const lines: string[] = [`import torch`, ``, `class ${name}(torch.nn.Module):`, `    def __init__(`, `        self,`];
  const merged = { ...defaultResidualLnModelData(), ...d };
  const keys = (order.length ? order : DEFAULT_RESIDUAL_LN_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const k of keys) {
    const ck = k as keyof ResidualLnModelNodeData;
    const sn = camelToSnakeCase(String(ck));
    lines.push(`        ${sn}: ${pyTypeForKey(ck)} = ${formatPyDefault(ck, merged[ck])},`);
  }
  lines.push(`    ):`);
  lines.push(`        super().__init__()`);
  lines.push(`        # Pre/Post-LN residual stack with alpha-scaled FC2 (see ComfyResearch residual LN spec).`);
  lines.push(``);
  lines.push(`    def forward(self, x: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        return x  # stub: training uses the compiled graph`);
  return lines.join("\n");
}

export function parseResidualLnModelSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<ResidualLnModelNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  let specName = "";
  let params: Array<{ snakeName: string; rawValue: string }> = [];
  const cls = parseClassInitHeader(code);
  if (!cls.error) {
    specName = cls.className;
    const rows = cls.paramsChunk
      .split("\n")
      .map((l) => l.replace(/#.*$/, "").trim())
      .filter((l) => l.length > 0);
    for (const line of rows) {
      const noComma = line.endsWith(",") ? line.slice(0, -1).trim() : line;
      const pm = noComma.match(/^(\w+)\s*:\s*[^=]+=\s*(.+)$/);
      if (!pm) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Bad parameter line: ${noComma}` };
      }
      params.push({ snakeName: pm[1]!, rawValue: pm[2]! });
    }
  } else {
    const p = parsePythonFunctionSpecHeader(code);
    if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: cls.error };
    specName = p.funcName;
    params = p.params.map((x) => ({ snakeName: x.snakeName, rawValue: x.rawValue }));
  }
  if (params.length === 0) {
    return { specName, paramOrder: [], patch: {}, extras: {}, error: "No parameters found." };
  }
  const patch: Partial<ResidualLnModelNodeData> = {};
  const extras: Record<string, string | number | boolean> = {};
  const paramOrder: string[] = [];
  for (const row of params) {
    const camel = snakeToCamelCase(row.snakeName);
    const val = parsePythonDefault(row.rawValue);
    if (!KNOWN_KEYS.has(camel)) {
      if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") {
        extras[camel] = val;
      }
      continue;
    }
    paramOrder.push(camel);
    const key = camel as keyof ResidualLnModelNodeData;
    if (key === "lnMode") {
      const s = String(val);
      if (!LN_MODES.has(s)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `ln_mode must be pre_ln or post_ln, got ${s}` };
      }
      patch[key] = s as ResidualLnMode;
    } else if (key === "activation") {
      patch[key] = String(val) as MlpActivationId;
    } else if (key === "alpha") {
      const n = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(n)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid float for ${camel}` };
      }
      patch[key] = n as never;
    } else {
      const n = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(n) || !Number.isInteger(n)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
      }
      patch[key] = n as never;
    }
  }
  return { specName, paramOrder, patch, extras };
}
