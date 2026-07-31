import type { RmsNormLayerNodeData } from "../../components/nodes/rmsNormLayerDefaults";
import { defaultRmsNormLayerData } from "../../components/nodes/rmsNormLayerDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set(["normalizedShape", "eps", "elementwiseAffine"]);

export const DEFAULT_RMS_NORM_LAYER_SPEC_NAME = "RMSNormLayer";

export const DEFAULT_RMS_NORM_LAYER_PARAM_ORDER: (keyof RmsNormLayerNodeData)[] = [
  "normalizedShape",
  "eps",
  "elementwiseAffine",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(key: keyof RmsNormLayerNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (key === "elementwiseAffine") {
    const n = typeof s === "number" ? s : Number(s);
    return n !== 0 ? "True" : "False";
  }
  if (key === "eps") {
    if (typeof s === "number") return String(s);
    return JSON.stringify(s);
  }
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

function pyTypeForKey(key: keyof RmsNormLayerNodeData): string {
  if (key === "eps") return "float";
  if (key === "elementwiseAffine") return "bool";
  return "int";
}

export function generateRmsNormLayerSpecCode(d: RmsNormLayerNodeData, order: string[], specName: string): string {
  const name = specName.trim() || DEFAULT_RMS_NORM_LAYER_SPEC_NAME;
  const lines: string[] = [`import torch`, `import torch.nn as nn`, ``, `class ${name}(torch.nn.Module):`, `    def __init__(`, `        self,`];
  const merged = { ...defaultRmsNormLayerData(), ...d };
  const keys = (order.length ? order : DEFAULT_RMS_NORM_LAYER_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const k of keys) {
    const ck = k as keyof RmsNormLayerNodeData;
    const sn = camelToSnakeCase(String(ck));
    lines.push(`        ${sn}: ${pyTypeForKey(ck)} = ${formatPyDefault(ck, merged[ck])},`);
  }
  lines.push(`    ):`);
  lines.push(`        super().__init__()`);
  lines.push(`        self.dim = int(normalized_shape)`);
  lines.push(`        self.eps = float(eps)`);
  lines.push(`        self.weight = nn.Parameter(torch.ones(self.dim)) if elementwise_affine else None`);
  lines.push(``);
  lines.push(`    def forward(self, x: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()`);
  lines.push(`        out = x / rms`);
  lines.push(`        if self.weight is not None:`);
  lines.push(`            out = out * self.weight`);
  lines.push(`        return out`);
  return lines.join("\n");
}

function parseRmsNormClassInitHeader(code: string): { className: string; paramsChunk: string; error?: string } {
  const cm = code.match(/^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*torch\.nn\.Module\s*\)\s*:/m);
  if (!cm) return { className: "", paramsChunk: "", error: "Expected class Name(torch.nn.Module):" };
  const className = cm[1]!;
  const after = code.slice(code.indexOf(cm[0]) + cm[0].length);
  const im = after.match(/def\s+__init__\s*\(\s*self\s*,([\s\S]*?)\)\s*:/m);
  if (!im) return { className, paramsChunk: "", error: "Expected __init__(self, ...) in class." };
  return { className, paramsChunk: im[1] ?? "" };
}

export function parseRmsNormLayerSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<RmsNormLayerNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  let specName = "";
  let params: Array<{ snakeName: string; rawValue: string }> = [];
  const cls = parseRmsNormClassInitHeader(code);
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
  const patch: Partial<RmsNormLayerNodeData> = {};
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
    const key = camel as keyof RmsNormLayerNodeData;
    if (key === "elementwiseAffine") {
      patch[key] = val === true || val === 1 || val === "1" ? 1 : 0;
    } else if (key === "eps") {
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
