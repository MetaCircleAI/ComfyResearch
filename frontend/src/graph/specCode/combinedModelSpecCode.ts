import type { CombinedModelNodeData } from "../../components/nodes/combinedModelDefaults";
import { defaultCombinedModelData } from "../../components/nodes/combinedModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set(["displayName"]);

export const DEFAULT_COMBINED_MODEL_SPEC_NAME = "CombinedModel";

export const DEFAULT_COMBINED_MODEL_PARAM_ORDER: (keyof CombinedModelNodeData)[] = ["displayName"];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

export function generateCombinedModelSpecCode(
  d: CombinedModelNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_COMBINED_MODEL_SPEC_NAME;
  const merged = { ...defaultCombinedModelData(), ...d };
  const lines: string[] = [`import torch`, ``, `class ${name}(torch.nn.Module):`, `    def __init__(`, `        self,`];
  const keys = (order.length ? order : DEFAULT_COMBINED_MODEL_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const k of keys) {
    const ck = k as keyof CombinedModelNodeData;
    const sn = camelToSnakeCase(String(ck));
    const s = firstScalar(merged[ck]);
    lines.push(`        ${sn}: str = ${JSON.stringify(String(s))},`);
  }
  lines.push(`    ):`);
  lines.push(`        super().__init__()`);
  lines.push(``);
  lines.push(`    def forward(self, x: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        return x`);
  return lines.join("\n");
}

function parseCombinedClassInitHeader(code: string): { className: string; paramsChunk: string; error?: string } {
  const cm = code.match(/^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*torch\.nn\.Module\s*\)\s*:/m);
  if (!cm) return { className: "", paramsChunk: "", error: "Expected class Name(torch.nn.Module):" };
  const className = cm[1]!;
  const after = code.slice(code.indexOf(cm[0]) + cm[0].length);
  const im = after.match(/def\s+__init__\s*\(\s*self\s*,([\s\S]*?)\)\s*:/m);
  if (!im) return { className, paramsChunk: "", error: "Expected __init__(self, ...) in class." };
  return { className, paramsChunk: im[1] ?? "" };
}

export function parseCombinedModelSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<CombinedModelNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  let specName = "";
  let params: Array<{ snakeName: string; rawValue: string }> = [];
  const cls = parseCombinedClassInitHeader(code);
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
  const patch: Partial<CombinedModelNodeData> = {};
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
    patch.displayName = String(val);
  }
  return { specName, paramOrder, patch, extras };
}
