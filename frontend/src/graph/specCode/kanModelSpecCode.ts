import type { KanBaseFunId, KanModelNodeData } from "../../components/nodes/kanModelDefaults";
import { defaultKanModelData } from "../../components/nodes/kanModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set(["inputDim", "outputDim", "depth", "width", "grid", "k", "baseFun", "seed"]);

export const DEFAULT_KAN_SPEC_NAME = "KanModel";

export const DEFAULT_KAN_PARAM_ORDER: (keyof KanModelNodeData)[] = [
  "inputDim",
  "outputDim",
  "depth",
  "width",
  "grid",
  "k",
  "baseFun",
  "seed",
];

function pyTypeForKey(key: keyof KanModelNodeData): string {
  if (key === "baseFun") return "str";
  return "int";
}

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(key: keyof KanModelNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (key === "baseFun") return JSON.stringify(String(s));
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

export function generateKanModelSpecCode(d: KanModelNodeData, order: string[], specName: string): string {
  const name = specName.trim() || DEFAULT_KAN_SPEC_NAME;
  const lines: string[] = [`import torch`, ``, `class ${name}(torch.nn.Module):`, `    def __init__(`, `        self,`];
  const merged = { ...defaultKanModelData(), ...d };
  const keys = (order.length ? order : DEFAULT_KAN_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const k of keys) {
    const ck = k as keyof KanModelNodeData;
    const sn = camelToSnakeCase(String(ck));
    lines.push(`        ${sn}: ${pyTypeForKey(ck)} = ${formatPyDefault(ck, merged[ck])},`);
  }
  lines.push(`    ):`);
  lines.push(`        super().__init__()`);
  lines.push(`        from kan import KAN`);
  lines.push(
    `        w = [int(input_dim)] + [int(width)] * int(depth) + [int(output_dim)]`,
  );
  lines.push(`        self.kan = KAN(`);
  lines.push(`            width=w,`);
  lines.push(`            grid=int(grid),`);
  lines.push(`            k=int(k),`);
  lines.push(`            seed=int(seed),`);
  lines.push(`            base_fun=str(base_fun),`);
  lines.push(`            symbolic_enabled=False,`);
  lines.push(`            auto_save=False,`);
  lines.push(`            save_act=False,`);
  lines.push(`            device="cpu",`);
  lines.push(`        )`);
  lines.push(`        self.kan.speed()`);
  lines.push(``);
  lines.push(`    def forward(self, x: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        return self.kan(x)`);
  return lines.join("\n");
}

function parseKanClassInitHeader(code: string): { className: string; paramsChunk: string; error?: string } {
  const cm = code.match(/^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*torch\.nn\.Module\s*\)\s*:/m);
  if (!cm) return { className: "", paramsChunk: "", error: "Expected class Name(torch.nn.Module):" };
  const className = cm[1]!;
  const after = code.slice(code.indexOf(cm[0]) + cm[0].length);
  const im = after.match(/def\s+__init__\s*\(\s*self\s*,([\s\S]*?)\)\s*:/m);
  if (!im) return { className, paramsChunk: "", error: "Expected __init__(self, ...) in class." };
  return { className, paramsChunk: im[1] ?? "" };
}

export function parseKanModelSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<KanModelNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  let specName = "";
  let params: Array<{ snakeName: string; rawValue: string }> = [];
  const cls = parseKanClassInitHeader(code);
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
  const patch: Partial<KanModelNodeData> = {};
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
    const key = camel as keyof KanModelNodeData;
    if (key === "baseFun") {
      patch[key] = String(val) as KanBaseFunId;
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
