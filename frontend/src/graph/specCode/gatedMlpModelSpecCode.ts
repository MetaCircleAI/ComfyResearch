import type { MlpActivationId } from "../../components/nodes/mlpModelDefaults";
import type { GatedMlpModelNodeData } from "../../components/nodes/gatedMlpModelDefaults";
import { defaultGatedMlpModelData } from "../../components/nodes/gatedMlpModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set(["inputDim", "outputDim", "depth", "width", "activation", "seed"]);

export const DEFAULT_GATED_MLP_SPEC_NAME = "GatedMlpModel";

export const DEFAULT_GATED_MLP_PARAM_ORDER: (keyof GatedMlpModelNodeData)[] = [
  "inputDim",
  "outputDim",
  "depth",
  "width",
  "activation",
  "seed",
];

function pyTypeForKey(key: keyof GatedMlpModelNodeData): string {
  if (key === "activation") return "str";
  return "int";
}

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(key: keyof GatedMlpModelNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (key === "activation") return JSON.stringify(String(s));
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

export function generateGatedMlpModelSpecCode(d: GatedMlpModelNodeData, order: string[], specName: string): string {
  const name = specName.trim() || DEFAULT_GATED_MLP_SPEC_NAME;
  const lines: string[] = [`import torch`, ``, `class ${name}(torch.nn.Module):`, `    def __init__(`, `        self,`];
  const merged = { ...defaultGatedMlpModelData(), ...d };
  const keys = (order.length ? order : DEFAULT_GATED_MLP_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const k of keys) {
    const ck = k as keyof GatedMlpModelNodeData;
    const sn = camelToSnakeCase(String(ck));
    lines.push(`        ${sn}: ${pyTypeForKey(ck)} = ${formatPyDefault(ck, merged[ck])},`);
  }
  lines.push(`    ):`);
  lines.push(`        super().__init__()`);
  lines.push(`        acts = {`);
  lines.push(`            "relu": torch.nn.ReLU,`);
  lines.push(`            "gelu": torch.nn.GELU,`);
  lines.push(`            "tanh": torch.nn.Tanh,`);
  lines.push(`            "sigmoid": torch.nn.Sigmoid,`);
  lines.push(`            "leaky_relu": torch.nn.LeakyReLU,`);
  lines.push(`            "silu": torch.nn.SiLU,`);
  lines.push(`            "identity": torch.nn.Identity,`);
  lines.push(`        }`);
  lines.push(`        self.act = acts.get(str(activation), torch.nn.SiLU)()`);
  lines.push(`        self.gates = torch.nn.ModuleList()`);
  lines.push(`        self.values = torch.nn.ModuleList()`);
  lines.push(`        in_f = int(input_dim)`);
  lines.push(`        for _ in range(int(depth)):`); 
  lines.push(`            self.gates.append(torch.nn.Linear(in_f, int(width)))`);
  lines.push(`            self.values.append(torch.nn.Linear(in_f, int(width)))`);
  lines.push(`            in_f = int(width)`);
  lines.push(`        self.out = torch.nn.Linear(in_f, int(output_dim))`);
  lines.push(``);
  lines.push(`    def forward(self, x: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        h = x`);
  lines.push(`        for gate, value in zip(self.gates, self.values):`);
  lines.push(`            h = self.act(gate(h)) * value(h)`);
  lines.push(`        return self.out(h)`);
  return lines.join("\n");
}

function parseGatedMlpClassInitHeader(code: string): { className: string; paramsChunk: string; error?: string } {
  const cm = code.match(/^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*torch\.nn\.Module\s*\)\s*:/m);
  if (!cm) return { className: "", paramsChunk: "", error: "Expected class Name(torch.nn.Module):" };
  const className = cm[1]!;
  const after = code.slice(code.indexOf(cm[0]) + cm[0].length);
  const im = after.match(/def\s+__init__\s*\(\s*self\s*,([\s\S]*?)\)\s*:/m);
  if (!im) return { className, paramsChunk: "", error: "Expected __init__(self, ...) in class." };
  return { className, paramsChunk: im[1] ?? "" };
}

export function parseGatedMlpModelSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<GatedMlpModelNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  let specName = "";
  let params: Array<{ snakeName: string; rawValue: string }> = [];
  const cls = parseGatedMlpClassInitHeader(code);
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
  const patch: Partial<GatedMlpModelNodeData> = {};
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
    const key = camel as keyof GatedMlpModelNodeData;
    if (key === "activation") {
      patch[key] = String(val) as MlpActivationId;
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
