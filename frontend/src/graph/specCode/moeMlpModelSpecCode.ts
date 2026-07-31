import type { MlpActivationId } from "../../components/nodes/mlpModelDefaults";
import type { MoeMlpModelNodeData } from "../../components/nodes/moeMlpModelDefaults";
import { defaultMoeMlpModelData } from "../../components/nodes/moeMlpModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set(["inputDim", "outputDim", "depth", "width", "numExperts", "activation", "seed"]);

export const DEFAULT_MOE_MLP_SPEC_NAME = "MoeMlpModel";

export const DEFAULT_MOE_MLP_PARAM_ORDER: (keyof MoeMlpModelNodeData)[] = [
  "inputDim",
  "outputDim",
  "depth",
  "width",
  "numExperts",
  "activation",
  "seed",
];

function pyTypeForKey(key: keyof MoeMlpModelNodeData): string {
  if (key === "activation") return "str";
  return "int";
}

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(key: keyof MoeMlpModelNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (key === "activation") return JSON.stringify(String(s));
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

export function generateMoeMlpModelSpecCode(d: MoeMlpModelNodeData, order: string[], specName: string): string {
  const name = specName.trim() || DEFAULT_MOE_MLP_SPEC_NAME;
  const lines: string[] = [`import torch`, ``, `class ${name}(torch.nn.Module):`, `    def __init__(`, `        self,`];
  const merged = { ...defaultMoeMlpModelData(), ...d };
  const keys = (order.length ? order : DEFAULT_MOE_MLP_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const k of keys) {
    const ck = k as keyof MoeMlpModelNodeData;
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
  lines.push(`        act_cls = acts.get(str(activation), torch.nn.SiLU)`);
  lines.push(`        self.gate = torch.nn.Linear(int(input_dim), int(num_experts))`);
  lines.push(`        self.experts = torch.nn.ModuleList()`);
  lines.push(`        for _ in range(int(num_experts)):`); 
  lines.push(`            layers = []`);
  lines.push(`            in_f = int(input_dim)`);
  lines.push(`            for _ in range(int(depth)):`); 
  lines.push(`                layers.append(torch.nn.Linear(in_f, int(width)))`);
  lines.push(`                layers.append(act_cls())`);
  lines.push(`                in_f = int(width)`);
  lines.push(`            layers.append(torch.nn.Linear(in_f, int(output_dim)))`);
  lines.push(`            self.experts.append(torch.nn.Sequential(*layers))`);
  lines.push(``);
  lines.push(`    def forward(self, x: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        g = torch.softmax(self.gate(x), dim=-1)`);
  lines.push(`        ys = [expert(x) for expert in self.experts]`);
  lines.push(`        stacked = torch.stack(ys, dim=1)`);
  lines.push(`        return (stacked * g.unsqueeze(-1)).sum(dim=1)`);
  return lines.join("\n");
}

function parseMoeMlpClassInitHeader(code: string): { className: string; paramsChunk: string; error?: string } {
  const cm = code.match(/^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*torch\.nn\.Module\s*\)\s*:/m);
  if (!cm) return { className: "", paramsChunk: "", error: "Expected class Name(torch.nn.Module):" };
  const className = cm[1]!;
  const after = code.slice(code.indexOf(cm[0]) + cm[0].length);
  const im = after.match(/def\s+__init__\s*\(\s*self\s*,([\s\S]*?)\)\s*:/m);
  if (!im) return { className, paramsChunk: "", error: "Expected __init__(self, ...) in class." };
  return { className, paramsChunk: im[1] ?? "" };
}

export function parseMoeMlpModelSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<MoeMlpModelNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  let specName = "";
  let params: Array<{ snakeName: string; rawValue: string }> = [];
  const cls = parseMoeMlpClassInitHeader(code);
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
  const patch: Partial<MoeMlpModelNodeData> = {};
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
    const key = camel as keyof MoeMlpModelNodeData;
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
