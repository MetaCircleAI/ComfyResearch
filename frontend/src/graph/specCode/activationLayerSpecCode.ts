import type { ActivationLayerNodeData } from "../../components/nodes/activationLayerDefaults";
import { defaultActivationLayerData } from "../../components/nodes/activationLayerDefaults";
import type { MlpActivationId } from "../../components/nodes/mlpModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set(["activation", "leakyP"]);

export const DEFAULT_ACTIVATION_LAYER_SPEC_NAME = "ActivationLayer";

export const DEFAULT_ACTIVATION_LAYER_PARAM_ORDER: (keyof ActivationLayerNodeData)[] = ["activation", "leakyP"];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function clampLeakyP(n: number): number {
  if (!Number.isFinite(n)) return 0;
  return Math.max(-1, Math.min(1, n));
}

function formatActivationLayerPyDefault(key: keyof ActivationLayerNodeData, merged: ActivationLayerNodeData): string {
  const s = firstScalar(merged[key]);
  if (key === "activation") return JSON.stringify(String(s));
  if (key === "leakyP") {
    const n = typeof s === "number" ? s : Number(s);
    return String(clampLeakyP(Number.isFinite(n) ? n : 0));
  }
  return JSON.stringify(s);
}

function pyTypeForActivationKey(key: keyof ActivationLayerNodeData): string {
  if (key === "leakyP") return "float";
  return "str";
}

export function generateActivationLayerSpecCode(
  d: ActivationLayerNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_ACTIVATION_LAYER_SPEC_NAME;
  const lines: string[] = [`import torch`, ``, `class ${name}(torch.nn.Module):`, `    def __init__(`, `        self,`];
  const merged = { ...defaultActivationLayerData(), ...d };
  const keys = (order.length ? order : DEFAULT_ACTIVATION_LAYER_PARAM_ORDER).filter((k) => KNOWN_KEYS.has(k));
  for (const k of keys) {
    const ck = k as keyof ActivationLayerNodeData;
    const sn = camelToSnakeCase(String(ck));
    lines.push(`        ${sn}: ${pyTypeForActivationKey(ck)} = ${formatActivationLayerPyDefault(ck, merged)},`);
  }
  lines.push(`    ):`);
  lines.push(`        super().__init__()`);
  lines.push(`        m = {`);
  lines.push(`            "relu": torch.nn.ReLU,`);
  lines.push(`            "gelu": torch.nn.GELU,`);
  lines.push(`            "tanh": torch.nn.Tanh,`);
  lines.push(`            "sigmoid": torch.nn.Sigmoid,`);
  lines.push(`            "leaky_relu": torch.nn.LeakyReLU,`);
  lines.push(`            "silu": torch.nn.SiLU,`);
  lines.push(`            "identity": torch.nn.Identity,`);
  lines.push(`        }`);
  lines.push(`        if str(activation) == "leaky_relu":`);
  lines.push(`            self.act = torch.nn.LeakyReLU(negative_slope=float(leaky_p))`);
  lines.push(`        else:`);
  lines.push(`            self.act = m.get(str(activation), torch.nn.ReLU)()`);
  lines.push(``);
  lines.push(`    def forward(self, x: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        return self.act(x)`);
  return lines.join("\n");
}

function parseActivationClassInitHeader(code: string): { className: string; paramsChunk: string; error?: string } {
  const cm = code.match(/^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*torch\.nn\.Module\s*\)\s*:/m);
  if (!cm) return { className: "", paramsChunk: "", error: "Expected class Name(torch.nn.Module):" };
  const className = cm[1]!;
  const after = code.slice(code.indexOf(cm[0]) + cm[0].length);
  const im = after.match(/def\s+__init__\s*\(\s*self\s*,([\s\S]*?)\)\s*:/m);
  if (!im) return { className, paramsChunk: "", error: "Expected __init__(self, ...) in class." };
  return { className, paramsChunk: im[1] ?? "" };
}

export function parseActivationLayerSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<ActivationLayerNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  let specName = "";
  let params: Array<{ snakeName: string; rawValue: string }> = [];
  const cls = parseActivationClassInitHeader(code);
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
  const patch: Partial<ActivationLayerNodeData> = {};
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
    const key = camel as keyof ActivationLayerNodeData;
    if (key === "activation") {
      patch.activation = String(val) as MlpActivationId;
    } else if (key === "leakyP") {
      const n = typeof val === "number" ? val : Number(val);
      if (!Number.isFinite(n)) {
        return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid float for ${camel}` };
      }
      patch.leakyP = clampLeakyP(n);
    }
  }
  const actScalar = String(firstScalar(patch.activation) ?? "");
  if (actScalar === "leaky_relu" && !paramOrder.includes("leakyP")) {
    const ai = paramOrder.indexOf("activation");
    if (ai >= 0) paramOrder.splice(ai + 1, 0, "leakyP");
    else paramOrder.push("leakyP");
    if (patch.leakyP === undefined) patch.leakyP = defaultActivationLayerData().leakyP;
  }
  return { specName, paramOrder, patch, extras };
}
