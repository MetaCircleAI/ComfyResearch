import type { MlpActivationId } from "../../components/nodes/mlpModelDefaults";
import type { MlpTokenModelNodeData } from "../../components/nodes/mlpTokenModelDefaults";
import { defaultMlpTokenModelData } from "../../components/nodes/mlpTokenModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";
import { parseMlpClassInitHeader } from "./mlpModelSpecCode";

const KNOWN_KEYS = new Set([
  "vocabSize",
  "embedDim",
  "tokensPerInput",
  "depth",
  "width",
  "numExperts",
  "activation",
  "tieWeights",
  "seed",
]);
export const DEFAULT_MLP_TOKEN_MODEL_SPEC_NAME = "MLPTokenModel";
export const DEFAULT_MLP_TOKEN_MODEL_PARAM_ORDER: (keyof MlpTokenModelNodeData)[] = [
  "vocabSize",
  "embedDim",
  "tokensPerInput",
  "depth",
  "width",
  "numExperts",
  "activation",
  "tieWeights",
  "seed",
];

function pyTypeForKey(key: keyof MlpTokenModelNodeData): string {
  if (key === "activation" || key === "tieWeights") return "str";
  return "int";
}

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyDefault(key: keyof MlpTokenModelNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (key === "activation" || key === "tieWeights") return JSON.stringify(String(s));
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

export function generateMlpTokenModelSpecCode(d: MlpTokenModelNodeData, order: string[], specName: string): string {
  return generateMlpTokenModelVariantSpecCode(d, order, specName, "plain");
}

export function generateMlpTokenModelVariantSpecCode(
  d: MlpTokenModelNodeData,
  order: string[],
  specName: string,
  variant: "plain" | "gated" | "moe",
): string {
  const name = specName.trim() || DEFAULT_MLP_TOKEN_MODEL_SPEC_NAME;
  const lines: string[] = [`import torch`, ``, `class ${name}(torch.nn.Module):`, `    def __init__(`, `        self,`];
  const merged = { ...defaultMlpTokenModelData(), ...d };
  const keys = (order.length ? order : DEFAULT_MLP_TOKEN_MODEL_PARAM_ORDER)
    .filter((k) => KNOWN_KEYS.has(k))
    .filter((k) => (variant === "moe" ? true : k !== "numExperts"));
  for (const k of keys) {
    const ck = k as keyof MlpTokenModelNodeData;
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
  lines.push(`        self.vocab_size = int(vocab_size)`);
  lines.push(`        self.embed_dim = int(embed_dim)`);
  lines.push(`        self.tokens_per_input = int(tokens_per_input)`);
  lines.push(`        self.tie_weights = str(tie_weights).lower() not in ("no", "false", "0")`);
  lines.push(`        self.seed = int(seed)`);
  lines.push(`        d_flat = int(self.embed_dim) * int(self.tokens_per_input)`);
  lines.push(`        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_dim)`);
  if (variant === "gated") {
    lines.push(`        self.act = acts.get(str(activation), torch.nn.SiLU)()`);
    lines.push(`        self.gates = torch.nn.ModuleList()`);
    lines.push(`        self.values = torch.nn.ModuleList()`);
    lines.push(`        in_f = d_flat`);
    lines.push(`        for _ in range(int(depth)):`); 
    lines.push(`            self.gates.append(torch.nn.Linear(in_f, int(width)))`);
    lines.push(`            self.values.append(torch.nn.Linear(in_f, int(width)))`);
    lines.push(`            in_f = int(width)`);
    lines.push(`        self.body_out = torch.nn.Linear(in_f, d_flat)`);
  } else if (variant === "moe") {
    lines.push(`        act_cls = acts.get(str(activation), torch.nn.SiLU)`);
    lines.push(`        self.gate = torch.nn.Linear(d_flat, int(num_experts))`);
    lines.push(`        self.experts = torch.nn.ModuleList()`);
    lines.push(`        for _ in range(int(num_experts)):`); 
    lines.push(`            layers = []`);
    lines.push(`            in_f = d_flat`);
    lines.push(`            for _ in range(int(depth)):`); 
    lines.push(`                layers.append(torch.nn.Linear(in_f, int(width)))`);
    lines.push(`                layers.append(act_cls())`);
    lines.push(`                in_f = int(width)`);
    lines.push(`            layers.append(torch.nn.Linear(in_f, d_flat))`);
    lines.push(`            self.experts.append(torch.nn.Sequential(*layers))`);
  } else {
    lines.push(`        body_layers: list[torch.nn.Module] = []`);
    lines.push(`        in_f = d_flat`);
    lines.push(`        for _ in range(int(depth)):`);
    lines.push(`            body_layers.append(torch.nn.Linear(in_f, int(width)))`);
    lines.push(`            body_layers.append(acts.get(str(activation), torch.nn.ReLU)())`);
    lines.push(`            in_f = int(width)`);
    lines.push(`        body_layers.append(torch.nn.Linear(in_f, d_flat))`);
    lines.push(`        self.body = torch.nn.Sequential(*body_layers)`);
  }
  lines.push(`        self.unembed = torch.nn.Linear(d_flat, self.vocab_size, bias=True)`);
  lines.push(`        if self.tie_weights and self.unembed.weight.shape == self.embedding.weight.shape:`);
  lines.push(`            self.embedding.weight = self.unembed.weight`);
  lines.push(``);
  lines.push(`    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        x = token_ids.long()`);
  lines.push(`        if x.ndim != 2:`);
  lines.push(`            raise ValueError("MLPTokenModel expects shape [batch, tokens_per_input]")`);
  lines.push(`        if x.shape[1] != self.tokens_per_input:`);
  lines.push(`            raise ValueError("tokens_per_input must match input width")`);
  lines.push(`        h = self.embedding(x).reshape(x.shape[0], -1)`);
  if (variant === "gated") {
    lines.push(`        for gate, value in zip(self.gates, self.values):`);
    lines.push(`            h = self.act(gate(h)) * value(h)`);
    lines.push(`        h = self.body_out(h)`);
  } else if (variant === "moe") {
    lines.push(`        g = torch.softmax(self.gate(h), dim=-1)`);
    lines.push(`        ys = [expert(h) for expert in self.experts]`);
    lines.push(`        stacked = torch.stack(ys, dim=1)`);
    lines.push(`        h = (stacked * g.unsqueeze(-1)).sum(dim=1)`);
  } else {
    lines.push(`        h = self.body(h)`);
  }
  lines.push(`        if self.tie_weights and self.unembed.weight.shape == self.embedding.weight.shape:`);
  lines.push(`            return torch.nn.functional.linear(h, self.unembed.weight, self.unembed.bias)`);
  lines.push(`        return self.unembed(h)`);
  return lines.join("\n");
}

export function parseMlpTokenModelSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<MlpTokenModelNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  let specName = "";
  let params: Array<{ snakeName: string; rawValue: string }> = [];
  const cls = parseMlpClassInitHeader(code);
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
  const patch: Partial<MlpTokenModelNodeData> = {};
  const extras: Record<string, string | number | boolean> = {};
  const paramOrder: string[] = [];
  for (const row of params) {
    const camel = snakeToCamelCase(row.snakeName);
    const val = parsePythonDefault(row.rawValue);
    if (!KNOWN_KEYS.has(camel)) {
      if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") extras[camel] = val;
      continue;
    }
    paramOrder.push(camel);
    if (camel === "tieWeights") {
      patch.tieWeights = String(val).toLowerCase() === "no" ? "no" : "yes";
      continue;
    }
    if (camel === "activation") {
      patch.activation = String(val) as MlpActivationId;
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[camel as keyof MlpTokenModelNodeData] = n as never;
  }
  return { specName, paramOrder, patch, extras };
}
