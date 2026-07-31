import type { NumericTransformerModelNodeData } from "../../components/nodes/numericTransformerModelDefaults";
import { defaultNumericTransformerModelData } from "../../components/nodes/numericTransformerModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "contextLength",
  "inputDim",
  "outputDim",
  "modelDim",
  "numHeads",
  "numLayers",
  "ffDim",
  "causalAttention",
  "seed",
]);

export const DEFAULT_NUMERIC_TRANSFORMER_MODEL_SPEC_NAME = "NumericTransformerModel";
export const DEFAULT_NUMERIC_TRANSFORMER_MODEL_PARAM_ORDER: (keyof NumericTransformerModelNodeData)[] = [
  "contextLength",
  "inputDim",
  "outputDim",
  "modelDim",
  "numHeads",
  "numLayers",
  "ffDim",
  "causalAttention",
  "seed",
];

/** Always emitted in generated `__init__` so the body can reference `context_length` / token dims. */
const LAYOUT_INIT_KEYS: (keyof NumericTransformerModelNodeData)[] = ["contextLength", "inputDim", "outputDim"];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function pyTypeForKey(key: keyof NumericTransformerModelNodeData): string {
  if (key === "causalAttention") return "bool";
  return "int";
}

function formatPyDefault(key: keyof NumericTransformerModelNodeData, v: unknown): string {
  if (key === "causalAttention") {
    const s = firstScalar(v);
    const on = s === true || s === "true" || s === "True" || s === "yes" || s === "Yes";
    return on ? "True" : "False";
  }
  const s = firstScalar(v);
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

export function generateNumericTransformerModelSpecCode(
  d: NumericTransformerModelNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_NUMERIC_TRANSFORMER_MODEL_SPEC_NAME;
  const merged = { ...defaultNumericTransformerModelData(), ...d };
  const raw = (order.length ? order : DEFAULT_NUMERIC_TRANSFORMER_MODEL_PARAM_ORDER).filter((k) =>
    KNOWN_KEYS.has(k),
  );
  const layoutSeen = new Set<string>(LAYOUT_INIT_KEYS.map(String));
  const keys = [...LAYOUT_INIT_KEYS, ...raw.filter((k) => !layoutSeen.has(String(k)))];
  const lines: string[] = [
    `import torch`,
    `from comfy_research.engine.transformer_encoder_custom import StableTransformerEncoder, StableTransformerEncoderLayer`,
    ``,
    `class ${name}(torch.nn.Module):`,
    `    def __init__(`,
    `        self,`,
  ];
  for (const k of keys) {
    const ck = k as keyof NumericTransformerModelNodeData;
    lines.push(`        ${camelToSnakeCase(String(ck))}: ${pyTypeForKey(ck)} = ${formatPyDefault(ck, merged[ck])},`);
  }
  lines.push(`    ):`);
  lines.push(`        super().__init__()`);
  lines.push(`        if context_length < 1 or input_dim < 1 or output_dim < 1 or model_dim < 1:`);
  lines.push(`            raise ValueError("context_length, input_dim, output_dim, model_dim must be >= 1")`);
  lines.push(`        if num_heads < 1 or num_layers < 1 or ff_dim < 1:`);
  lines.push(`            raise ValueError("num_heads, num_layers, ff_dim must be >= 1")`);
  lines.push(`        if model_dim % num_heads != 0:`);
  lines.push(`            raise ValueError("model_dim must be divisible by num_heads")`);
  lines.push(`        self.context_length = int(context_length)`);
  lines.push(`        self.token_dim = int(input_dim)`);
  lines.push(`        self.output_token_dim = int(output_dim)`);
  lines.push(`        self.model_dim = int(model_dim)`);
  lines.push(`        self.causal_attention = bool(causal_attention)`);
  lines.push(`        self.seed = int(seed)`);
  lines.push(`        self.token_proj = torch.nn.Linear(self.token_dim, self.model_dim, bias=True)`);
  lines.push(`        self.pos_embed = torch.nn.Parameter(torch.zeros(self.context_length, self.model_dim))`);
  lines.push(
    `        enc_layer = StableTransformerEncoderLayer(self.model_dim, int(num_heads), int(ff_dim), dropout=0.0, activation="gelu", batch_first=True)`,
  );
  lines.push(`        self.encoder = StableTransformerEncoder(enc_layer, num_layers=int(num_layers))`);
  lines.push(`        self.out_proj = torch.nn.Linear(self.model_dim, self.output_token_dim, bias=True)`);
  lines.push(``);
  lines.push(`    def forward(self, x: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        if x.dim() == 2:`);
  lines.push(`            if int(x.shape[1]) == self.context_length * self.token_dim:`);
  lines.push(`                x = x.reshape(x.shape[0], self.context_length, self.token_dim)`);
  lines.push(`            elif self.token_dim == 1 and int(x.shape[1]) == self.context_length:`);
  lines.push(`                x = x.unsqueeze(-1)`);
  lines.push(`            else:`);
  lines.push(`                raise ValueError(f"bad x shape {tuple(x.shape)}")`);
  lines.push(`        if x.dim() != 3:`);
  lines.push(`            raise ValueError("x must be [batch, T, D] or compatible flat/broadcast rank-2")`);
  lines.push(`        tok = self.token_proj(x) + self.pos_embed.unsqueeze(0)`);
  lines.push(`        h = self.encoder(tok, is_causal=self.causal_attention)`);
  lines.push(`        return self.out_proj(h)`);
  return lines.join("\n");
}

export function parseNumericTransformerModelSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<NumericTransformerModelNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<NumericTransformerModelNodeData> = {};
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
    const key = camel as keyof NumericTransformerModelNodeData;
    if (key === "causalAttention") {
      const on = val === true || val === "True" || val === "true" || val === "yes" || val === "Yes";
      patch.causalAttention = (on ? "yes" : "no") as never;
      continue;
    }
    const n = typeof val === "number" ? val : Number(val);
    if (!Number.isFinite(n) || !Number.isInteger(n)) {
      return { specName: "", paramOrder: [], patch: {}, extras: {}, error: `Invalid int for ${camel}` };
    }
    patch[key] = n as never;
  }
  return { specName: p.funcName, paramOrder, patch, extras };
}
