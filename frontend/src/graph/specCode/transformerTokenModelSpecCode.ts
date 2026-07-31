import type { TransformerTokenModelNodeData } from "../../components/nodes/transformerTokenModelDefaults";
import { defaultTransformerTokenModelData } from "../../components/nodes/transformerTokenModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "contextLength",
  "vocabSize",
  "modelDim",
  "numHeads",
  "numLayers",
  "ffDim",
  "activation",
  "tieEmbeddingLmHead",
  "causalAttention",
  "localMixingKernel",
  "seed",
]);

export const DEFAULT_TRANSFORMER_TOKEN_MODEL_SPEC_NAME = "TransformerTokenModel";
export const DEFAULT_TRANSFORMER_TOKEN_MODEL_PARAM_ORDER: (keyof TransformerTokenModelNodeData)[] = [
  "contextLength",
  "vocabSize",
  "modelDim",
  "numHeads",
  "numLayers",
  "ffDim",
  "activation",
  "tieEmbeddingLmHead",
  "causalAttention",
  "localMixingKernel",
  "seed",
];

const LAYOUT_INIT_KEYS: (keyof TransformerTokenModelNodeData)[] = ["contextLength", "vocabSize"];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyIntDefault(key: keyof TransformerTokenModelNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

function yesNoToBoolPy(v: unknown): string {
  const s = String(firstScalar(v) ?? "yes").trim().toLowerCase();
  return s === "no" || s === "false" || s === "0" ? "False" : "True";
}

function normalizeEncoderActivation(v: unknown): "gelu" | "relu" | "silu" {
  const s = String(firstScalar(v) ?? "gelu").trim().toLowerCase();
  if (s === "relu") return "relu";
  if (s === "silu") return "silu";
  return "gelu";
}

export function generateTransformerTokenModelSpecCode(
  d: TransformerTokenModelNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_TRANSFORMER_TOKEN_MODEL_SPEC_NAME;
  const merged = { ...defaultTransformerTokenModelData(), ...d };
  const lmRaw = Number(firstScalar(merged.localMixingKernel) ?? 0);
  const lmK =
    Number.isFinite(lmRaw) && lmRaw >= 3 ? (Math.floor(lmRaw) % 2 === 1 ? Math.floor(lmRaw) : Math.floor(lmRaw) + 1) : 0;
  const raw = (order.length ? order : DEFAULT_TRANSFORMER_TOKEN_MODEL_PARAM_ORDER).filter((k) =>
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
    const ck = k as keyof TransformerTokenModelNodeData;
    if (ck === "tieEmbeddingLmHead" || ck === "causalAttention") {
      lines.push(`        ${camelToSnakeCase(String(ck))}: bool = ${yesNoToBoolPy(merged[ck])},`);
    } else if (ck === "activation") {
      const act = normalizeEncoderActivation(merged[ck]);
      lines.push(`        activation: str = "${act}",`);
    } else {
      lines.push(`        ${camelToSnakeCase(String(ck))}: int = ${formatPyIntDefault(ck, merged[ck])},`);
    }
  }
  lines.push(`    ):`);
  lines.push(`        super().__init__()`);
  lines.push(`        if context_length < 1 or vocab_size < 2 or model_dim < 1:`);
  lines.push(`            raise ValueError("context_length must be >= 1, vocab_size >= 2, model_dim >= 1")`);
  lines.push(`        if num_heads < 1 or num_layers < 1 or ff_dim < 1:`);
  lines.push(`            raise ValueError("num_heads, num_layers, ff_dim must be >= 1")`);
  lines.push(`        if model_dim % num_heads != 0:`);
  lines.push(`            raise ValueError("model_dim must be divisible by num_heads")`);
  lines.push(`        self.context_length = int(context_length)`);
  lines.push(`        self.vocab_size = int(vocab_size)`);
  lines.push(`        self.model_dim = int(model_dim)`);
  lines.push(`        self.seed = int(seed)`);
  lines.push(`        self.causal_attention = bool(causal_attention)`);
  lines.push(`        self.tie_embedding_lm_head = bool(tie_embedding_lm_head)`);
  lines.push(`        self.embedding = torch.nn.Embedding(self.vocab_size, self.model_dim)`);
  lines.push(`        self.pos_embed = torch.nn.Parameter(torch.zeros(self.context_length, self.model_dim))`);
  if (lmK >= 3) {
    lines.push(`        self._canon_kernel = int(${lmK})`);
    lines.push(
      `        self.local_mix = torch.nn.Conv1d(self.model_dim, self.model_dim, self._canon_kernel, groups=self.model_dim, bias=True)`,
    );
  } else {
    lines.push(`        self.local_mix = None`);
  }
  lines.push(
    `        enc_layer = StableTransformerEncoderLayer(self.model_dim, int(num_heads), int(ff_dim), dropout=0.0, activation=activation, batch_first=True)`,
  );
  lines.push(`        self.encoder = StableTransformerEncoder(enc_layer, num_layers=int(num_layers))`);
  lines.push(
    `        self.lm_head = torch.nn.Linear(self.model_dim, self.vocab_size, bias=not self.tie_embedding_lm_head)`,
  );
  lines.push(`        if self.tie_embedding_lm_head:`);
  lines.push(`            self.embedding.weight = self.lm_head.weight`);
  lines.push(``);
  lines.push(`    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        if token_ids.dim() != 2:`);
  lines.push(`            raise ValueError("token_ids must be [batch, seq_len]")`);
  lines.push(`        if int(token_ids.shape[1]) != self.context_length:`);
  lines.push(`            raise ValueError(f"seq_len {int(token_ids.shape[1])} != context_length {self.context_length}")`);
  lines.push(`        x = self.embedding(token_ids.long()) + self.pos_embed.unsqueeze(0)`);
  lines.push(`        if self.local_mix is not None:`);
  lines.push(
    `            z = torch.nn.functional.pad(x.transpose(1, 2), (self._canon_kernel - 1, 0))`,
  );
  lines.push(`            x = x + self.local_mix(z).transpose(1, 2)`);
  lines.push(`        h = self.encoder(x, is_causal=self.causal_attention)`);
  lines.push(`        return self.lm_head(h[:, -1, :])`);
  return lines.join("\n");
}

export function parseTransformerTokenModelSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<TransformerTokenModelNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<TransformerTokenModelNodeData> = {};
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
    const key = camel as keyof TransformerTokenModelNodeData;
    if (key === "tieEmbeddingLmHead" || key === "causalAttention") {
      const b = val === true || String(val).toLowerCase() === "true" || String(val).toLowerCase() === "yes";
      patch[key] = (b ? "yes" : "no") as never;
      continue;
    }
    if (key === "activation") {
      patch[key] = normalizeEncoderActivation(val) as never;
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
