import type { TransformerMultiTokenModelNodeData } from "../../components/nodes/transformerMultiTokenModelDefaults";
import { defaultTransformerMultiTokenModelData } from "../../components/nodes/transformerMultiTokenModelDefaults";
import { camelToSnakeCase, parsePythonDefault, parsePythonFunctionSpecHeader, snakeToCamelCase } from "./pythonFuncSpec";

const KNOWN_KEYS = new Set([
  "contextLength",
  "vocabSize",
  "tokensPerPosition",
  "modelDim",
  "numHeads",
  "numLayers",
  "ffDim",
  "tieEmbeddingLmHead",
  "causalAttention",
  "seed",
]);

export const DEFAULT_TRANSFORMER_MULTI_TOKEN_MODEL_SPEC_NAME = "MultiTokenTransformerModel";
export const DEFAULT_TRANSFORMER_MULTI_TOKEN_MODEL_PARAM_ORDER: (keyof TransformerMultiTokenModelNodeData)[] = [
  "contextLength",
  "vocabSize",
  "tokensPerPosition",
  "modelDim",
  "numHeads",
  "numLayers",
  "ffDim",
  "tieEmbeddingLmHead",
  "causalAttention",
  "seed",
];

const LAYOUT_INIT_KEYS: (keyof TransformerMultiTokenModelNodeData)[] = [
  "contextLength",
  "vocabSize",
  "tokensPerPosition",
];

function firstScalar(v: unknown): unknown {
  return Array.isArray(v) && v.length ? v[0] : v;
}

function formatPyIntDefault(key: keyof TransformerMultiTokenModelNodeData, v: unknown): string {
  const s = firstScalar(v);
  if (typeof s === "number") return String(s);
  return JSON.stringify(s);
}

function yesNoToBoolPy(v: unknown): string {
  const s = String(firstScalar(v) ?? "yes").trim().toLowerCase();
  return s === "no" || s === "false" || s === "0" ? "False" : "True";
}

export function generateTransformerMultiTokenModelSpecCode(
  d: TransformerMultiTokenModelNodeData,
  order: string[],
  specName: string,
): string {
  const name = specName.trim() || DEFAULT_TRANSFORMER_MULTI_TOKEN_MODEL_SPEC_NAME;
  const merged = { ...defaultTransformerMultiTokenModelData(), ...d };
  const raw = (order.length ? order : DEFAULT_TRANSFORMER_MULTI_TOKEN_MODEL_PARAM_ORDER).filter((k) =>
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
    const ck = k as keyof TransformerMultiTokenModelNodeData;
    if (ck === "tieEmbeddingLmHead" || ck === "causalAttention") {
      lines.push(`        ${camelToSnakeCase(String(ck))}: bool = ${yesNoToBoolPy(merged[ck])},`);
    } else {
      lines.push(`        ${camelToSnakeCase(String(ck))}: int = ${formatPyIntDefault(ck, merged[ck])},`);
    }
  }
  lines.push(`    ):`);
  lines.push(`        super().__init__()`);
  lines.push(`        if context_length < 1 or vocab_size < 2 or model_dim < 1 or tokens_per_position < 1:`);
  lines.push(`            raise ValueError("context_length, vocab_size, model_dim, tokens_per_position must be valid")`);
  lines.push(`        if num_heads < 1 or num_layers < 1 or ff_dim < 1:`);
  lines.push(`            raise ValueError("num_heads, num_layers, ff_dim must be >= 1")`);
  lines.push(`        if model_dim % num_heads != 0:`);
  lines.push(`            raise ValueError("model_dim must be divisible by num_heads")`);
  lines.push(`        self.context_length = int(context_length)`);
  lines.push(`        self.vocab_size = int(vocab_size)`);
  lines.push(`        self.tokens_per_position = int(tokens_per_position)`);
  lines.push(`        self.model_dim = int(model_dim)`);
  lines.push(`        self.seed = int(seed)`);
  lines.push(`        self.causal_attention = bool(causal_attention)`);
  lines.push(`        self.tie_embedding_lm_head = bool(tie_embedding_lm_head)`);
  lines.push(`        self.embedding = torch.nn.Embedding(self.vocab_size, self.model_dim)`);
  lines.push(`        fused_in = self.tokens_per_position * self.model_dim`);
  lines.push(`        self.token_fuse = torch.nn.Linear(fused_in, self.model_dim)`);
  lines.push(`        self.pos_embed = torch.nn.Parameter(torch.zeros(self.context_length, self.model_dim))`);
  lines.push(
    `        enc_layer = StableTransformerEncoderLayer(self.model_dim, int(num_heads), int(ff_dim), dropout=0.0, activation="gelu", batch_first=True)`,
  );
  lines.push(`        self.encoder = StableTransformerEncoder(enc_layer, num_layers=int(num_layers))`);
  lines.push(
    `        self.lm_heads = torch.nn.ModuleList([torch.nn.Linear(self.model_dim, self.vocab_size, bias=not self.tie_embedding_lm_head) for _ in range(self.tokens_per_position)])`,
  );
  lines.push(`        if self.tie_embedding_lm_head:`);
  lines.push(`            _ref = self.lm_heads[0].weight`);
  lines.push(`            self.embedding.weight = _ref`);
  lines.push(`            for _head in self.lm_heads[1:]:`);
  lines.push(`                _head.weight = _ref`);
  lines.push(``);
  lines.push(`    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:`);
  lines.push(`        if token_ids.dim() != 3:`);
  lines.push(`            raise ValueError("token_ids must be [batch, seq_len, tokens_per_position]")`);
  lines.push(`        b, t, k = token_ids.shape`);
  lines.push(`        if int(k) != self.tokens_per_position or int(t) != self.context_length:`);
  lines.push(`            raise ValueError("shape mismatch vs context_length / tokens_per_position")`);
  lines.push(`        e = self.embedding(token_ids.long())`);
  lines.push(`        fused = self.token_fuse(e.reshape(b, t, -1))`);
  lines.push(`        x = fused + self.pos_embed.unsqueeze(0)`);
  lines.push(`        h = self.encoder(x, is_causal=self.causal_attention)`);
  lines.push(`        h_last = h[:, -1, :]`);
  lines.push(`        return torch.stack([head(h_last) for head in self.lm_heads], dim=1)`);
  return lines.join("\n");
}

export function parseTransformerMultiTokenModelSpecCode(code: string): {
  specName: string;
  paramOrder: string[];
  patch: Partial<TransformerMultiTokenModelNodeData>;
  extras: Record<string, string | number | boolean>;
  error?: string;
} {
  const p = parsePythonFunctionSpecHeader(code);
  if (p.error) return { specName: "", paramOrder: [], patch: {}, extras: {}, error: p.error };
  const patch: Partial<TransformerMultiTokenModelNodeData> = {};
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
    const key = camel as keyof TransformerMultiTokenModelNodeData;
    if (key === "tieEmbeddingLmHead" || key === "causalAttention") {
      const b = val === true || String(val).toLowerCase() === "true" || String(val).toLowerCase() === "yes";
      patch[key] = (b ? "yes" : "no") as never;
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
