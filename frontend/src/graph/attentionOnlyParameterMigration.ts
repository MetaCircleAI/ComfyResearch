/** Parameter names changed when attention-only models gained a token-LM wrapper. */
const LEGACY_ATTENTION_ONLY_PARAMETER = /^w_[qkvo]\.(?:weight|bias)$/;

/** Convert a pre-wrapper attention parameter name to its current model parameter name. */
export function migrateAttentionOnlyParameterKey(key: string): string {
  return LEGACY_ATTENTION_ONLY_PARAMETER.test(key) ? `block.${key}` : key === "unembed.weight" ? "lm_head.weight" : key;
}

/** Update persisted tensor-selector data without changing unrelated fields. */
export function migrateAttentionOnlyTensorSelectorData(
  data: Record<string, unknown>,
): Record<string, unknown> {
  const selectedTensorKey = typeof data.selectedTensorKey === "string"
    ? migrateAttentionOnlyParameterKey(data.selectedTensorKey)
    : data.selectedTensorKey;
  const selectedTensorKeys = Array.isArray(data.selectedTensorKeys)
    ? data.selectedTensorKeys.map((key) => (
      typeof key === "string" ? migrateAttentionOnlyParameterKey(key) : key
    ))
    : data.selectedTensorKeys;
  return { ...data, selectedTensorKey, selectedTensorKeys };
}

/** Refresh cached parameter-payload keys persisted by model_weight_tensors. */
export function migrateAttentionOnlyWeightTensorPayloads(
  data: Record<string, unknown>,
): Record<string, unknown> {
  const payloads = data.weightTensorPayloads;
  if (!payloads || typeof payloads !== "object" || Array.isArray(payloads)) return data;
  return {
    ...data,
    weightTensorPayloads: Object.fromEntries(
      Object.entries(payloads).map(([key, value]) => [migrateAttentionOnlyParameterKey(key), value]),
    ),
  };
}
