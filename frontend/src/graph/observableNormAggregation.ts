/** Matches backend ``_observable_l2_aggregation`` / ``normAggregation`` on observable nodes. */
export type NormAggregation = "global" | "top_level_module" | "tensor";

export function readNormAggregation(data: object | undefined): NormAggregation {
  const raw = (data ?? {}) as { normAggregation?: unknown; perTopLevel?: unknown };
  const v = raw.normAggregation;
  if (v === "global" || v === "top_level_module" || v === "tensor") return v;
  if (Boolean(raw.perTopLevel)) return "top_level_module";
  return "global";
}

export const NORM_AGGREGATION_LABELS: Record<NormAggregation, string> = {
  global: "Global (single ‖·‖₂)",
  top_level_module: "Top-level module",
  tensor: "Per-parameter tensor",
};

const NORM_AGGREGATION_ORDER: NormAggregation[] = ["global", "top_level_module", "tensor"];

/** Options for `DiscreteMultiSelect` (`singleSelect`), stable menu order. */
export const NORM_AGGREGATION_OPTIONS: { id: NormAggregation; label: string }[] = NORM_AGGREGATION_ORDER.map((id) => ({
  id,
  label: NORM_AGGREGATION_LABELS[id],
}));
