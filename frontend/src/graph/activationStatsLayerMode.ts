/** Matches backend ``_activation_stats_layer_mode`` / ``activationStatsLayers`` on activation stats observable nodes. */
export type ActivationStatsLayerMode = "global" | "all_layers";

export function readActivationStatsLayerMode(data: object | undefined): ActivationStatsLayerMode {
  const raw = (data ?? {}) as { activationStatsLayers?: unknown };
  const v = raw.activationStatsLayers;
  if (v === "all_layers") return "all_layers";
  return "global";
}

const ORDER: ActivationStatsLayerMode[] = ["global", "all_layers"];

/** Options for ``DiscreteMultiSelect`` (`singleSelect`). */
export const ACTIVATION_STATS_LAYER_OPTIONS: { id: ActivationStatsLayerMode; label: string }[] = ORDER.map((id) => ({
  id,
  label:
    id === "global"
      ? "Global (average over all layers)"
      : "All layers (mean/std per layer)",
}));
