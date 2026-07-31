/** Matches backend ``_sink_attention_mass_layer_mode`` / ``sinkAttentionMassLayers`` on sink attention observable nodes. */
export type SinkAttentionMassLayerMode = "global" | "all_layers";

export function readSinkAttentionMassLayerMode(data: object | undefined): SinkAttentionMassLayerMode {
  const raw = (data ?? {}) as { sinkAttentionMassLayers?: unknown };
  const v = raw.sinkAttentionMassLayers;
  if (v === "all_layers") return "all_layers";
  return "global";
}

const ORDER: SinkAttentionMassLayerMode[] = ["global", "all_layers"];

/** Options for ``DiscreteMultiSelect`` (`singleSelect`). */
export const SINK_ATTENTION_MASS_LAYER_OPTIONS: { id: SinkAttentionMassLayerMode; label: string }[] =
  ORDER.map((id) => ({
    id,
    label:
      id === "global"
        ? "Global (average over all layers)"
        : "All layers (one series per layer)",
  }));
