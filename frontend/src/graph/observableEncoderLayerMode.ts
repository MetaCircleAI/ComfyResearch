/** Matches backend ``_obs_encoder_layer_mode`` / ``observableEncoderLayers`` on several observables. */
export type ObservableEncoderLayerMode = "global" | "all_layers";

export function readObservableEncoderLayerMode(data: object | undefined): ObservableEncoderLayerMode {
  const raw = (data ?? {}) as { observableEncoderLayers?: unknown };
  const v = raw.observableEncoderLayers;
  if (v === "all_layers") return "all_layers";
  return "global";
}

const ORDER: ObservableEncoderLayerMode[] = ["global", "all_layers"];

/** Options for ``DiscreteMultiSelect`` (`singleSelect`). */
export const OBSERVABLE_ENCODER_LAYER_OPTIONS: { id: ObservableEncoderLayerMode; label: string }[] =
  ORDER.map((id) => ({
    id,
    label:
      id === "global"
        ? "Global (average over all layers)"
        : "All layers (one series per layer)",
  }));
