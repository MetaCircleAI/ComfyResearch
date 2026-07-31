/**
 * Group full parameter paths (gradient / weight L2 per-tensor breakdown) into a stable "kind"
 * so the viz can show e.g. all `linear1.weight` curves across layers together.
 */

/** True when labels look like per-parameter paths with `...layers.{i}....` (vs top-level segments only). */
export function isPerTensorLayerSeriesLabels(labels: string[]): boolean {
  if (labels.length < 2) return false;
  return labels.slice(1).some((lab) => /\.layers\.\d+\./.test(lab));
}

/**
 * Suffix after `...layers.{n}.` if present; otherwise the full label (e.g. `embedding.weight`, `pos_embed`).
 */
export function tensorFamilyFromParameterLabel(label: string): string {
  const m = label.match(/\.layers\.\d+\.(.+)$/);
  if (m) return m[1]!;
  return label;
}

/** Legend / checkbox text: layer index if matched, else full label. */
export function tensorCurveShortLabel(fullLabel: string): string {
  if (fullLabel === "global") return "global";
  const m = fullLabel.match(/\.layers\.(\d+)\./);
  if (m) return `layer ${m[1]}`;
  return fullLabel;
}

export function sortedTensorFamiliesFromLabels(labels: string[]): string[] {
  const fams = new Set<string>();
  for (let i = 1; i < labels.length; i++) {
    const lab = labels[i];
    if (typeof lab === "string" && lab.length > 0) fams.add(tensorFamilyFromParameterLabel(lab));
  }
  return [...fams].sort((a, b) => a.localeCompare(b));
}
