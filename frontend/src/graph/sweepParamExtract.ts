import {
  formatTrainSeriesSweptLines,
  type SerializedTrainNode,
  type TrainSeriesAssignment,
} from "./trainSeriesPlan";

/** Coerce a single trainer config scalar (not a comma sweep list). */
export function scalarTrainConfigField(value: unknown): string | null {
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "string") {
    const t = value.trim();
    if (!t || t.includes(",")) return null;
    const n = Number(t);
    if (Number.isFinite(n)) return String(n);
  }
  return null;
}

/** Snapshot trainer run params for sweep / curve series rows. */
export function buildTrainerRunSweepParams(trainerData: Record<string, unknown>): Record<string, string> {
  const out: Record<string, string> = {};
  const bs = scalarTrainConfigField(trainerData.batchSize);
  const steps = scalarTrainConfigField(trainerData.trainingSteps);
  if (bs != null) out["trainer.batchSize"] = bs;
  if (steps != null) out["trainer.trainingSteps"] = steps;
  if (trainerData.trainingLengthMode === "epochs") {
    const ep = scalarTrainConfigField(trainerData.trainingEpochs);
    if (ep != null) out["trainer.trainingEpochs"] = ep;
  }
  const runTitle = typeof trainerData.instanceTitle === "string" ? trainerData.instanceTitle.trim() : "";
  if (runTitle) out["trainer.run"] = runTitle;
  return out;
}

export function formatSweepParamsSummary(params: Record<string, string>): string {
  return Object.entries(params)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
}

/** Parse comma-separated ``prefix.field=value`` tokens from ``lastSweepSummary``. */
export function parseSweepParamsFromSummary(raw: string): Record<string, string> {
  const out: Record<string, string> = {};
  const trimmed = raw.trim();
  if (!trimmed) return out;
  for (const seg of trimmed.split(",").map((s) => s.trim()).filter(Boolean)) {
    const eq = seg.indexOf("=");
    if (eq === -1) {
      out.run = seg;
      continue;
    }
    out[seg.slice(0, eq).trim()] = seg.slice(eq + 1).trim();
  }
  return out;
}

/**
 * Build sweep param map from the train-series combo (same keys as {@link formatTrainSeriesSweptLines}).
 * Prefer this over re-parsing ``lastSweepSummary`` so values stay aligned with what was actually run.
 */
export function buildSweepParamsFromCombo(
  combo: TrainSeriesAssignment[],
  nodes: SerializedTrainNode[],
  sweptAxisIds: Set<string>,
): Record<string, string> {
  const lines = formatTrainSeriesSweptLines(combo, nodes, sweptAxisIds);
  if (lines.length === 1 && lines[0] === "defaults") return {};
  const out: Record<string, string> = {};
  for (const line of lines) {
    const eq = line.indexOf("=");
    if (eq === -1) {
      out.run = line.trim();
      continue;
    }
    out[line.slice(0, eq).trim()] = line.slice(eq + 1).trim();
  }
  return out;
}

/** Structured params win; summary-parsed keys fill gaps only. */
export function mergeSweepParamRecords(
  structured: Record<string, string>,
  fromSummary: Record<string, string>,
): Record<string, string> {
  return { ...fromSummary, ...structured };
}

export function parseSweepParamNumeric(value: string): number | null {
  const t = value.trim();
  if (!t) return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

/** Coerce string sweep params to numbers where possible (for calculators / plots). */
export function coerceSweepParamsNumeric(params: Record<string, string>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(params)) {
    const n = parseSweepParamNumeric(v);
    if (n !== null) out[k] = n;
  }
  return out;
}

/** Look up a param by full key or suffix (``learningRate`` → ``sgd.learningRate``). */
export function resolveSweepParamValue(params: Record<string, string>, key: string): string | undefined {
  const k = key.trim();
  if (!k) return undefined;
  if (params[k] !== undefined) return params[k];
  const suffix = `.${k}`;
  const matches = Object.entries(params).filter(([full]) => full.endsWith(suffix) || full === k);
  if (matches.length === 1) return matches[0]![1];
  return undefined;
}
