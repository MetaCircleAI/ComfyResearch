import type { CurvePoint } from "./observableCurvePayload";

export type LpdSegment = {
  phase_index?: number;
  t_start: number;
  t_end: number;
  mechanism?: string;
  mechanism_label?: string;
  formula?: string;
  segment_r2?: number;
};

export type LpdPredictResult = {
  breakpoints?: number[];
  segments?: LpdSegment[];
  global_r2?: number;
  mean_segment_r2?: number;
  num_phases?: number;
  data?: CurvePoint[];
  fitted?: CurvePoint[];
  auto_k?: boolean;
  selected_k?: number;
  auto_k_stopped_early?: boolean;
};

export type CurveStarerAnalyzedEntry = {
  entryId: string;
  nodeId: string;
  seriesId: string;
  label: string;
  yAxisLabel: string;
  points: CurvePoint[];
  lpd: LpdPredictResult | null;
  lpdError?: string;
};

export type CurveStarerRankBy =
  | "default"
  | "hard_to_fit"
  | "easy_to_fit"
  | "few_stages"
  | "many_stages"
  | "small_variation"
  | "large_variation"
  | "interestingness";

export type LpdBatchItemResult =
  | { ok: true; result: LpdPredictResult }
  | { ok: false; error: string };

const DEFAULT_LPD_PREDICT_BODY = {
  K: 3,
  auto_k: true,
  k_max: 6,
  auto_k_target: 0.99,
  simplicity: 0.25,
} as const;

/** Parallel LPD requests; keep modest so CPU stays responsive with many curves. */
export const LPD_PREDICT_CONCURRENCY = 8;

async function runWithConcurrencyLimit<T>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<void>,
): Promise<void> {
  if (items.length === 0) return;
  const workers = Math.min(Math.max(1, limit), items.length);
  let cursor = 0;
  const runWorker = async () => {
    for (;;) {
      const index = cursor++;
      if (index >= items.length) return;
      await fn(items[index]!, index);
    }
  };
  await Promise.all(Array.from({ length: workers }, () => runWorker()));
}

export async function fetchLpdPredict(points: CurvePoint[]): Promise<LpdPredictResult> {
  const res = await fetch("/api/curve-lpd/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points,
      ...DEFAULT_LPD_PREDICT_BODY,
    }),
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = (await res.json()) as { detail?: unknown };
      if (j.detail != null) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return (await res.json()) as LpdPredictResult;
}

export async function fetchLpdPredictBatch(
  curves: CurvePoint[][],
  options?: {
    concurrency?: number;
    onResult?: (index: number, result: LpdBatchItemResult) => void;
  },
): Promise<LpdBatchItemResult[]> {
  if (curves.length === 0) return [];

  const concurrency = options?.concurrency ?? LPD_PREDICT_CONCURRENCY;
  const results: LpdBatchItemResult[] = new Array(curves.length);

  await runWithConcurrencyLimit(curves, concurrency, async (points, index) => {
    let item: LpdBatchItemResult;
    try {
      const result = await fetchLpdPredict(points);
      item = { ok: true, result };
    } catch (e) {
      item = { ok: false, error: e instanceof Error ? e.message : String(e) };
    }
    results[index] = item;
    options?.onResult?.(index, item);
  });

  return results;
}

export async function analyzeObservableCurvesWithLpd(
  entries: {
    entryId: string;
    nodeId: string;
    seriesId: string;
    label: string;
    yAxisLabel: string;
    points: CurvePoint[];
  }[],
): Promise<CurveStarerAnalyzedEntry[]> {
  const batch = await fetchLpdPredictBatch(entries.map((e) => e.points));
  return entries.map((entry, i) => {
    const item = batch[i];
    if (item?.ok) return { ...entry, lpd: item.result };
    return {
      ...entry,
      lpd: null,
      lpdError: item?.ok === false ? item.error : "Missing LPD result",
    };
  });
}
