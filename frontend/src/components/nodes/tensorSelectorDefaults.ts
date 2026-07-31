/** Browser copy of the selected activation tensor (filled after fetch from server). */
export type ActivationTensorCache = {
  runId: string;
  tensorKey: string;
  shape: number[];
  values: number[];
};

/** Stable id of the tensor chosen for downstream analysis wiring (execution TBD). */
export type TensorSelectorNodeData = {
  /** First selected key; kept in sync for older Python / graph consumers. */
  selectedTensorKey: string;
  /** Ordered multi-select (subset of upstream list, top-to-bottom order). Omitted / undefined = not yet set (legacy uses ``selectedTensorKey`` only). ``[]`` = explicitly no tensors selected. */
  selectedTensorKeys?: string[];
  /** Legacy single fetch cache; mirror of first key when present. */
  activationTensorCache: ActivationTensorCache | null;
  /** Per-tensor activation payloads (when upstream is Activation + server run). */
  activationTensorCaches: Partial<Record<string, ActivationTensorCache>>;
  /** True while the list sweep animation is running (downstream sweep table can accumulate rows). */
  tensorSelectorSweeping?: boolean;
  /** Increments on each sweep step; stable across renders for deduping table rows. */
  tensorSelectorSweepSeq?: number;
  /**
   * Per-step tensor ids for each {@link tensorSelectorSweepSeq} (the sliding `slice` at tick time).
   * Lets downstream sweep tables label rows correctly even if async hydration finishes after later ticks.
   */
  tensorSelectorSweepSnapshots?: Partial<Record<number, string[]>>;
};

export function defaultTensorSelectorData(): TensorSelectorNodeData {
  return {
    selectedTensorKey: "",
    activationTensorCache: null,
    activationTensorCaches: {},
    tensorSelectorSweeping: false,
    tensorSelectorSweepSeq: 0,
  };
}

/** Legacy ``selected_tensor`` / empty handle and ``tensor_N`` (1-based). */
export function tensorSelectorOutputIndexFromSourceHandle(sh: string | null | undefined): number {
  const h = (sh ?? "").trim();
  if (!h || h === "selected_tensor") return 0;
  const m = /^tensor_(\d+)$/.exec(h);
  if (m) return Math.max(0, parseInt(m[1]!, 10) - 1);
  return 0;
}

export function isTensorSelectorSourceHandle(sh: string | null | undefined): boolean {
  const h = (sh ?? "").trim();
  return h === "" || h === "selected_tensor" || /^tensor_\d+$/.test(h);
}

/** Merge legacy single-key / single-cache fields into the normalized shape. */
export function normalizeTensorSelectorData(
  raw: Partial<TensorSelectorNodeData> | undefined,
): TensorSelectorNodeData {
  const def = defaultTensorSelectorData();
  const cur = raw ?? {};
  let keys: string[] = [];
  if (Array.isArray(cur.selectedTensorKeys)) {
    keys = cur.selectedTensorKeys.map((k) => String(k).trim()).filter(Boolean);
  } else {
    const one = String(cur.selectedTensorKey ?? "").trim();
    if (one) keys = [one];
  }
  let caches: Partial<Record<string, ActivationTensorCache>> =
    cur.activationTensorCaches && Object.keys(cur.activationTensorCaches).length > 0
      ? { ...cur.activationTensorCaches }
      : {};
  const legacy = cur.activationTensorCache;
  if (legacy && legacy.tensorKey && !caches[legacy.tensorKey]) {
    caches = { ...caches, [legacy.tensorKey]: legacy };
  }
  const primary = keys[0] ?? "";
  const outKeys: string[] | undefined = Array.isArray(cur.selectedTensorKeys)
    ? keys
    : keys.length > 0
      ? keys
      : undefined;
  return {
    selectedTensorKey: primary || cur.selectedTensorKey || def.selectedTensorKey,
    selectedTensorKeys: outKeys,
    activationTensorCache: primary && caches[primary] ? caches[primary]! : legacy ?? def.activationTensorCache,
    activationTensorCaches: caches,
    tensorSelectorSweeping: cur.tensorSelectorSweeping ?? def.tensorSelectorSweeping,
    tensorSelectorSweepSeq:
      typeof cur.tensorSelectorSweepSeq === "number" ? cur.tensorSelectorSweepSeq : def.tensorSelectorSweepSeq,
    tensorSelectorSweepSnapshots:
      cur.tensorSelectorSweepSnapshots && typeof cur.tensorSelectorSweepSnapshots === "object"
        ? { ...cur.tensorSelectorSweepSnapshots }
        : def.tensorSelectorSweepSnapshots,
  };
}
