/** Throttle trainer ``persistTrainUi`` during NDJSON progress (each call runs ``setNodes``). */

export type TrainProgressUiPatch = {
  progressPct: number;
  seriesBarPct: number;
  phaseText: string | null;
};

export function createTrainProgressUiThrottler(args: {
  flush: (patch: TrainProgressUiPatch) => void;
  intervalMs?: number;
}): {
  schedule: (patch: TrainProgressUiPatch, force?: boolean) => void;
  cancel: () => void;
  flushNow: () => void;
} {
  const intervalMs = args.intervalMs ?? 250;
  let lastAt = 0;
  let pending: ReturnType<typeof setTimeout> | null = null;
  let lastPatch: TrainProgressUiPatch | null = null;

  const cancel = () => {
    if (pending !== null) {
      window.clearTimeout(pending);
      pending = null;
    }
    lastPatch = null;
  };

  const flushNow = () => {
    if (pending !== null) {
      window.clearTimeout(pending);
      pending = null;
    }
    if (!lastPatch) return;
    args.flush(lastPatch);
    lastPatch = null;
    lastAt = performance.now();
  };

  const schedule = (patch: TrainProgressUiPatch, force = false) => {
    lastPatch = patch;
    if (force || patch.progressPct >= 100) {
      flushNow();
      return;
    }
    const now = performance.now();
    if (now - lastAt >= intervalMs) {
      flushNow();
      return;
    }
    if (pending === null) {
      pending = window.setTimeout(flushNow, intervalMs - (now - lastAt));
    }
  };

  return { schedule, cancel, flushNow };
}
