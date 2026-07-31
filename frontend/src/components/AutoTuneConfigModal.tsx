import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { createPortal } from "react-dom";

export type AutoTuneAxisInput = {
  nodeId: string;
  path: string;
  values: number[];
};

export type AutoTuneConfig = {
  axes: AutoTuneAxisInput[];
  maxRounds: number;
  endWeight: number;
  smoothnessWeight: number;
};

type AxisSuggestion = {
  key: string;
  nodeId: string;
  path: string;
  currentValue: number;
  defaultCandidates: string;
};

export function AutoTuneConfigModal({
  open,
  suggestions,
  onCancel,
  onRun,
}: {
  open: boolean;
  suggestions: AxisSuggestion[];
  onCancel: () => void;
  onRun: (cfg: AutoTuneConfig) => void;
}) {
  const [selected, setSelected] = useState<Record<string, string>>({});
  const [maxRounds, setMaxRounds] = useState("3");
  const [endWeight, setEndWeight] = useState("1.5");
  const [smoothnessWeight, setSmoothnessWeight] = useState("0");
  const prevOpenRef = useRef(false);

  useEffect(() => {
    if (open) {
      if (!prevOpenRef.current) {
        setSelected(Object.fromEntries(suggestions.map((s) => [s.key, s.defaultCandidates])));
      }
      prevOpenRef.current = true;
    } else {
      prevOpenRef.current = false;
    }
  }, [open, suggestions]);

  const parsed = useMemo(() => {
    const axes: AutoTuneAxisInput[] = [];
    for (const s of suggestions) {
      const raw = selected[s.key] ?? s.defaultCandidates;
      if (!raw || !raw.trim()) continue;
      const vals = raw
        .split(",")
        .map((x) => Number.parseFloat(x.trim()))
        .filter((x) => Number.isFinite(x));
      if (!vals.length) continue;
      axes.push({ nodeId: s.nodeId, path: s.path, values: vals });
    }
    return axes;
  }, [selected, suggestions]);

  const handleBackdropMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) onCancel();
    },
    [onCancel],
  );

  if (!open) return null;
  const node = (
    <div className="cr-modal-backdrop" style={{ zIndex: 10080 }} onMouseDown={handleBackdropMouseDown}>
      <div className="cr-modal cr-auto-tune-modal" role="dialog" aria-modal="true">
        <h2 className="cr-modal__title">Start auto-tuning</h2>
        <p className="cr-modal__hint">
          Choose candidate values per hyperparameter (comma-separated). Coordinate descent will tune one axis at a
          time.
        </p>
        <div className="cr-auto-tune-grid">
          {suggestions.map((s) => (
            <label key={s.key} className="cr-auto-tune-row">
              <span className="cr-auto-tune-row__label">
                {s.path}{" "}
                <em title={`node ${s.nodeId}`}>
                  (
                  {Number.isInteger(s.currentValue) || Math.abs(s.currentValue - Math.round(s.currentValue)) < 1e-6
                    ? String(Math.round(s.currentValue))
                    : s.currentValue.toPrecision(4)}
                  )
                </em>
              </span>
              <input
                className="cr-input"
                placeholder={s.defaultCandidates}
                value={selected[s.key] ?? s.defaultCandidates}
                onChange={(e) => setSelected((prev) => ({ ...prev, [s.key]: e.target.value }))}
              />
            </label>
          ))}
        </div>
        <div className="cr-auto-tune-fields">
          <label>
            rounds
            <input className="cr-input" value={maxRounds} onChange={(e) => setMaxRounds(e.target.value)} />
          </label>
          <label>
            end weight
            <input className="cr-input" value={endWeight} onChange={(e) => setEndWeight(e.target.value)} />
          </label>
          <label>
            smoothness
            <input
              className="cr-input"
              value={smoothnessWeight}
              onChange={(e) => setSmoothnessWeight(e.target.value)}
            />
          </label>
        </div>
        <div className="cr-modal__actions">
          <button type="button" className="cr-modal__btn" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            className="cr-modal__btn cr-modal__btn--primary"
            disabled={parsed.length === 0}
            onClick={() => {
              onRun({
                axes: parsed,
                maxRounds: Math.max(1, Math.round(Number.parseFloat(maxRounds) || 3)),
                endWeight: Math.max(1, Number.parseFloat(endWeight) || 1),
                smoothnessWeight: Math.max(0, Number.parseFloat(smoothnessWeight) || 0),
              });
            }}
          >
            Run coordinate descent
          </button>
        </div>
      </div>
    </div>
  );

  return createPortal(node, document.body);
}
