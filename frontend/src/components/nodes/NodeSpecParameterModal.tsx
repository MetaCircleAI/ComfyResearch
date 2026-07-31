import { useCallback, useEffect, useId, useMemo, useState, type MouseEvent as ReactMouseEvent } from "react";
import { createPortal } from "react-dom";
import { flattenNestedToNumbers, reshapeFlatToNested } from "../../graph/tensorNestedJson";
import type { WeightTensorPayload } from "./modelWeightTensorsDefaults";

type NodeSpecParameterModalProps = {
  open: boolean;
  tensors: Record<string, WeightTensorPayload>;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
  onSaveTensor: (name: string, payload: WeightTensorPayload) => void;
  onClose: () => void;
};

export function NodeSpecParameterModal({
  open,
  tensors,
  loading,
  error,
  onRefresh,
  onSaveTensor,
  onClose,
}: NodeSpecParameterModalProps) {
  const titleId = useId().replace(/:/g, "");
  const names = useMemo(() => Object.keys(tensors).sort(), [tensors]);
  const normalizedNames = useMemo(
    () =>
      Object.fromEntries(
        names.map((name) => [
          name,
          name
            .split(".")
            .flatMap((segment) => {
              if (segment.endsWith("_weight") && segment.length > "_weight".length) {
                return [segment.slice(0, -"_weight".length), "weight"];
              }
              if (segment.endsWith("_bias") && segment.length > "_bias".length) {
                return [segment.slice(0, -"_bias".length), "bias"];
              }
              return [segment];
            })
            .join("."),
        ]),
      ),
    [names],
  );
  const displayNames = useMemo(() => {
    if (names.length <= 1) {
      return Object.fromEntries(names.map((name) => [name, normalizedNames[name] ?? name]));
    }
    const segments = new Map<string, string[]>();
    names.forEach((name) => segments.set(name, (normalizedNames[name] ?? name).split(".")));
    const maxDepth = Math.max(...names.map((name) => segments.get(name)?.length ?? 1), 1);
    const chosen = new Map<string, string>();
    for (let depth = 1; depth <= maxDepth; depth += 1) {
      const suffixes = new Map<string, string[]>();
      names.forEach((name) => {
        if (chosen.has(name)) return;
        const parts = segments.get(name) ?? [name];
        const suffix = parts.slice(-Math.min(depth, parts.length)).join(".");
        const bucket = suffixes.get(suffix);
        if (bucket) bucket.push(name);
        else suffixes.set(suffix, [name]);
      });
      suffixes.forEach((bucket, suffix) => {
        if (bucket.length !== 1) return;
        chosen.set(bucket[0], suffix);
      });
      if (chosen.size === names.length) break;
    }
    return Object.fromEntries(names.map((name) => [name, chosen.get(name) ?? normalizedNames[name] ?? name]));
  }, [names, normalizedNames]);
  const [activeName, setActiveName] = useState<string>("");
  const [draftJson, setDraftJson] = useState("");
  const [draftErr, setDraftErr] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setActiveName("");
      setDraftErr(null);
      return;
    }
    if (!activeName || !(activeName in tensors)) {
      setActiveName(names[0] ?? "");
    }
  }, [activeName, names, open, tensors]);

  useEffect(() => {
    if (!open || !activeName) return;
    const t = tensors[activeName];
    if (!t) return;
    setDraftJson(JSON.stringify(reshapeFlatToNested(t.values, t.shape), null, 2));
    setDraftErr(null);
  }, [activeName, open, tensors]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const onBackdrop = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) onClose();
    },
    [onClose],
  );

  const activeTensor = activeName ? tensors[activeName] ?? null : null;

  const handleSaveTensor = useCallback(() => {
    if (!activeName || !activeTensor) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(draftJson);
    } catch (e) {
      setDraftErr(e instanceof Error ? e.message : String(e));
      return;
    }
    const values = flattenNestedToNumbers(parsed);
    if (values.some((x) => !Number.isFinite(x))) {
      setDraftErr("values must contain only finite numbers.");
      return;
    }
    const shape = activeTensor.shape;
    const expected = shape.length ? shape.reduce((acc, n) => acc * n, 1) : 1;
    if (expected !== values.length) {
      setDraftErr(`Expected ${expected} values for this tensor shape, but got ${values.length}.`);
      return;
    }
    const normalized = reshapeFlatToNested(values, shape);
    const normalizedFlat = flattenNestedToNumbers(normalized);
    setDraftErr(null);
    onSaveTensor(activeName, { shape, values: normalizedFlat });
  }, [activeName, activeTensor, draftJson, onSaveTensor]);

  if (!open) return null;

  return createPortal(
    <div className="cr-modal-backdrop" style={{ zIndex: 10028 }} role="presentation" onMouseDown={onBackdrop}>
      <div
        className="cr-modal cr-modal--tensor-constant cr-modal--model-params"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="cr-modal--tensor-constant__header">
          <h2 id={titleId} className="cr-modal__title">
            Model · parameters
          </h2>
          <button type="button" className="cr-modal--code-view__close nodrag nopan" aria-label="Close" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="cr-modal--tensor-constant__body nodrag nopan">
          <div className="cr-model-params-tabs">
            {names.map((name) => (
              <button
                key={name}
                type="button"
                className={`cr-model-params-tab${name === activeName ? " cr-model-params-tab--active" : ""}`}
                onClick={() => setActiveName(name)}
              >
                {displayNames[name] ?? name}
              </button>
            ))}
          </div>

          {loading && names.length === 0 ? <p className="cr-activation-scan-msg">Loading parameter tensors…</p> : null}
          {error ? <p className="cr-activation-scan-msg">{error}</p> : null}
          {!loading && !error && names.length === 0 ? (
            <p className="cr-activation-scan-msg">No parameter tensors found for this model node.</p>
          ) : null}

          <div className="cr-tensor-constant-preview-block">
            <h3 className="cr-tensor-constant-preview-title">Values</h3>
            <textarea
              className="cr-input cr-modal--model-params__textarea"
              value={draftJson}
              onChange={(e) => setDraftJson(e.target.value)}
              spellCheck={false}
              aria-label="Editable tensor JSON for selected parameter"
            />
          </div>

          {draftErr ? <p className="cr-trainer-train-err">{draftErr}</p> : null}
          <div className="cr-modal--model-params__actions">
            <button type="button" className="cr-modal__btn" onClick={onRefresh}>
              Refresh
            </button>
            <button type="button" className="cr-modal__btn cr-modal__btn--ghost" onClick={onClose}>
              Cancel
            </button>
            <button
              type="button"
              className="cr-modal__btn cr-modal__btn--primary"
              onClick={handleSaveTensor}
              disabled={!activeTensor}
            >
              Save tensor
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
