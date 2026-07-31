import { useCallback, useEffect, useId, type MouseEvent as ReactMouseEvent } from "react";
import { createPortal } from "react-dom";

type TensorReaderModalProps = {
  open: boolean;
  shapeText: string;
  sourceSummary: string | null;
  jsonText: string;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
  onClose: () => void;
};

export function TensorReaderModal({
  open,
  shapeText,
  sourceSummary,
  jsonText,
  loading,
  error,
  onRefresh,
  onClose,
}: TensorReaderModalProps) {
  const titleId = useId().replace(/:/g, "");

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
            Tensor reader
          </h2>
          <button type="button" className="cr-modal--code-view__close nodrag nopan" aria-label="Close" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="cr-modal--tensor-constant__body nodrag nopan">
          <p className="cr-activation-scan-msg">
            shape: <strong>{shapeText}</strong>
            {sourceSummary ? ` · ${sourceSummary}` : ""}
          </p>
          {loading ? <p className="cr-activation-scan-msg">Loading tensor from server…</p> : null}
          {error ? <p className="cr-activation-scan-msg">{error}</p> : null}

          <div className="cr-tensor-constant-preview-block">
            <h3 className="cr-tensor-constant-preview-title">Values (read-only)</h3>
            <textarea
              className="cr-input cr-modal--model-params__textarea"
              value={jsonText}
              readOnly
              spellCheck={false}
              aria-label="Tensor values as nested JSON"
            />
          </div>

          <div className="cr-modal--model-params__actions">
            <button type="button" className="cr-modal__btn" onClick={onRefresh} disabled={loading}>
              Refresh
            </button>
            <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
