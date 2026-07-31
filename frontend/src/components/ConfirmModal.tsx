import { useCallback, useEffect, useId, type MouseEvent as ReactMouseEvent } from "react";
import { createPortal } from "react-dom";

export type ConfirmModalProps = {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  /** Higher z-index when stacking over other modals (default 10032). */
  zIndex?: number;
  onCancel: () => void;
  onConfirm: () => void;
};

export function ConfirmModal({
  open,
  title,
  message,
  confirmLabel = "OK",
  cancelLabel = "Cancel",
  zIndex = 10032,
  onCancel,
  onConfirm,
}: ConfirmModalProps) {
  const titleId = useId();

  const handleBackdropMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) onCancel();
    },
    [onCancel],
  );

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onCancel();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  if (!open) return null;

  const node = (
    <div
      className="cr-modal-backdrop"
      style={{ zIndex }}
      role="presentation"
      onMouseDown={handleBackdropMouseDown}
    >
      <div
        className="cr-modal"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={`${titleId}-desc`}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id={titleId} className="cr-modal__title">
          {title}
        </h2>
        <p id={`${titleId}-desc`} className="cr-modal__hint">
          {message}
        </p>
        <div className="cr-modal__actions">
          <button type="button" className="cr-modal__btn cr-modal__btn--ghost" onClick={onCancel}>
            {cancelLabel}
          </button>
          <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );

  return createPortal(node, document.body);
}
