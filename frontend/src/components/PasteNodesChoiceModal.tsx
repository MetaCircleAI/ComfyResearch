import { useCallback, useEffect, useId, type MouseEvent as ReactMouseEvent } from "react";
import { createPortal } from "react-dom";

export type PasteNodesChoiceModalProps = {
  open: boolean;
  onCancel: () => void;
  /** `true` = keep copied display titles and data; `false` = assign next `instanceTitle` indices per type. */
  onConfirm: (shareParams: boolean) => void;
};

export function PasteNodesChoiceModal({ open, onCancel, onConfirm }: PasteNodesChoiceModalProps) {
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
      style={{ zIndex: 10031 }}
      role="presentation"
      onMouseDown={handleBackdropMouseDown}
    >
      <div
        className="cr-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id={titleId} className="cr-modal__title">
          Paste copied nodes
        </h2>
        <p className="cr-modal__hint">
          Pasted nodes always get new graph IDs. Choose whether to keep the same display titles and parameter values as
          the copy, or assign the next display index for each node type (e.g. Linear Layer 1).
        </p>
        <div className="cr-modal__actions cr-modal__actions--paste-choice">
          <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={() => onConfirm(true)}>
            share params (same id)
          </button>
          <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={() => onConfirm(false)}>
            non-share params (new id)
          </button>
          <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={onCancel}>
            cancel
          </button>
        </div>
      </div>
    </div>
  );

  return createPortal(node, document.body);
}
