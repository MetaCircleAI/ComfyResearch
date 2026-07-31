import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { createPortal } from "react-dom";

export type RenameCombinedModelModalProps = {
  open: boolean;
  initialName: string;
  onCancel: () => void;
  onConfirm: (name: string) => void | Promise<void>;
};

export function RenameCombinedModelModal({
  open,
  initialName,
  onCancel,
  onConfirm,
}: RenameCombinedModelModalProps) {
  const titleId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setName(initialName);
    setBusy(false);
    const t = requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });
    return () => cancelAnimationFrame(t);
  }, [open, initialName]);

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

  const submit = useCallback(async () => {
    if (!open || busy) return;
    setBusy(true);
    try {
      await onConfirm(name.trim() || initialName);
    } finally {
      setBusy(false);
    }
  }, [busy, initialName, name, onConfirm, open]);

  if (!open) return null;

  const node = (
    <div
      className="cr-modal-backdrop"
      style={{ zIndex: 10030 }}
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
          Rename combined model
        </h2>
        <label className="cr-modal__label" htmlFor="cr-rename-combined-model-name">
          Model name
        </label>
        <input
          ref={inputRef}
          id="cr-rename-combined-model-name"
          className="cr-modal__input"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              void submit();
            }
          }}
          autoComplete="off"
          spellCheck={false}
        />
        <div className="cr-modal__actions">
          <button type="button" className="cr-modal__btn cr-modal__btn--ghost" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={() => void submit()} disabled={busy}>
            {busy ? "…" : "Confirm"}
          </button>
        </div>
      </div>
    </div>
  );

  return createPortal(node, document.body);
}
