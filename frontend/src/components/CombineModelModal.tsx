import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { createPortal } from "react-dom";

export type CombineModelModalProps = {
  open: boolean;
  selectedCount: number;
  defaultName: string;
  onCancel: () => void;
  /** When true, saves a reusable copy under the Nodes → model list (stored as a workflow entry, not Templates). */
  onConfirm: (name: string, saveToLibrary: boolean) => void | Promise<void>;
};

export function CombineModelModal({
  open,
  selectedCount,
  defaultName,
  onCancel,
  onConfirm,
}: CombineModelModalProps) {
  const titleId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [saveToLibrary, setSaveToLibrary] = useState(true);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setName(defaultName);
    setSaveToLibrary(true);
    setBusy(false);
    const t = requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });
    return () => cancelAnimationFrame(t);
  }, [open, defaultName]);

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
      await onConfirm(name.trim() || defaultName, saveToLibrary);
    } finally {
      setBusy(false);
    }
  }, [busy, defaultName, name, onConfirm, open, saveToLibrary]);

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
          Combine into model node
        </h2>
        <p className="cr-modal__hint">
          Merges {selectedCount} selected nodes into one combined model marker. Edges that touch only nodes inside the
          selection are kept inside the combined model; other edges are removed from the canvas.
        </p>
        <label className="cr-modal__label" htmlFor="cr-combine-model-name">
          New model node name
        </label>
        <input
          ref={inputRef}
          id="cr-combine-model-name"
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
        <label className="cr-combine-modal__check">
          <input
            type="checkbox"
            checked={saveToLibrary}
            onChange={(e) => setSaveToLibrary(e.target.checked)}
          />
          <span>Add to Nodes library (model list)</span>
        </label>
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
