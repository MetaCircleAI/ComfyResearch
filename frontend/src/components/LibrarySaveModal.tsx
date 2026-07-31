import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { createPortal } from "react-dom";
import type { GraphDocument } from "../types/graph";
import type { GraphFileExportTier } from "../graph/graphFileExportTier";

export type LibrarySaveDraft = {
  kind: "workflow" | "template";
  document: GraphDocument;
  tier: GraphFileExportTier;
};

type LibrarySaveModalProps = {
  draft: LibrarySaveDraft | null;
  onDismiss: () => void;
  onConfirm: (trimmedName: string) => void;
};

const defaultNameForKind = (kind: "workflow" | "template") =>
  kind === "workflow" ? "Untitled workflow" : "Untitled template";

export function LibrarySaveModal({ draft, onDismiss, onConfirm }: LibrarySaveModalProps) {
  const titleId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");

  useEffect(() => {
    if (!draft) return;
    setName(defaultNameForKind(draft.kind));
    const t = requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });
    return () => cancelAnimationFrame(t);
  }, [draft]);

  const handleBackdropMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) onDismiss();
    },
    [onDismiss],
  );

  useEffect(() => {
    if (!draft) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onDismiss();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [draft, onDismiss]);

  const submit = useCallback(() => {
    if (!draft) return;
    const trimmed = name.trim();
    onConfirm(trimmed || defaultNameForKind(draft.kind));
  }, [draft, name, onConfirm]);

  if (!draft) return null;

  const heading =
    draft.kind === "workflow"
      ? "Save as workflow"
      : "Save as template";

  const node = (
    <div
      className="cr-modal-backdrop"
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
          {heading}
        </h2>
        <p className="cr-modal__hint">Choose a name for this entry in the library.</p>
        <label className="cr-modal__label" htmlFor="cr-library-save-name">
          Name
        </label>
        <input
          ref={inputRef}
          id="cr-library-save-name"
          className="cr-modal__input"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              submit();
            }
          }}
          autoComplete="off"
          spellCheck={false}
        />
        <div className="cr-modal__actions">
          <button type="button" className="cr-modal__btn cr-modal__btn--ghost" onClick={onDismiss}>
            Cancel
          </button>
          <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={submit}>
            Save
          </button>
        </div>
      </div>
    </div>
  );

  return createPortal(node, document.body);
}
