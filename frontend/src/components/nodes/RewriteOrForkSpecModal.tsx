import { useCallback, useEffect, useId, useState } from "react";

export type RewriteOrForkSpecModalProps = {
  open: boolean;
  title: string;
  body: string;
  forkNameLabel?: string;
  defaultForkName: string;
  onRewrite: () => void;
  onFork: (newSpecName: string) => void;
  onCancel: () => void;
};

export function RewriteOrForkSpecModal({
  open,
  title,
  body,
  forkNameLabel = "New spec name",
  defaultForkName,
  onRewrite,
  onFork,
  onCancel,
}: RewriteOrForkSpecModalProps) {
  const titleId = useId();
  const [forkName, setForkName] = useState(defaultForkName);

  useEffect(() => {
    if (open) setForkName(defaultForkName);
  }, [open, defaultForkName]);

  const commitFork = useCallback(() => {
    const t = forkName.trim();
    if (!t) return;
    onFork(t);
  }, [forkName, onFork]);

  if (!open) return null;

  return (
    <div className="cr-modal-backdrop cr-modal-backdrop--spec" role="presentation" onMouseDown={onCancel}>
      <div
        className="cr-modal cr-modal--spec-fork"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id={titleId} className="cr-modal__title">
          {title}
        </h2>
        <p className="cr-modal__text">{body}</p>
        <label className="cr-modal__field">
          <span className="cr-modal__field-label">{forkNameLabel}</span>
          <input
            type="text"
            className="cr-input cr-modal__input"
            value={forkName}
            onChange={(e) => setForkName(e.target.value)}
            placeholder={defaultForkName}
            spellCheck={false}
            autoComplete="off"
          />
        </label>
        <div className="cr-modal__actions">
          <button type="button" className="cr-modal__btn" onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="cr-modal__btn" onClick={() => onRewrite()}>
            Rewrite this node
          </button>
          <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={commitFork}>
            Save as new name
          </button>
        </div>
      </div>
    </div>
  );
}
