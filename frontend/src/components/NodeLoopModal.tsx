import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { createPortal } from "react-dom";

export type NodeLoopModalProps = {
  open: boolean;
  initialCount: number;
  /** When loop count is 2+, whether weights are tied across repeats (default false). */
  initialShareParams: boolean;
  onCancel: () => void;
  onConfirm: (loopCount: number, loopShareParams: boolean) => void | Promise<void>;
};

function parseLoopCount(raw: string): number | null {
  const t = raw.trim();
  if (!t) return null;
  const n = Math.floor(Number(t));
  if (!Number.isFinite(n)) return null;
  return n;
}

export function NodeLoopModal({
  open,
  initialCount,
  initialShareParams,
  onCancel,
  onConfirm,
}: NodeLoopModalProps) {
  const titleId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [value, setValue] = useState(String(Math.max(2, initialCount)));
  const [shareParams, setShareParams] = useState(initialShareParams);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setValue(String(Math.max(2, initialCount)));
    setShareParams(initialShareParams);
    setBusy(false);
    const t = requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });
    return () => cancelAnimationFrame(t);
  }, [open, initialCount, initialShareParams]);

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
    const n = parseLoopCount(value);
    if (n === null) return;
    setBusy(true);
    try {
      await onConfirm(n, n >= 2 && shareParams);
    } finally {
      setBusy(false);
    }
  }, [busy, onConfirm, open, shareParams, value]);

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
          Loop node
        </h2>
        <p className="cr-modal__hint">
          How many times should this node run? A badge appears above the node when the count is 2 or more. Use 1 to clear
          the loop marker. For training, the block is stacked that many times; input and output sizes must match when the
          count is 2 or more.
        </p>
        <label className="cr-modal__label" htmlFor="cr-node-loop-count">
          Times to loop
        </label>
        <input
          ref={inputRef}
          id="cr-node-loop-count"
          className="cr-modal__input"
          type="number"
          inputMode="numeric"
          min={1}
          step={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              void submit();
            }
          }}
          autoComplete="off"
          spellCheck={false}
        />
        <label className="cr-modal__checkbox-row">
          <input
            type="checkbox"
            checked={shareParams}
            onChange={(e) => setShareParams(e.target.checked)}
          />
          <span>Share parameters across loops (same weights each time; default is separate copies)</span>
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
