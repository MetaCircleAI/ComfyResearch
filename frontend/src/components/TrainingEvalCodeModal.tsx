import { useCallback, useEffect, useId } from "react";
import { createPortal } from "react-dom";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import type { MouseEvent as ReactMouseEvent } from "react";

export type TrainingEvalCodeModalProps = {
  open: boolean;
  code: string;
  title?: string;
  onClose: () => void;
};

/**
 * Full-screen overlay with VS Code Dark+–styled (Prism) syntax highlighting for training eval pseudo-code.
 */
export function TrainingEvalCodeModal({ open, code, title = "Training eval code", onClose }: TrainingEvalCodeModalProps) {
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const handleBackdropMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) onClose();
    },
    [onClose],
  );

  if (!open) return null;

  const node = (
    <div
      className="cr-modal-backdrop cr-modal-backdrop--code"
      role="presentation"
      onMouseDown={handleBackdropMouseDown}
    >
      <div
        className="cr-modal cr-modal--code-view"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="cr-modal--code-view__header">
          <h2 id={titleId} className="cr-modal__title">
            {title}
          </h2>
          <button
            type="button"
            className="cr-modal--code-view__close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>
        <p className="cr-modal__hint cr-modal--code-view__hint">
          Pseudo-Python produced from your tensor path (read-only). Tab size matches the source; theme mirrors VS Code
          Dark+.
        </p>
        <div className="cr-modal--code-view__editor">
          <SyntaxHighlighter
            language="python"
            style={vscDarkPlus}
            showLineNumbers
            showInlineLineNumbers={false}
            wrapLines={false}
            wrapLongLines={false}
            lineNumberContainerStyle={{
              float: "left",
              paddingRight: "12px",
              marginRight: "4px",
              borderRight: "1px solid rgba(255, 255, 255, 0.12)",
              userSelect: "none",
            }}
            lineNumberStyle={{
              minWidth: "2.75em",
              paddingRight: "0.35em",
              color: "rgb(133, 133, 133)",
              userSelect: "none",
              display: "block",
              textAlign: "right",
            }}
            codeTagProps={{
              className: "language-python cr-modal--code-view__pre",
              style: { tabSize: 4, MozTabSize: 4 },
            }}
            customStyle={{
              margin: 0,
              padding: "12px 14px",
              borderRadius: 8,
              fontSize: "13px",
              lineHeight: 1.55,
              maxHeight: "min(70vh, 720px)",
              overflow: "auto",
              fontFamily: 'ui-monospace, "SF Mono", "Cascadia Code", Menlo, Monaco, Consolas, monospace',
              background: "rgb(30, 30, 30)",
            }}
          >
            {code}
          </SyntaxHighlighter>
        </div>
        <div className="cr-modal__actions cr-modal--code-view__actions">
          <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );

  return createPortal(node, document.body);
}
