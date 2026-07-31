import { createPortal } from "react-dom";

/**
 * Shown while an async delete (or similar) runs so the UI does not feel frozen.
 * Indeterminate bar only — server does not expose byte-level progress.
 */
export function DeletingBusyOverlay({ open, message = "Deleting…" }: { open: boolean; message?: string }) {
  if (!open) return null;
  return createPortal(
    <div className="cr-deleting-overlay" role="status" aria-live="polite" aria-busy="true">
      <div className="cr-deleting-overlay__panel">
        <p className="cr-deleting-overlay__text">{message}</p>
        <div className="cr-deleting-overlay__track" aria-hidden>
          <div className="cr-deleting-overlay__bar" />
        </div>
      </div>
    </div>,
    document.body,
  );
}
