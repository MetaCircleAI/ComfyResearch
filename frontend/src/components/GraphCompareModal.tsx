import { useCallback, useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import type { GraphDocument } from "../types/graph";
import { buildIdPreservingEditPath } from "../graph/graphEditPath";

export type GraphCompareTarget = {
  key: string;
  projectTitle: string;
  canvasTitle: string;
  document: GraphDocument;
};

type GraphCompareModalProps = {
  open: boolean;
  onClose: () => void;
  sourceLabel: string;
  sourceDocument: GraphDocument;
  targets: GraphCompareTarget[];
};

export function GraphCompareModal({
  open,
  onClose,
  sourceLabel,
  sourceDocument,
  targets,
}: GraphCompareModalProps) {
  const [selectedKey, setSelectedKey] = useState<string>("");

  useEffect(() => {
    if (!open) return;
    if (targets.length === 1) setSelectedKey(targets[0]!.key);
    else setSelectedKey("");
  }, [open, targets]);

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

  const selected = useMemo(
    () => targets.find((t) => t.key === selectedKey) ?? null,
    [selectedKey, targets],
  );

  const path = useMemo(() => {
    if (!selected) return null;
    return buildIdPreservingEditPath(sourceDocument, selected.document);
  }, [selected, sourceDocument]);

  const copyPathJson = useCallback(() => {
    if (!path) return;
    const payload = {
      sourceLabel,
      targetLabel: selected ? `${selected.projectTitle} / ${selected.canvasTitle}` : "",
      distance: path.distance,
      summaries: path.summaries,
      edits: path.edits,
      snapshots: path.snapshots,
    };
    void navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
  }, [path, selected, sourceLabel]);

  if (!open || typeof document === "undefined") return null;

  return createPortal(
    <div
      className="cr-modal-backdrop"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="cr-modal cr-graph-compare-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="cr-graph-compare-title"
      >
        <h2 id="cr-graph-compare-title" className="cr-modal__title">
          Compare graphs (edit path)
        </h2>
        <p className="cr-modal__text cr-graph-compare__intro">
          <strong>From:</strong> {sourceLabel}
          <br />
          Nodes are matched by <code>id</code>; edges by endpoints and handles. Each listed step is one atomic change; the
          sequence interpolates from the current graph to the target. Pan/zoom (viewport) is{" "}
          <strong>not</strong> counted as an edit. Unrelated graphs may show a large path (remove all, add all).
        </p>
        {targets.length === 0 ? (
          <p className="cr-modal__hint">Add another project to compare against.</p>
        ) : (
          <>
            <label className="cr-graph-compare__label" htmlFor="cr-graph-compare-select">
              Compare to
            </label>
            <select
              id="cr-graph-compare-select"
              className="cr-graph-compare__select"
              value={selectedKey}
              onChange={(e) => setSelectedKey(e.target.value)}
            >
              <option value="">Choose a project…</option>
              {targets.map((t) => (
                <option key={t.key} value={t.key}>
                  {t.projectTitle} — {t.canvasTitle}
                </option>
              ))}
            </select>
            {path && selected ? (
              <div className="cr-graph-compare__result">
                <p className="cr-graph-compare__distance">
                  <strong>Edit distance:</strong> {path.distance}{" "}
                  <span className="cr-graph-compare__muted">
                    ({path.snapshots.length} graphs along the path, including endpoints)
                  </span>
                </p>
                <ol className="cr-graph-compare__steps" start={1}>
                  {path.summaries.map((line, i) => (
                    <li key={i}>{line}</li>
                  ))}
                </ol>
                <div className="cr-graph-compare__actions">
                  <button type="button" className="cr-modal__btn cr-modal__btn--primary" onClick={copyPathJson}>
                    Copy path JSON
                  </button>
                </div>
              </div>
            ) : null}
          </>
        )}
        <div className="cr-modal__actions">
          <button type="button" className="cr-modal__btn cr-modal__btn--ghost" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
