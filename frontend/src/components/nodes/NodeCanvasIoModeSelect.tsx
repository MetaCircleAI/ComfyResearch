import { useCallback, useId, useRef, useState, type ReactNode } from "react";
import { useReactFlow } from "@xyflow/react";
import { useDismissOnOutsidePointer } from "../../hooks/useDismissOnOutsidePointer";
import { readNodeCanvasIoMode, type NodeCanvasIoMode } from "../../graph/nodeCanvasIoMode";
import { readNodeCanvasLevelMode, type NodeCanvasLevelMode } from "../../graph/nodeCanvasLevelMode";

const IO_OPTIONS: { value: NodeCanvasIoMode; label: string }[] = [
  { value: "model", label: "mode: model" },
  { value: "input-output", label: "mode: input-output" },
];

const LEVEL_OPTIONS: { value: NodeCanvasLevelMode; label: string }[] = [
  { value: "high", label: "level: high" },
  { value: "low", label: "level: low" },
];

export function NodeCanvasIoModeSelect({
  id,
  data,
  onAfterChange,
}: {
  id: string;
  /** Merged or raw node `data` (unknown fields like `ioMode` are preserved on update). */
  data: Record<string, unknown>;
  onAfterChange?: (next: NodeCanvasIoMode, prev: NodeCanvasIoMode) => void;
}) {
  const { setNodes } = useReactFlow();
  const mode = readNodeCanvasIoMode(data);
  const wrapRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const panelUid = useId().replace(/:/g, "");
  const panelId = `cr-node-io-mode-panel-${panelUid}`;

  useDismissOnOutsidePointer(open, () => setOpen(false), wrapRef);

  const commit = useCallback(
    (next: NodeCanvasIoMode) => {
      const prev = readNodeCanvasIoMode(data);
      if (next === prev) return;
      setNodes((nds) =>
        nds.map((n) =>
          n.id === id ? { ...n, data: { ...(n.data as Record<string, unknown>), ioMode: next } } : n,
        ),
      );
      onAfterChange?.(next, prev);
    },
    [data, id, onAfterChange, setNodes],
  );

  const onPick = useCallback(
    (next: NodeCanvasIoMode) => {
      commit(next);
      setOpen(false);
      triggerRef.current?.focus();
    },
    [commit],
  );

  const summary = IO_OPTIONS.find((o) => o.value === mode)?.label ?? "mode: model";

  return (
    <div ref={wrapRef} className="cr-discrete-multi-dd cr-node-io-mode-dd nodrag nopan">
      <button
        ref={triggerRef}
        type="button"
        className="cr-select cr-discrete-multi-dd__btn nodrag nopan"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        aria-label="Canvas I/O mode"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="cr-discrete-multi-dd__btn-text">{summary}</span>
        <span className="cr-discrete-multi-dd__caret" aria-hidden>
          ▾
        </span>
      </button>
      {open ? (
        <div
          id={panelId}
          role="listbox"
          aria-label="Canvas I/O mode"
          className="cr-discrete-multi-dd__panel nodrag nopan"
          onMouseDown={(e) => e.stopPropagation()}
          onWheel={(e) => e.stopPropagation()}
        >
          {IO_OPTIONS.map((o) => {
            const rowId = `cr-iomode-${id}-${o.value}`;
            return (
              <label key={o.value} className="cr-discrete-multi-dd__row" htmlFor={rowId}>
                <input
                  id={rowId}
                  type="checkbox"
                  className="nodrag nopan"
                  checked={mode === o.value}
                  onChange={() => {
                    if (mode !== o.value) onPick(o.value);
                  }}
                />
                <span>{o.label}</span>
              </label>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

export function NodeCanvasLevelModeSelect({
  id,
  data,
}: {
  id: string;
  data: Record<string, unknown>;
}) {
  const { setNodes } = useReactFlow();
  const mode = readNodeCanvasLevelMode(data);
  const wrapRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const panelUid = useId().replace(/:/g, "");
  const panelId = `cr-node-level-mode-panel-${panelUid}`;

  useDismissOnOutsidePointer(open, () => setOpen(false), wrapRef);

  const commit = useCallback(
    (next: NodeCanvasLevelMode) => {
      const prev = readNodeCanvasLevelMode(data);
      if (next === prev) return;
      setNodes((nds) =>
        nds.map((n) =>
          n.id === id ? { ...n, data: { ...(n.data as Record<string, unknown>), levelMode: next } } : n,
        ),
      );
    },
    [data, id, setNodes],
  );

  const onPick = useCallback(
    (next: NodeCanvasLevelMode) => {
      commit(next);
      setOpen(false);
      triggerRef.current?.focus();
    },
    [commit],
  );

  const summary = LEVEL_OPTIONS.find((o) => o.value === mode)?.label ?? "level: high";

  return (
    <div ref={wrapRef} className="cr-discrete-multi-dd cr-node-level-mode-dd nodrag nopan">
      <button
        ref={triggerRef}
        type="button"
        className="cr-select cr-discrete-multi-dd__btn nodrag nopan"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        aria-label="Canvas level mode"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="cr-discrete-multi-dd__btn-text">{summary}</span>
        <span className="cr-discrete-multi-dd__caret" aria-hidden>
          ▾
        </span>
      </button>
      {open ? (
        <div
          id={panelId}
          role="listbox"
          aria-label="Canvas level mode"
          className="cr-discrete-multi-dd__panel nodrag nopan"
          onMouseDown={(e) => e.stopPropagation()}
          onWheel={(e) => e.stopPropagation()}
        >
          {LEVEL_OPTIONS.map((o) => {
            const rowId = `cr-levelmode-${id}-${o.value}`;
            return (
              <label key={o.value} className="cr-discrete-multi-dd__row" htmlFor={rowId}>
                <input
                  id={rowId}
                  type="checkbox"
                  className="nodrag nopan"
                  checked={mode === o.value}
                  onChange={() => {
                    if (mode !== o.value) onPick(o.value);
                  }}
                />
                <span>{o.label}</span>
              </label>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

/** Standard header: title row (with optional actions) + I/O/level row below. Optional subtitle below. */
export function NodeHeaderWithIoMode({
  id,
  data,
  children,
  subtitle,
  headerActions,
  onIoModeChange,
}: {
  id: string;
  data: Record<string, unknown>;
  children: ReactNode;
  subtitle?: ReactNode;
  headerActions?: ReactNode;
  onIoModeChange?: (next: NodeCanvasIoMode, prev: NodeCanvasIoMode) => void;
}) {
  return (
    <div className="cr-node__header">
      <div className="cr-node__header-row cr-node__header-row--title-actions">
        <div className="cr-node__header-title">{children}</div>
        {headerActions ? <div className="cr-node__header-actions">{headerActions}</div> : null}
      </div>
      <div className="cr-node__header-row cr-node__header-row--io-mode">
        <div className="cr-node__header-modes">
          <NodeCanvasIoModeSelect id={id} data={data} onAfterChange={onIoModeChange} />
          <NodeCanvasLevelModeSelect id={id} data={data} />
        </div>
      </div>
      {subtitle}
    </div>
  );
}
