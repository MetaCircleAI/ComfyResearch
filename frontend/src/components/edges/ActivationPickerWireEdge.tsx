import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from "@xyflow/react";
import { useCallback, useEffect, useRef, useState, type MouseEvent, type PointerEvent } from "react";
import { createPortal } from "react-dom";
import { useActivationPickerWireEditor } from "./activationPickerWireContext";

const GAUGE_CTX_MENU_W = 112;
const GAUGE_CTX_MENU_H = 36;

function clampGaugeCtxMenuPosition(clientX: number, clientY: number) {
  if (typeof window === "undefined") return { left: clientX, top: clientY };
  const pad = 6;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const left = Math.max(pad, Math.min(clientX, vw - GAUGE_CTX_MENU_W - pad));
  const top = Math.max(pad, Math.min(clientY, vh - GAUGE_CTX_MENU_H - pad));
  return { left, top };
}

/** Small dial / gauge (decorative); half-scale vs original art — center sits on wire midpoint via layout in parent. */
function PickerWireGaugeIcon() {
  return (
    <svg
      className="cr-apw-edge-gauge__svg"
      width="14"
      height="11"
      viewBox="0 0 28 22"
      aria-hidden
    >
      <path
        d="M 4 18 A 10 10 0 0 1 24 18"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <line x1="14" y1="18" x2="14" y2="10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <circle cx="14" cy="18" r="2" fill="currentColor" />
    </svg>
  );
}

/**
 * Bezier edge for the activation wire picker: input above gauge while editing; label after Enter (save) or blur
 * (click outside the input). Double-click label or gauge to edit again.
 */
export function ActivationPickerWireEdge(props: EdgeProps) {
  const {
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style,
    markerStart,
    markerEnd,
    interactionWidth,
    selected,
    data,
  } = props;

  const editor = useActivationPickerWireEditor();
  const savedLabel = String((data as { pickerSavedLabel?: string } | undefined)?.pickerSavedLabel ?? "");
  const pickable = (data as { pickerPickableEdge?: boolean } | undefined)?.pickerPickableEdge !== false;
  const pickerAfterModuleIndex = (data as { pickerAfterModuleIndex?: number } | undefined)?.pickerAfterModuleIndex;
  const nameInputRef = useRef<HTMLInputElement>(null);
  const gaugeCtxMenuRef = useRef<HTMLDivElement>(null);
  const [gaugeCtxMenu, setGaugeCtxMenu] = useState<{ left: number; top: number } | null>(null);

  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const showGadget = Boolean(pickable && selected && editor);
  /** Saved pick for this wire: show label + gauge even when the edge is not selected (reopen modal). */
  const showPassiveSaved = Boolean(pickable && !selected && savedLabel && editor);
  const focusNonce = editor?.wireNameFocusNonce ?? 0;
  const nameEditActive = Boolean(editor?.nameEditActive);
  const labelText =
    editor?.nameDraft.trim() || savedLabel || editor?.defaultNameDisplay || "";

  useEffect(() => {
    if (!selected || !nameEditActive || !nameInputRef.current) return;
    const t = requestAnimationFrame(() => {
      nameInputRef.current?.focus();
      nameInputRef.current?.select();
    });
    return () => cancelAnimationFrame(t);
  }, [focusNonce, selected, nameEditActive]);

  const onGadgetDoubleClick = useCallback(
    (e: MouseEvent) => {
      e.stopPropagation();
      e.preventDefault();
      editor?.openWireEditorForEdge(id);
    },
    [editor, id],
  );

  const onGadgetPointerDown = useCallback((e: PointerEvent) => {
    e.stopPropagation();
  }, []);

  useEffect(() => {
    if (!gaugeCtxMenu) return;
    const close = () => setGaugeCtxMenu(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    /** Capture phase runs before the menu button's target phase so we never close before Delete handles the event. */
    const onPointerDownCapture = (e: PointerEvent) => {
      const root = gaugeCtxMenuRef.current;
      if (root && e.target instanceof Node && root.contains(e.target)) return;
      close();
    };
    window.addEventListener("pointerdown", onPointerDownCapture, true);
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("scroll", close, true);
    return () => {
      window.removeEventListener("pointerdown", onPointerDownCapture, true);
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("scroll", close, true);
    };
  }, [gaugeCtxMenu]);

  const onGaugeContextMenu = useCallback(
    (e: MouseEvent) => {
      if (!editor?.removePickForEdge) return;
      e.preventDefault();
      e.stopPropagation();
      setGaugeCtxMenu(clampGaugeCtxMenuPosition(e.clientX, e.clientY));
    },
    [editor],
  );

  const onGaugeCtxDelete = useCallback(() => {
    editor?.removePickForEdge(id, pickerAfterModuleIndex ?? null);
    setGaugeCtxMenu(null);
  }, [editor, id, pickerAfterModuleIndex]);

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={style}
        markerStart={markerStart}
        markerEnd={markerEnd}
        interactionWidth={interactionWidth}
      />
      {showPassiveSaved ? (
        <EdgeLabelRenderer>
          <div
            className="cr-apw-edge-anchor cr-apw-edge-anchor--passive nodrag nopan"
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: "none",
            }}
            aria-hidden
          >
            <div className="cr-apw-edge-anchor__mid">
              <div className="cr-apw-edge-above">
                <div className="cr-apw-edge-name-label cr-apw-edge-name-label--floated">{savedLabel}</div>
              </div>
              <div className="cr-apw-edge-gadget-wrap cr-apw-edge-gadget-wrap--passive">
                <span className="cr-apw-edge-gauge">
                  <PickerWireGaugeIcon />
                </span>
              </div>
            </div>
          </div>
        </EdgeLabelRenderer>
      ) : null}
      {showGadget && editor ? (
        <EdgeLabelRenderer>
          <div
            className="cr-apw-edge-anchor nodrag nopan"
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: "all",
            }}
            onDoubleClick={(e) => e.stopPropagation()}
          >
            <div className="cr-apw-edge-anchor__mid">
              <div className="cr-apw-edge-above">
                {nameEditActive ? (
                  <>
                    <div
                      className="cr-apw-edge-name-slot nodrag nopan"
                      onPointerDown={(e) => e.stopPropagation()}
                      onClick={(e) => e.stopPropagation()}
                      onDoubleClick={(e) => e.stopPropagation()}
                    >
                      <input
                        ref={nameInputRef}
                        type="text"
                        className="cr-apw-edge-editor__input"
                        value={editor.nameDraft}
                        onChange={(e) => editor.setNameDraft(e.target.value)}
                        placeholder={savedLabel || "name…"}
                        title="Press Enter to save this name"
                        autoComplete="off"
                        aria-label="Activation name for this wire"
                        onBlur={() => editor.exitNameEditMode()}
                        onDoubleClick={(e) => e.stopPropagation()}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && editor.canCommitWirePick) {
                            e.preventDefault();
                            editor.commitWirePick();
                          }
                        }}
                      />
                    </div>
                    <div
                      className="cr-apw-edge-hit-bridge nodrag nopan"
                      aria-hidden
                      onDoubleClick={(e) => e.stopPropagation()}
                    />
                  </>
                ) : (
                  <div
                    className="cr-apw-edge-name-label cr-apw-edge-name-label--floated nodrag nopan"
                    title={labelText}
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => e.stopPropagation()}
                    onDoubleClick={(e) => {
                      e.stopPropagation();
                      editor.openWireEditorForEdge(id);
                    }}
                  >
                    {labelText}
                  </div>
                )}
              </div>
              <button
                type="button"
                className="cr-apw-edge-gadget-wrap nodrag nopan"
                title="Double-click to edit the name on this wire. Right-click or Ctrl+click for Delete."
                aria-label="Activation gauge on wire; double-click to edit name"
                onPointerDown={onGadgetPointerDown}
                onDoubleClick={onGadgetDoubleClick}
                onContextMenu={onGaugeContextMenu}
              >
                <span className="cr-apw-edge-gauge" aria-hidden>
                  <PickerWireGaugeIcon />
                </span>
              </button>
            </div>
          </div>
        </EdgeLabelRenderer>
      ) : null}
      {gaugeCtxMenu && editor
        ? createPortal(
            <div
              ref={gaugeCtxMenuRef}
              className="cr-apw-gauge-ctx-menu nodrag nopan"
              role="menu"
              style={{ position: "fixed", left: gaugeCtxMenu.left, top: gaugeCtxMenu.top, zIndex: 100020 }}
            >
              <button
                type="button"
                className="cr-apw-gauge-ctx-menu__item"
                role="menuitem"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onGaugeCtxDelete();
                }}
              >
                Delete
              </button>
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
