import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useNodeId,
  useReactFlow,
  useUpdateNodeInternals,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { createPortal } from "react-dom";
import {
  buildActivationWirePickerGraph,
  sanitizeTensorKey,
  type ActivationPickerWireEdge,
} from "../graph/activationWirePickerGraph";
import type { ActivationWireResolvedModel } from "../graph/resolveActivationWireModel";
import type { ActivationWirePick } from "./nodes/activationDefaults";
import { ActivationPickerWireEdge as ActivationPickerWireEdgeView } from "./edges/ActivationPickerWireEdge";
import {
  ActivationPickerWireEditorContext,
  type ActivationPickerWireEditorContextValue,
} from "./edges/activationPickerWireContext";
import { CombinedModelIoStrip } from "./nodes/CombinedModelIoStrip";
import { withActivationPickerNodeResize } from "./nodes/activationPickerNodeResize";
import { CombinedSubgraphIoEdge } from "./edges/CombinedSubgraphIoEdge";
import { COMBINED_SUBGRAPH_IO_EDGE_TYPE } from "../graph/layerStripHandles";

const pickerEdgeTypes = {
  activation_picker_wire: ActivationPickerWireEdgeView,
  [COMBINED_SUBGRAPH_IO_EDGE_TYPE]: CombinedSubgraphIoEdge,
};

function ActivationPickerFrameInner({ data, width, height }: NodeProps) {
  const nodeId = useNodeId();
  const updateNodeInternals = useUpdateNodeInternals();
  const d = (data ?? {}) as { title?: string; loopCount?: number };
  const title = String(d.title ?? "");
  const loopCount = typeof d.loopCount === "number" && d.loopCount >= 2 ? Math.floor(d.loopCount) : 0;
  const w = typeof width === "number" && width > 0 ? width : 320;
  const h = typeof height === "number" && height > 0 ? height : 200;

  useLayoutEffect(() => {
    if (nodeId) updateNodeInternals(nodeId);
  }, [nodeId, updateNodeInternals, w, h]);

  return (
    <div className="cr-activation-picker-frame" style={{ width: w, height: h, position: "relative" }}>
      <div className="cr-activation-picker-frame__chrome">
        {loopCount >= 2 ? (
          <div className="cr-node-loop-badge cr-activation-picker-frame__loop" aria-hidden>
            <span className="cr-node-loop-badge__icon" title={`Loop ×${loopCount}`}>
              ⟲
            </span>
            <span className="cr-node-loop-badge__times">×{loopCount}</span>
          </div>
        ) : null}
        <div className="cr-activation-picker-frame__title" title={title}>
          {title}
        </div>
      </div>
      <div
        className="cr-activation-picker-frame__io nodrag nopan"
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: "50%",
          transform: "translateY(-50%)",
        }}
      >
        <CombinedModelIoStrip isConnectable={false} />
      </div>
    </div>
  );
}

function ActivationPickerBlockInner({ data, width, height }: NodeProps) {
  const label = String((data as { shortLabel?: string }).shortLabel ?? "");
  const w = typeof width === "number" && width > 0 ? width : 96;
  const h = typeof height === "number" && height > 0 ? height : 30;
  return (
    <div
      className="cr-activation-picker-block"
      style={{ width: w, height: h, minWidth: w, minHeight: h, maxWidth: w, maxHeight: h }}
    >
      <Handle
        type="target"
        position={Position.Left}
        id="in"
        className="cr-activation-picker-handle react-flow__handle-left"
      />
      <span className="cr-activation-picker-block__label">{label}</span>
      <Handle
        type="source"
        position={Position.Right}
        id="out"
        className="cr-activation-picker-handle react-flow__handle-right"
      />
    </div>
  );
}

const pickerNodeTypes = {
  activationPickerFrame: withActivationPickerNodeResize(ActivationPickerFrameInner),
  activationPickerBlock: withActivationPickerNodeResize(ActivationPickerBlockInner),
};

function FitViewOnReady({ nodeCount }: { nodeCount: number }) {
  const rf = useReactFlow();
  useEffect(() => {
    let inner: number | undefined;
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => {
        void rf.fitView({
          padding: 0.22,
          duration: 220,
          maxZoom: 1.2,
          minZoom: 0.32,
        });
      });
    });
    return () => {
      cancelAnimationFrame(outer);
      if (inner !== undefined) cancelAnimationFrame(inner);
    };
  }, [rf, nodeCount]);
  return null;
}

function PickerFlow({
  flowNodes,
  flowEdges,
  wires,
  picks,
  selectedEdgeId,
  graphStructureKey,
  onSelectWireEdge,
  wireEditor,
  flowHeightPx,
}: {
  flowNodes: Node[];
  flowEdges: Edge[];
  wires: ActivationPickerWireEdge[];
  picks: ActivationWirePick[];
  selectedEdgeId: string | null;
  /** When this changes (different chain), reset node positions from the builder. */
  graphStructureKey: string;
  onSelectWireEdge: (edgeId: string) => void;
  wireEditor: ActivationPickerWireEditorContextValue;
  /** When set, height matches 0.8× main canvas (see ``useWorkbenchCanvasPickerLayout``). */
  flowHeightPx: number | null;
}) {
  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges);
  const lastStructureKey = useRef<string>("");

  const reconcilePickerEdges = useCallback(
    (eds: Edge[]) =>
      eds.map((e) => {
        const isBridge = e.type === COMBINED_SUBGRAPH_IO_EDGE_TYPE;
        const w = wires.find((x) => x.edgeId === e.id);
        const pickerSavedLabel = w
          ? (picks.find((p) => p.afterModuleIndex === w.afterModuleIndex)?.label ?? "")
          : "";
        const explicitNoPick = (e.data as { pickerPickableEdge?: boolean } | undefined)?.pickerPickableEdge === false;
        const pickerPickableEdge = Boolean(w) && !explicitNoPick;
        const selected = !isBridge && e.id === selectedEdgeId;
        return {
          ...e,
          type: (isBridge ? COMBINED_SUBGRAPH_IO_EDGE_TYPE : "activation_picker_wire") as Edge["type"],
          selectable: false,
          focusable: false,
          data: {
            ...(e.data && typeof e.data === "object" ? e.data : {}),
            pickerPickableEdge,
            pickerSavedLabel,
            pickerAfterModuleIndex: w?.afterModuleIndex,
          },
          selected,
          style: isBridge
            ? { ...(e.style ?? {}), stroke: "var(--cr-edge-muted, #6b7280)", strokeWidth: 2 }
            : {
                ...(e.style ?? {}),
                stroke: selected ? "var(--cr-accent-tensor, #6cf)" : "var(--cr-edge-muted, #6b7280)",
                strokeWidth: selected ? 3 : 2,
              },
        };
      }),
    [wires, picks, selectedEdgeId],
  );

  useEffect(() => {
    if (lastStructureKey.current === graphStructureKey) return;
    lastStructureKey.current = graphStructureKey;
    setNodes(flowNodes);
  }, [graphStructureKey, flowNodes, setNodes]);

  useEffect(() => {
    setEdges(reconcilePickerEdges(flowEdges));
  }, [flowEdges, reconcilePickerEdges, setEdges]);

  const onEdgesChangeWrapped = useCallback(
    (changes: EdgeChange[]) => {
      onEdgesChange(changes);
      setEdges((eds) => reconcilePickerEdges(eds));
    },
    [onEdgesChange, setEdges, reconcilePickerEdges],
  );

  const onEdgeClickEv = useCallback(
    (_: ReactMouseEvent, e: Edge) => {
      if ((e.data as { pickerPickableEdge?: boolean } | undefined)?.pickerPickableEdge === false) return;
      if (!wires.some((w) => w.edgeId === e.id)) return;
      onSelectWireEdge(e.id);
    },
    [onSelectWireEdge, wires],
  );

  return (
    <ActivationPickerWireEditorContext.Provider value={wireEditor}>
      <div
        className="cr-activation-picker-flow"
        style={flowHeightPx != null ? { height: `${flowHeightPx}px` } : undefined}
      >
        <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChangeWrapped}
        nodeTypes={pickerNodeTypes}
        edgeTypes={pickerEdgeTypes}
        defaultEdgeOptions={{ type: "activation_picker_wire", selectable: false, focusable: false }}
        nodesDraggable
        nodesConnectable={false}
        elementsSelectable
        panOnScroll
        panOnDrag
        zoomOnScroll
        zoomOnDoubleClick={false}
        onEdgeClick={onEdgeClickEv}
        proOptions={{ hideAttribution: true }}
        minZoom={0.4}
        maxZoom={1.4}
      >
        <Background gap={16} />
        <Controls showInteractive={false} />
        <FitViewOnReady nodeCount={flowNodes.length} />
      </ReactFlow>
      </div>
    </ActivationPickerWireEditorContext.Provider>
  );
}

export type ActivationWirePickerModalProps = {
  open: boolean;
  onClose: () => void;
  onSave: (picks: ActivationWirePick[]) => void;
  resolved: ActivationWireResolvedModel;
  nodes: Node[];
  edges: Edge[];
  initialPicks: ActivationWirePick[];
};

function newPickId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `pick-${Math.random().toString(36).slice(2, 12)}`;
}

const PICKER_FLOW_MIN_HEIGHT_PX = 280;
/** Mini-flow height as a fraction of the live main canvas (``.cr-canvas-wrap``) height. */
const PICKER_FLOW_HEIGHT_VS_MAIN_CANVAS = 0.8;

/** Match ``.cr-canvas-wrap``: modal width = canvas width; picker flow height = 0.8× canvas height. */
function useWorkbenchCanvasPickerLayout(open: boolean): { widthPx: number | null; flowHeightPx: number | null } {
  const [layout, setLayout] = useState<{ widthPx: number | null; flowHeightPx: number | null }>({
    widthPx: null,
    flowHeightPx: null,
  });
  useLayoutEffect(() => {
    if (!open) {
      setLayout({ widthPx: null, flowHeightPx: null });
      return;
    }
    const el = document.querySelector(".cr-canvas-wrap");
    if (!el || !(el instanceof HTMLElement)) {
      setLayout({ widthPx: null, flowHeightPx: null });
      return;
    }
    const apply = () => {
      const r = el.getBoundingClientRect();
      const w = Math.round(r.width);
      const hRaw = r.height * PICKER_FLOW_HEIGHT_VS_MAIN_CANVAS;
      const h = Math.round(hRaw);
      setLayout({
        widthPx: Number.isFinite(w) && w > 0 ? Math.max(280, w) : null,
        flowHeightPx:
          Number.isFinite(h) && h > 0 ? Math.max(PICKER_FLOW_MIN_HEIGHT_PX, h) : null,
      });
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    window.addEventListener("resize", apply);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", apply);
    };
  }, [open]);
  return layout;
}

const DEFAULT_WIRE_NAME_DRAFT = "hidden_0";

/** Merge current name draft into picks (same rules as committing via Enter) so Save picks never drops in-flight text. */
function applyDraftToPicks(
  picks: ActivationWirePick[],
  wire: ActivationPickerWireEdge | null,
  nameDraft: string,
): ActivationWirePick[] {
  const draft = nameDraft.trim();
  if (!wire || !draft) return picks;
  const label = draft.slice(0, 200);
  const idx = picks.findIndex((p) => p.afterModuleIndex === wire.afterModuleIndex);
  if (idx >= 0) {
    const used = new Set(picks.filter((_, i) => i !== idx).map((p) => p.tensorKey));
    const tensorKey = uniqueTensorKey(label, used);
    return picks.map((p, i) => (i === idx ? { ...p, label, tensorKey } : p));
  }
  const used = new Set(picks.map((p) => p.tensorKey));
  const tensorKey = uniqueTensorKey(label, used);
  return [
    ...picks,
    {
      pickId: newPickId(),
      tensorKey,
      label,
      afterModuleIndex: wire.afterModuleIndex,
      loopScope: "all" as const,
    },
  ];
}

function uniqueTensorKey(base: string, existing: Set<string>): string {
  let k = sanitizeTensorKey(base);
  if (!k) k = "activation";
  let out = k;
  let n = 2;
  while (existing.has(out)) {
    out = `${k}_${n}`;
    n += 1;
  }
  return out;
}

export function ActivationWirePickerModal({
  open,
  onClose,
  onSave,
  resolved,
  nodes,
  edges,
  initialPicks,
}: ActivationWirePickerModalProps) {
  const { widthPx: canvasWidthPx, flowHeightPx: pickerFlowHeightPx } = useWorkbenchCanvasPickerLayout(open);
  const build = useMemo(() => buildActivationWirePickerGraph(resolved, nodes, edges), [resolved, nodes, edges]);
  const [picks, setPicks] = useState<ActivationWirePick[]>([]);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [nameDraft, setNameDraft] = useState("");
  const [wireNameFocusNonce, setWireNameFocusNonce] = useState(0);
  /** True = text input; false = label (after Enter saves, or blur / click outside the input). */
  const [nameEditActive, setNameEditActive] = useState(false);

  const initialPicksDigest = useMemo(
    () =>
      JSON.stringify(
        [...initialPicks]
          .sort((a, b) => a.afterModuleIndex - b.afterModuleIndex)
          .map((p) => [p.afterModuleIndex, p.pickId, p.tensorKey, p.label, p.loopScope ?? "all"]),
      ),
    [initialPicks],
  );

  useEffect(() => {
    if (!open) return;
    setPicks(initialPicks.map((p) => ({ ...p })));
    setSelectedEdgeId(null);
    setNameDraft("");
    setNameEditActive(false);
  }, [open, initialPicksDigest, initialPicks]);

  const selectWireEdge = useCallback((edgeId: string) => {
    setSelectedEdgeId(edgeId);
  }, []);

  const clearWireSelection = useCallback(() => {
    setSelectedEdgeId(null);
    setNameDraft("");
    setNameEditActive(false);
  }, []);

  useEffect(() => {
    if (selectedEdgeId) setNameEditActive(true);
    else setNameEditActive(false);
  }, [selectedEdgeId]);

  const exitNameEditMode = useCallback(() => {
    setNameEditActive(false);
  }, []);

  const wires = build.ok ? build.wires : [];
  const selectedWire = wires.find((w) => w.edgeId === selectedEdgeId) ?? null;
  const selectedAfterModuleIndex = useMemo(() => {
    if (!build.ok || !selectedEdgeId) return null;
    return wires.find((w) => w.edgeId === selectedEdgeId)?.afterModuleIndex ?? null;
  }, [build.ok, selectedEdgeId, wires]);

  const commitWirePick = useCallback(() => {
    if (!selectedWire || !nameDraft.trim()) return;
    setPicks((prev) => applyDraftToPicks(prev, selectedWire, nameDraft));
    setNameEditActive(false);
  }, [nameDraft, selectedWire]);

  /** Saved label for the selected wire only — avoids re-seeding when picks change for other wires. */
  const savedLabelForSelectedWire = useMemo(() => {
    if (selectedAfterModuleIndex == null) return "";
    return picks.find((p) => p.afterModuleIndex === selectedAfterModuleIndex)?.label ?? "";
  }, [selectedAfterModuleIndex, picks]);

  /** Seed or refresh the name field when selection changes or this wire's saved label changes. */
  useEffect(() => {
    if (!build.ok) return;
    if (selectedAfterModuleIndex == null) {
      setNameDraft("");
      return;
    }
    setNameDraft(savedLabelForSelectedWire || DEFAULT_WIRE_NAME_DRAFT);
  }, [build.ok, selectedAfterModuleIndex, savedLabelForSelectedWire]);

  const openWireEditorForEdge = useCallback(
    (edgeId: string) => {
      if (selectedEdgeId !== edgeId) return;
      setNameEditActive(true);
      setWireNameFocusNonce((n) => n + 1);
    },
    [selectedEdgeId],
  );

  const removePickForEdge = useCallback(
    (edgeId: string, afterModuleIndexFromEdge?: number | null) => {
      const wire = wires.find((w) => w.edgeId === edgeId);
      const ami =
        wire?.afterModuleIndex ??
        (typeof afterModuleIndexFromEdge === "number" && Number.isFinite(afterModuleIndexFromEdge)
          ? afterModuleIndexFromEdge
          : null);
      if (ami == null) return;
      setPicks((prev) => prev.filter((p) => p.afterModuleIndex !== ami));
      if (selectedEdgeId === edgeId) clearWireSelection();
    },
    [wires, selectedEdgeId, clearWireSelection],
  );

  const canCommitWirePick = Boolean(selectedWire && nameDraft.trim());

  const wireEditor = useMemo<ActivationPickerWireEditorContextValue>(
    () => ({
      nameDraft,
      setNameDraft,
      commitWirePick,
      canCommitWirePick,
      nameEditActive,
      exitNameEditMode,
      defaultNameDisplay: DEFAULT_WIRE_NAME_DRAFT,
      wireNameFocusNonce,
      openWireEditorForEdge,
      removePickForEdge,
    }),
    [
      nameDraft,
      commitWirePick,
      canCommitWirePick,
      nameEditActive,
      exitNameEditMode,
      wireNameFocusNonce,
      openWireEditorForEdge,
      removePickForEdge,
    ],
  );

  useEffect(() => {
    if (!open || !build.ok) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        const active = document.activeElement;
        if (active instanceof HTMLElement && active.closest(".cr-apw-edge-name-slot")) {
          e.preventDefault();
          active.blur();
        }
        return;
      }
      if (e.key !== "d" && e.key !== "D") return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const active = document.activeElement;
      if (active instanceof HTMLElement && active.closest(".cr-apw-edge-name-slot, .cr-apw-edge-hit-bridge")) return;
      if (!selectedEdgeId) return;
      e.preventDefault();
      if (selectedAfterModuleIndex != null) {
        setPicks((prev) => prev.filter((p) => p.afterModuleIndex !== selectedAfterModuleIndex));
      }
      clearWireSelection();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, build.ok, selectedEdgeId, selectedAfterModuleIndex, clearWireSelection]);

  const handleSave = useCallback(() => {
    const flushed = applyDraftToPicks(picks, selectedWire, nameDraft);
    onSave(flushed);
    onClose();
  }, [onClose, onSave, picks, selectedWire, nameDraft]);

  if (!open) return null;

  const body =
    build.ok === false ? (
      <p className="cr-activation-picker-modal__err nodrag nopan">{build.message}</p>
    ) : (
      <>
        <p className="cr-activation-picker-modal__hint nodrag nopan">
          Drag blocks to rearrange. Click a wire to select it — the gauge appears with an editable name (default{' '}
          {DEFAULT_WIRE_NAME_DRAFT}). Press Enter to save the pick and show the name as a label; click outside the text
          box (e.g. the flow or gauge) to show the label without saving. Double-click the label or gauge to edit again.
          Clicking empty flow does not clear the selection. With a wire selected, press D to remove its saved gauge
          (pick) when focus is not in the name field — or to clear selection if there is no pick yet; you can also
          right-click the gauge (Ctrl+click on Mac) and choose Delete. Escape blurs the
          name field. Save picks commits all picks (including the name you typed for the selected wire without pressing
          Enter). This preview does not change your canvas until you save.
        </p>
        <ReactFlowProvider>
          <div
            className={`cr-activation-picker-flow-wrap${
              build.loopDisplay && !build.loopBadgeInFlow ? " cr-activation-picker-flow-wrap--has-loop" : ""
            }`}
          >
            {build.loopDisplay && !build.loopBadgeInFlow ? (
              <div className="cr-node-loop-badge cr-activation-picker-loop-badge nodrag nopan" aria-hidden>
                <span className="cr-node-loop-badge__icon" title={`Loop ×${build.loopDisplay.count}`}>
                  ⟲
                </span>
                <span className="cr-node-loop-badge__times">×{build.loopDisplay.count}</span>
              </div>
            ) : null}
            <PickerFlow
              flowNodes={build.flowNodes}
              flowEdges={build.flowEdges}
              wires={build.wires}
              picks={picks}
              selectedEdgeId={selectedEdgeId}
              graphStructureKey={build.flowNodes.map((n) => n.id).join("|")}
              onSelectWireEdge={selectWireEdge}
              wireEditor={wireEditor}
              flowHeightPx={pickerFlowHeightPx}
            />
          </div>
        </ReactFlowProvider>
      </>
    );

  return createPortal(
    <div className="cr-modal-backdrop" role="presentation" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        className="cr-modal cr-activation-picker-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="cr-activation-picker-title"
        onMouseDown={(e) => e.stopPropagation()}
        style={canvasWidthPx != null ? { width: `${canvasWidthPx}px` } : undefined}
      >
        <h2 id="cr-activation-picker-title" className="cr-modal__title nodrag nopan">
          Pick activations (preview)
        </h2>
        {body}
        <div className="cr-modal__actions nodrag nopan">
          <button type="button" className="cr-btn" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="cr-btn cr-btn--primary" onClick={handleSave}>
            Save picks
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
