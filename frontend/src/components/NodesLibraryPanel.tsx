import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
  type CSSProperties,
} from "react";
import { useReactFlow } from "@xyflow/react";
import { DeletingBusyOverlay } from "./DeletingBusyOverlay";
import type { ResearchGraphActions } from "../context/ResearchGraphContext";
import { useResearchGraph } from "../context/ResearchGraphContext";
import {
  COMBINED_MODEL_TEMPLATE_DND_MIME,
  DND_MIME,
  DND_TEXT_PLAIN,
  GRAPH_COMBINED_MODEL_LIBRARY_CHANGED,
  GRAPH_TEMPLATE_LIBRARY_CHANGED,
  USER_LINEAR_DATASET_DND_MIME,
  USER_LINEAR_DATASETS_CHANGED,
  USER_OBSERVABLE_DND_MIME,
  USER_OBSERVABLES_CHANGED,
  USER_SYMBOLIC_FUNC_DATASET_DND_MIME,
  USER_SYMBOLIC_FUNC_DATASETS_CHANGED,
} from "../dnd";
import { removeSavedGraphEntry, type SavedGraphEntry } from "../graph/savedGraphLibrary";
import { useCombinedModelLibraryTemplates } from "../hooks/useCombinedModelLibraryTemplates";
import { useNodeCategories, type CatalogNodeChild, type CatalogNodeCategory } from "../hooks/useNodeCategories";
import { GENERATED_NODE_SPECS } from "../generated/generatedNodeSpecs";
import { nodeRegistryDefaults } from "../graph/nodeRegistrySpec";
import { appendResearchNode } from "../graph/nodeInstanceTitle";
import {
  beginLibraryNodeDrag,
  endLibraryNodeDrag,
  markLibraryDragNode,
  updateLibraryNodeDragTarget,
} from "../graph/libraryNodeDrag";

type NodeChild = CatalogNodeChild;

type NodeCategory = CatalogNodeCategory;

import { useOptionalTheme } from "../themeContext";
import { normalizeTheme } from "../theme";

const LIBRARY_CATEGORY_STORAGE_KEY = "cr.nodesLibrary.category";

const CATEGORY_ACCENT: Record<string, string> = {
  dataset: "dataset",
  optimizer: "optimizer",
  model: "model",
  loss: "loss",
  observables: "observable",
  training: "trainer",
  checkpoint: "checkpoint",
  visualization: "tensor",
  language: "hypothesis",
  analysis: "tensor",
};

export function accentVarForCategory(category: string | undefined): string {
  const accent = (category && CATEGORY_ACCENT[category]) || "dataset";
  return `var(--cr-accent-${accent})`;
}

function canvasAccentForNode(nodeType: string): string {
  const category = GENERATED_NODE_SPECS[nodeType]?.category;
  const accent = (category && CATEGORY_ACCENT[category]) || "dataset";
  return `var(--cr-accent-${accent})`;
}

/** Small per-category glyph for library rows (studio panel design; classic
 * keeps the legacy row DOM without it). */
export function LibraryNodeGlyph({ nodeType }: { nodeType: string }) {
  const contextTheme = useOptionalTheme()?.theme;
  const isStudio =
    (contextTheme ?? normalizeTheme(document.documentElement.dataset.crTheme)) !== "classic";
  const category = GENERATED_NODE_SPECS[nodeType]?.category;
  const accent = (category && CATEGORY_ACCENT[category]) || "dataset";
  const common = { width: 13, height: 13, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };
  let body;
  switch (accent) {
    case "dataset":
      body = (<><ellipse cx="12" cy="6" rx="7" ry="3" /><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6" /><path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3" /></>);
      break;
    case "model":
      body = (<><circle cx="5" cy="12" r="2" /><circle cx="12" cy="5" r="2" /><circle cx="12" cy="19" r="2" /><circle cx="19" cy="12" r="2" /><path d="M6.8 10.8 10.2 6.6M6.8 13.2l3.4 4.2M13.8 6.6l3.4 4.2M13.8 17.4l3.4-4.2" /></>);
      break;
    case "optimizer":
      body = (<><circle cx="12" cy="12" r="8" /><path d="M12 8v4l2.6 2.6" /></>);
      break;
    case "loss":
      body = (<><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="3.5" /><circle cx="12" cy="12" r="0.8" fill="currentColor" /></>);
      break;
    case "observable":
    case "observables":
      body = (<path d="M3 13l4-2 3 5 4-10 3 6 4-1" />);
      break;
    case "tensor":
      body = (<><rect x="4" y="4" width="16" height="16" rx="2" /><path d="M4 12h16M12 4v16" /></>);
      break;
    default:
      body = (<><rect x="4" y="5" width="16" height="14" rx="2" /><path d="M8 10l3 3 5-6" /></>);
  }
  if (!isStudio) return null;
  return (
    <svg {...common} className="cr-node-pill-glyph" style={{ color: `var(--cr-accent-${accent})` }}>
      {body}
    </svg>
  );
}

function visibleChildrenForCategory(c: NodeCategory, q: string): NodeChild[] {
  const raw = c.children ?? [];
  if (q === "") return raw;
  if (c.label.toLowerCase().includes(q)) return raw;
  return raw.filter((ch) => ch.label.toLowerCase().includes(q));
}

function LibraryChildListItem({
  ch,
  graph,
  onLibraryPointerDown,
  onUserObsDragStart,
  onUserLinearDatasetDragStart,
  onUserSymbolicFuncDatasetDragStart,
  onDeleteUserObservable,
  onDeleteUserLinearDataset,
  onDeleteUserSymbolicFuncDataset,
}: {
  ch: NodeChild;
  graph: ResearchGraphActions | null;
  onLibraryPointerDown: (event: PointerEvent<HTMLButtonElement>, ch: NodeChild) => void;
  onUserObsDragStart: (event: DragEvent, ch: NodeChild, label: string) => void;
  onUserLinearDatasetDragStart: (event: DragEvent, ch: NodeChild, label: string) => void;
  onUserSymbolicFuncDatasetDragStart: (event: DragEvent, ch: NodeChild, label: string) => void;
  onDeleteUserObservable: (e: MouseEvent, itemId: string) => void;
  onDeleteUserLinearDataset: (e: MouseEvent, itemId: string) => void;
  onDeleteUserSymbolicFuncDataset: (e: MouseEvent, itemId: string) => void;
}) {
  return (
    <li>
      {ch.deletable && ch.user_observable_id ? (
        <UserObservableLibraryRow
          ch={ch}
          graph={graph}
          onDragStartPayload={onUserObsDragStart}
          onDelete={onDeleteUserObservable}
        />
      ) : ch.deletable && ch.user_linear_dataset_id ? (
        <UserLinearDatasetLibraryRow
          ch={ch}
          graph={graph}
          onDragStartPayload={onUserLinearDatasetDragStart}
          onDelete={onDeleteUserLinearDataset}
        />
      ) : ch.deletable && ch.user_symbolic_func_dataset_id ? (
        <UserSymbolicFuncDatasetLibraryRow
          ch={ch}
          graph={graph}
          onDragStartPayload={onUserSymbolicFuncDatasetDragStart}
          onDelete={onDeleteUserSymbolicFuncDataset}
        />
      ) : (
        <div className="cr-nodes-panel__node-row">
          <button
            type="button"
            className="cr-nodes-panel__node-pill"
            style={{ "--cr-library-accent": canvasAccentForNode(ch.id) } as CSSProperties}
            draggable={false}
            onPointerDown={(e) => onLibraryPointerDown(e, ch)}
            title="Drag onto the canvas"
          >
            <LibraryNodeGlyph nodeType={ch.id} />
            {ch.label}
          </button>
        </div>
      )}
    </li>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="cr-nodes-panel__section">
      <h3 className="cr-nodes-panel__section-title">{title}</h3>
      <div className="cr-nodes-panel__section-body">{children}</div>
    </section>
  );
}

function UserSymbolicFuncDatasetLibraryRow({
  ch,
  graph,
  onDragStartPayload,
  onDelete,
}: {
  ch: NodeChild;
  graph: ResearchGraphActions | null;
  onDragStartPayload: (event: DragEvent, ch: NodeChild, label: string) => void;
  onDelete: (e: MouseEvent, itemId: string) => void;
}) {
  const [label, setLabel] = useState(ch.label);

  useEffect(() => {
    setLabel(ch.label);
  }, [ch.label]);

  const saveLabel = useCallback(async () => {
    const t = label.trim();
    if (!t || !ch.user_symbolic_func_dataset_id || t === ch.label) return;
    try {
      const r = await fetch(
        `/api/user-symbolic-func-datasets/${encodeURIComponent(ch.user_symbolic_func_dataset_id)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ label: t }),
        },
      );
      if (r.ok) {
        window.dispatchEvent(new Event(USER_SYMBOLIC_FUNC_DATASETS_CHANGED));
      }
    } catch {
      /* ignore */
    }
  }, [ch.label, ch.user_symbolic_func_dataset_id, label]);

  const uid = ch.user_symbolic_func_dataset_id ?? "x";

  return (
    <div
      className="cr-nodes-panel__node-row cr-nodes-panel__node-row--user-obs"
      onDoubleClick={(e) => {
        if ((e.target as HTMLElement).closest("input,textarea,button")) return;
        graph?.addNode("symbolic_func_dataset", undefined, {
          userSymbolicFuncDatasetId: ch.user_symbolic_func_dataset_id,
        });
      }}
      title="Edit name, drag this row onto the canvas, or double-click to add"
    >
      <input
        type="text"
        id={`user-sfd-label-${uid}`}
        name={`user-sfd-label-${uid}`}
        className="cr-nodes-panel__user-obs-label"
        draggable
        onDragStart={(e) => onDragStartPayload(e, ch, label.trim() || ch.label)}
        value={label}
        onChange={(e) => setLabel(e.target.value)}
        onBlur={() => void saveLabel()}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
        onMouseDown={(e) => e.stopPropagation()}
        aria-label="Symbolic dataset blueprint name — drag onto canvas"
      />
      <button
        type="button"
        className="cr-nodes-panel__icon-btn cr-nodes-panel__icon-btn--trash"
        draggable={false}
        title="Delete saved dataset"
        aria-label={`Delete ${ch.label}`}
        onClick={(e) => onDelete(e, ch.user_symbolic_func_dataset_id!)}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
          <path
            d="M9 3h6M4 7h16M6 7l1 14h10l1-14M10 11v6M14 11v6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  );
}

function UserLinearDatasetLibraryRow({
  ch,
  graph,
  onDragStartPayload,
  onDelete,
}: {
  ch: NodeChild;
  graph: ResearchGraphActions | null;
  onDragStartPayload: (event: DragEvent, ch: NodeChild, label: string) => void;
  onDelete: (e: MouseEvent, itemId: string) => void;
}) {
  const [label, setLabel] = useState(ch.label);

  useEffect(() => {
    setLabel(ch.label);
  }, [ch.label]);

  const saveLabel = useCallback(async () => {
    const t = label.trim();
    if (!t || !ch.user_linear_dataset_id || t === ch.label) return;
    try {
      const r = await fetch(`/api/user-linear-datasets/${encodeURIComponent(ch.user_linear_dataset_id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: t }),
      });
      if (r.ok) {
        window.dispatchEvent(new Event(USER_LINEAR_DATASETS_CHANGED));
      }
    } catch {
      /* ignore */
    }
  }, [ch.label, ch.user_linear_dataset_id, label]);

  const uid = ch.user_linear_dataset_id ?? "x";

  return (
    <div
      className="cr-nodes-panel__node-row cr-nodes-panel__node-row--user-obs"
      onDoubleClick={(e) => {
        if ((e.target as HTMLElement).closest("input,textarea,button")) return;
        graph?.addNode("linear_dataset", undefined, {
          userLinearDatasetId: ch.user_linear_dataset_id,
        });
      }}
      title="Edit name, drag this row onto the canvas, or double-click to add"
    >
      <input
        type="text"
        id={`user-ld-label-${uid}`}
        name={`user-ld-label-${uid}`}
        className="cr-nodes-panel__user-obs-label"
        draggable
        onDragStart={(e) => onDragStartPayload(e, ch, label.trim() || ch.label)}
        value={label}
        onChange={(e) => setLabel(e.target.value)}
        onBlur={() => void saveLabel()}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
        onMouseDown={(e) => e.stopPropagation()}
        aria-label="Dataset blueprint name — drag onto canvas"
      />
      <button
        type="button"
        className="cr-nodes-panel__icon-btn cr-nodes-panel__icon-btn--trash"
        draggable={false}
        title="Delete saved dataset"
        aria-label={`Delete ${ch.label}`}
        onClick={(e) => onDelete(e, ch.user_linear_dataset_id!)}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
          <path
            d="M9 3h6M4 7h16M6 7l1 14h10l1-14M10 11v6M14 11v6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  );
}

function CombinedModelLibraryRow({
  entry,
  storageKind,
  withDeleteOverlay,
}: {
  entry: SavedGraphEntry;
  storageKind: "workflow" | "template";
  withDeleteOverlay: (message: string, fn: () => Promise<void>) => Promise<void>;
}) {
  const n = entry.document?.nodes;
  const sourceNodeCount = Array.isArray(n) ? n.length : 0;

  const onDragStart = useCallback(
    (event: DragEvent) => {
      event.dataTransfer.setData(DND_MIME, "combined_model");
      event.dataTransfer.setData(DND_TEXT_PLAIN, "combined_model");
      const base = {
        templateId: entry.id,
        displayName: entry.name,
        sourceNodeCount,
        document: entry.document,
      };
      let payload = JSON.stringify(base);
      if (payload.length > 240_000) {
        payload = JSON.stringify({
          templateId: entry.id,
          displayName: entry.name,
          sourceNodeCount,
        });
      }
      event.dataTransfer.setData(COMBINED_MODEL_TEMPLATE_DND_MIME, payload);
      event.dataTransfer.effectAllowed = "move";
    },
    [entry.document, entry.id, entry.name, sourceNodeCount],
  );

  const onDelete = useCallback(
    async (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      await withDeleteOverlay("Removing combined model from library…", async () => {
        try {
          await removeSavedGraphEntry(storageKind, entry.id);
          window.dispatchEvent(new Event(GRAPH_COMBINED_MODEL_LIBRARY_CHANGED));
          if (storageKind === "template") {
            window.dispatchEvent(new Event(GRAPH_TEMPLATE_LIBRARY_CHANGED));
          }
        } catch {
          /* ignore */
        }
      });
    },
    [entry.id, storageKind, withDeleteOverlay],
  );

  return (
    <div className="cr-nodes-panel__node-row">
      <div
        className="cr-nodes-panel__library-pill-enclosed"
        title="Drag onto the canvas"
      >
        <button
          type="button"
          className="cr-nodes-panel__library-pill-enclosed__drag nodrag nopan"
          draggable
          onDragStart={onDragStart}
          onMouseDown={(e) => e.stopPropagation()}
        >
          {entry.name}
        </button>
        <button
          type="button"
          className="cr-nodes-panel__library-pill-enclosed__trash nodrag nopan"
          draggable={false}
          title="Remove from library"
          aria-label={`Delete ${entry.name}`}
          onClick={(e) => void onDelete(e)}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
            <path
              d="M9 3h6M4 7h16M6 7l1 14h10l1-14M10 11v6M14 11v6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}

function UserObservableLibraryRow({
  ch,
  graph,
  onDragStartPayload,
  onDelete,
}: {
  ch: NodeChild;
  graph: ResearchGraphActions | null;
  onDragStartPayload: (event: DragEvent, ch: NodeChild, label: string) => void;
  onDelete: (e: MouseEvent, itemId: string) => void;
}) {
  const [label, setLabel] = useState(ch.label);

  useEffect(() => {
    setLabel(ch.label);
  }, [ch.label]);

  const saveLabel = useCallback(async () => {
    const t = label.trim();
    if (!t || !ch.user_observable_id || t === ch.label) return;
    try {
      const r = await fetch(`/api/user-observables/${encodeURIComponent(ch.user_observable_id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: t }),
      });
      if (r.ok) {
        window.dispatchEvent(new Event(USER_OBSERVABLES_CHANGED));
      }
    } catch {
      /* ignore */
    }
  }, [ch.label, ch.user_observable_id, label]);

  const uid = ch.user_observable_id ?? "x";

  return (
    <div
      className="cr-nodes-panel__node-row cr-nodes-panel__node-row--user-obs"
      onDoubleClick={(e) => {
        if ((e.target as HTMLElement).closest("input,textarea,button")) return;
        graph?.addNode("observable_user", undefined, {
          userObservableId: ch.user_observable_id,
          label: label.trim() || ch.label,
          tensorVizNodeId: ch.tensor_viz_node_id,
          tensorSelectorNodeId: ch.tensor_selector_node_id,
        });
      }}
      title="Edit name, drag this row onto the canvas, or double-click to add"
    >
      <input
        type="text"
        id={`user-obs-label-${uid}`}
        name={`user-obs-label-${uid}`}
        className="cr-nodes-panel__user-obs-label"
        draggable
        onDragStart={(e) => onDragStartPayload(e, ch, label.trim() || ch.label)}
        value={label}
        onChange={(e) => setLabel(e.target.value)}
        onBlur={() => void saveLabel()}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
        onMouseDown={(e) => e.stopPropagation()}
        aria-label="Observable name — drag onto canvas"
      />
      <button
        type="button"
        className="cr-nodes-panel__icon-btn cr-nodes-panel__icon-btn--trash"
        draggable={false}
        title="Delete user observable"
        aria-label={`Delete ${ch.label}`}
        onClick={(e) => onDelete(e, ch.user_observable_id!)}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
          <path
            d="M9 3h6M4 7h16M6 7l1 14h10l1-14M10 11v6M14 11v6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  );
}

export function NodesLibraryPanel() {
  const graph = useResearchGraph();
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"all" | "blueprints">("all");
  const [libraryCategoryId, setLibraryCategoryId] = useState<string | null>(() => {
    try {
      const v = sessionStorage.getItem(LIBRARY_CATEGORY_STORAGE_KEY);
      if (v == null || v === "" || v === "all") return null;
      return v;
    } catch {
      return null;
    }
  });
  const [libraryDeletingMessage, setLibraryDeletingMessage] = useState<string | null>(null);
  const libraryDeleteDepthRef = useRef(0);
  const withLibraryDeleteOverlay = useCallback(async (message: string, fn: () => Promise<void>) => {
    libraryDeleteDepthRef.current += 1;
    if (libraryDeleteDepthRef.current === 1) setLibraryDeletingMessage(message);
    try {
      await fn();
    } finally {
      libraryDeleteDepthRef.current -= 1;
      if (libraryDeleteDepthRef.current <= 0) {
        libraryDeleteDepthRef.current = 0;
        setLibraryDeletingMessage(null);
      }
    }
  }, []);
  const categories = useNodeCategories();
  const combinedModelEntries = useCombinedModelLibraryTemplates();
  const { screenToFlowPosition } = useReactFlow();

  useEffect(
    () => () => endLibraryNodeDrag(),
    [],
  );

  const onLibraryPointerDown = useCallback(
    (event: PointerEvent<HTMLButtonElement>, ch: NodeChild) => {
      if (event.button !== 0 || !graph?.setFlowNodes) return;

      const start = { x: event.clientX, y: event.clientY };
      const isOverCanvas = (x: number, y: number) =>
        document.elementFromPoint(x, y)?.closest(".react-flow") !== null;

      const draftId = `library-drag-${ch.id}-${Math.random().toString(36).slice(2, 10)}`;
      const initialPosition = screenToFlowPosition(start);
      // Library-created drafts and existing canvas nodes share the same active
      // drag identity, so their rail stacking and clipping rules stay aligned.
      markLibraryDragNode(draftId);
      beginLibraryNodeDrag();
      const startsOverLibrary = updateLibraryNodeDragTarget(start.x, start.y);
      graph.setFlowNodes((nodes) => {
        const draft = appendResearchNode(
          nodes,
          ch.id,
          initialPosition,
          nodeRegistryDefaults(ch.id) ?? {},
          draftId,
        );
        return [
          ...nodes,
          {
            ...draft,
            className: `cr-library-drag-draft${startsOverLibrary ? " cr-library-drag-source-hidden" : ""}`,
            dragging: true,
            zIndex: 1000,
          },
        ];
      });

      const move = (pointer: globalThis.PointerEvent) => {
        const overLibrary = updateLibraryNodeDragTarget(pointer.clientX, pointer.clientY);
        const position = screenToFlowPosition({ x: pointer.clientX, y: pointer.clientY });
        graph.setFlowNodes((nodes) =>
          nodes.map((node) =>
            node.id === draftId
              ? {
                  ...node,
                  className: `cr-library-drag-draft${overLibrary ? " cr-library-drag-source-hidden" : ""}`,
                  position,
                  dragging: true,
                }
              : node,
          ),
        );
      };

      const finish = (pointer: globalThis.PointerEvent) => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", finish);
        window.removeEventListener("pointercancel", finish);
        endLibraryNodeDrag();

        const keepDraft = isOverCanvas(pointer.clientX, pointer.clientY);
        graph.setFlowNodes((nodes) =>
          keepDraft
            ? nodes.map((node) => {
                if (node.id !== draftId) return node;
                const { className: _dragClass, zIndex: _dragZIndex, ...rest } = node;
                return {
                  ...rest,
                  position: screenToFlowPosition({ x: pointer.clientX, y: pointer.clientY }),
                  dragging: false,
                  selected: false,
                };
              })
            : nodes.filter((node) => node.id !== draftId),
        );
        if (keepDraft) {
          window.requestAnimationFrame(() => {
            const finalNode = [...document.querySelectorAll<HTMLElement>(".react-flow__node")].find(
              (node) => node.dataset.id === draftId,
            );
            finalNode?.style.removeProperty("visibility");
            finalNode?.classList.remove("cr-library-drag-source-hidden", "cr-library-active-drag");
          });
        }
      };

      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", finish);
      window.addEventListener("pointercancel", finish);
    },
    [graph, screenToFlowPosition],
  );

  const onUserObsDragStart = useCallback((event: DragEvent, ch: NodeChild, label: string) => {
    const id = String(ch.id ?? "").trim();
    event.dataTransfer.setData(DND_MIME, id);
    event.dataTransfer.setData(DND_TEXT_PLAIN, id);
    event.dataTransfer.setData(
      USER_OBSERVABLE_DND_MIME,
      JSON.stringify({
        userObservableId: ch.user_observable_id,
        label,
        tensorVizNodeId: ch.tensor_viz_node_id ?? "",
        tensorSelectorNodeId: ch.tensor_selector_node_id ?? "",
      }),
    );
    event.dataTransfer.effectAllowed = "move";
  }, []);

  const onUserLinearDatasetDragStart = useCallback((event: DragEvent, ch: NodeChild, label: string) => {
    const id = String(ch.id ?? "").trim();
    event.dataTransfer.setData(DND_MIME, id);
    event.dataTransfer.setData(DND_TEXT_PLAIN, id);
    event.dataTransfer.setData(
      USER_LINEAR_DATASET_DND_MIME,
      JSON.stringify({
        userLinearDatasetId: ch.user_linear_dataset_id,
        label,
      }),
    );
    event.dataTransfer.effectAllowed = "move";
  }, []);

  const onUserSymbolicFuncDatasetDragStart = useCallback((event: DragEvent, ch: NodeChild, label: string) => {
    const id = String(ch.id ?? "").trim();
    event.dataTransfer.setData(DND_MIME, id);
    event.dataTransfer.setData(DND_TEXT_PLAIN, id);
    event.dataTransfer.setData(
      USER_SYMBOLIC_FUNC_DATASET_DND_MIME,
      JSON.stringify({
        userSymbolicFuncDatasetId: ch.user_symbolic_func_dataset_id,
        label,
      }),
    );
    event.dataTransfer.effectAllowed = "move";
  }, []);

  const onDeleteUserObservable = useCallback(
    async (e: React.MouseEvent, itemId: string) => {
      e.preventDefault();
      e.stopPropagation();
      await withLibraryDeleteOverlay("Deleting user observable…", async () => {
        try {
          const r = await fetch(`/api/user-observables/${encodeURIComponent(itemId)}`, {
            method: "DELETE",
          });
          if (r.ok) {
            window.dispatchEvent(
              new CustomEvent(USER_OBSERVABLES_CHANGED, {
                detail: { deletedUserObservableId: itemId },
              }),
            );
          }
        } catch {
          /* ignore */
        }
      });
    },
    [withLibraryDeleteOverlay],
  );

  const onDeleteUserLinearDataset = useCallback(
    async (e: React.MouseEvent, itemId: string) => {
      e.preventDefault();
      e.stopPropagation();
      await withLibraryDeleteOverlay("Deleting saved linear dataset…", async () => {
        try {
          const r = await fetch(`/api/user-linear-datasets/${encodeURIComponent(itemId)}`, {
            method: "DELETE",
          });
          if (r.ok) {
            window.dispatchEvent(
              new CustomEvent(USER_LINEAR_DATASETS_CHANGED, {
                detail: { deletedLinearDatasetId: itemId },
              }),
            );
          }
        } catch {
          /* ignore */
        }
      });
    },
    [withLibraryDeleteOverlay],
  );

  const onDeleteUserSymbolicFuncDataset = useCallback(
    async (e: React.MouseEvent, itemId: string) => {
      e.preventDefault();
      e.stopPropagation();
      await withLibraryDeleteOverlay("Deleting saved symbolic dataset…", async () => {
        try {
          const r = await fetch(`/api/user-symbolic-func-datasets/${encodeURIComponent(itemId)}`, {
            method: "DELETE",
          });
          if (r.ok) {
            window.dispatchEvent(
              new CustomEvent(USER_SYMBOLIC_FUNC_DATASETS_CHANGED, {
                detail: { deletedSymbolicFuncDatasetId: itemId },
              }),
            );
          }
        } catch {
          /* ignore */
        }
      });
    },
    [withLibraryDeleteOverlay],
  );

  const q = query.trim().toLowerCase();
  const combinedFiltered =
    q === ""
      ? combinedModelEntries
      : combinedModelEntries.filter((e) => e.entry.name.toLowerCase().includes(q));
  const filtered =
    q === ""
      ? categories
      : categories.filter(
          (c) =>
            c.label.toLowerCase().includes(q) ||
            (c.children?.some((ch) => ch.label.toLowerCase().includes(q)) ?? false) ||
            (c.id === "model" && combinedFiltered.length > 0),
        );

  const libraryTabCategories = useMemo(
    () => (q === "" ? categories : filtered),
    [q, categories, filtered],
  );

  useEffect(() => {
    try {
      sessionStorage.setItem(
        LIBRARY_CATEGORY_STORAGE_KEY,
        libraryCategoryId === null ? "all" : libraryCategoryId,
      );
    } catch {
      /* ignore */
    }
  }, [libraryCategoryId]);

  useEffect(() => {
    if (libraryCategoryId === null) return;
    const allowed = new Set(libraryTabCategories.map((c) => c.id));
    if (!allowed.has(libraryCategoryId)) setLibraryCategoryId(null);
  }, [libraryCategoryId, libraryTabCategories]);

  const selectedLibraryCategory =
    libraryCategoryId === null ? null : categories.find((c) => c.id === libraryCategoryId) ?? null;

  return (
    <>
    <aside className="cr-nodes-panel" aria-label="Node library">
      <header className="cr-nodes-panel__header">
        <h2 className="cr-nodes-panel__title">Nodes</h2>
        <div className="cr-nodes-panel__search-row">
          <input
            type="search"
            className="cr-nodes-panel__search"
            placeholder="Search…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search nodes"
          />
          <button type="button" className="cr-nodes-panel__icon-btn" title="Sort" aria-label="Sort">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M4 6h16M8 12h12M11 18h9" strokeLinecap="round" />
            </svg>
          </button>
          <button type="button" className="cr-nodes-panel__icon-btn" title="Filter" aria-label="Filter">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M4 5h16l-6 7v5l-4 2v-7L4 5z" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
        <div className="cr-nodes-panel__tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "all"}
            className={`cr-nodes-panel__tab${tab === "all" ? " cr-nodes-panel__tab--active" : ""}`}
            onClick={() => setTab("all")}
          >
            All
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "blueprints"}
            className={`cr-nodes-panel__tab${tab === "blueprints" ? " cr-nodes-panel__tab--active" : ""}`}
            onClick={() => setTab("blueprints")}
          >
            Blueprints
          </button>
        </div>
      </header>

      <div className="cr-nodes-panel__scroll">
        <Section title="Bookmarked">
          <p className="cr-nodes-panel__empty">No favorites yet.</p>
        </Section>

        <Section title="Blueprints">
          <p className="cr-nodes-panel__empty">No blueprints yet.</p>
        </Section>

        {tab === "all" ? (
          <Section title="Library">
            {filtered.length === 0 ? (
              <p className="cr-nodes-panel__empty">
                {categories.length === 0
                  ? "No node categories yet. Register types in the app and they will appear here."
                  : "No nodes match your search."}
              </p>
            ) : (
              <>
                <div
                  className="cr-nodes-panel__library-tabs"
                  role="tablist"
                  aria-label="Library categories"
                >
                  <button
                    type="button"
                    role="tab"
                    aria-selected={libraryCategoryId === null}
                    className={`cr-nodes-panel__library-tab${
                      libraryCategoryId === null ? " cr-nodes-panel__library-tab--active" : ""
                    }`}
                    onClick={() => setLibraryCategoryId(null)}
                  >
                    All
                  </button>
                  {libraryTabCategories.map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      role="tab"
                      aria-selected={libraryCategoryId === c.id}
                      className={`cr-nodes-panel__library-tab${
                        libraryCategoryId === c.id ? " cr-nodes-panel__library-tab--active" : ""
                      }`}
                      onClick={() => setLibraryCategoryId(c.id)}
                    >
                      {c.label}
                    </button>
                  ))}
                </div>
                {libraryCategoryId === null ? (
                  <ul className="cr-nodes-panel__tree">
                    {filtered.map((c) => (
                      <li key={c.id}>
                        <span className="cr-nodes-panel__folder">{c.label}</span>
                        {c.children?.length || (c.id === "model" && combinedFiltered.length > 0) ? (
                          <ul>
                            {(c.children ?? []).map((ch) => (
                              <LibraryChildListItem
                                key={
                                  ch.user_observable_id
                                    ? `uo-${ch.user_observable_id}`
                                    : ch.user_linear_dataset_id
                                      ? `uld-${ch.user_linear_dataset_id}`
                                      : ch.user_symbolic_func_dataset_id
                                        ? `usfd-${ch.user_symbolic_func_dataset_id}`
                                        : `${c.id}-${ch.id}-${ch.label}`
                                }
                                ch={ch}
                                graph={graph}
                                onLibraryPointerDown={onLibraryPointerDown}
                                onUserObsDragStart={onUserObsDragStart}
                                onUserLinearDatasetDragStart={onUserLinearDatasetDragStart}
                                onUserSymbolicFuncDatasetDragStart={onUserSymbolicFuncDatasetDragStart}
                                onDeleteUserObservable={onDeleteUserObservable}
                                onDeleteUserLinearDataset={onDeleteUserLinearDataset}
                                onDeleteUserSymbolicFuncDataset={onDeleteUserSymbolicFuncDataset}
                              />
                            ))}
                            {c.id === "model" && combinedFiltered.length > 0
                              ? combinedFiltered.map((row) => (
                                  <li key={`saved-cm-${row.entry.id}`}>
                                    <CombinedModelLibraryRow
                                      entry={row.entry}
                                      storageKind={row.storageKind}
                                      withDeleteOverlay={withLibraryDeleteOverlay}
                                    />
                                  </li>
                                ))
                              : null}
                          </ul>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                ) : selectedLibraryCategory ? (
                  (() => {
                    const c = selectedLibraryCategory;
                    const children = visibleChildrenForCategory(c, q);
                    const showCombined = c.id === "model" && combinedFiltered.length > 0;
                    if (children.length === 0 && !showCombined) {
                      return <p className="cr-nodes-panel__empty">No nodes match your search.</p>;
                    }
                    return (
                      <ul className="cr-nodes-panel__tree">
                        {children.map((ch) => (
                          <LibraryChildListItem
                            key={
                              ch.user_observable_id
                                ? `uo-${ch.user_observable_id}`
                                : ch.user_linear_dataset_id
                                  ? `uld-${ch.user_linear_dataset_id}`
                                  : ch.user_symbolic_func_dataset_id
                                    ? `usfd-${ch.user_symbolic_func_dataset_id}`
                                    : `${c.id}-${ch.id}-${ch.label}`
                            }
                            ch={ch}
                            graph={graph}
                            onLibraryPointerDown={onLibraryPointerDown}
                            onUserObsDragStart={onUserObsDragStart}
                            onUserLinearDatasetDragStart={onUserLinearDatasetDragStart}
                            onUserSymbolicFuncDatasetDragStart={onUserSymbolicFuncDatasetDragStart}
                            onDeleteUserObservable={onDeleteUserObservable}
                            onDeleteUserLinearDataset={onDeleteUserLinearDataset}
                            onDeleteUserSymbolicFuncDataset={onDeleteUserSymbolicFuncDataset}
                          />
                        ))}
                        {showCombined
                          ? combinedFiltered.map((row) => (
                              <li key={`saved-cm-${row.entry.id}`}>
                                <CombinedModelLibraryRow
                                  entry={row.entry}
                                  storageKind={row.storageKind}
                                  withDeleteOverlay={withLibraryDeleteOverlay}
                                />
                              </li>
                            ))
                          : null}
                      </ul>
                    );
                  })()
                ) : (
                  <p className="cr-nodes-panel__empty">No nodes match your search.</p>
                )}
              </>
            )}
          </Section>
        ) : null}
      </div>
    </aside>
    <DeletingBusyOverlay open={libraryDeletingMessage !== null} message={libraryDeletingMessage ?? "Deleting…"} />
    </>
  );
}
