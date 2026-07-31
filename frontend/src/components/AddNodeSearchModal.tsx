import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";
import type { AddNodeOptions } from "../context/ResearchGraphContext";
import { nodeRegistryHint } from "../graph/nodeRegistrySpec";
import type { SavedGraphEntry } from "../graph/savedGraphLibrary";
import { useCombinedModelLibraryTemplates } from "../hooks/useCombinedModelLibraryTemplates";
import { LibraryNodeGlyph, accentVarForCategory } from "./NodesLibraryPanel";
import { GENERATED_NODE_SPECS } from "../generated/generatedNodeSpecs";
import { InfoRichText } from "./nodes/InfoRichText";
import {
  type CatalogNodeChild,
  type CatalogNodeCategory,
  useNodeCategories,
} from "../hooks/useNodeCategories";

type FlatEntry = {
  key: string;
  nodeType: string;
  label: string;
  categoryId: string;
  categoryLabel: string;
  child: CatalogNodeChild;
  /** Library template row under Model (not from ``/api/node-categories``). */
  combinedTemplate?: SavedGraphEntry;
};

function flattenCategories(categories: CatalogNodeCategory[]): FlatEntry[] {
  const out: FlatEntry[] = [];
  for (const c of categories) {
    const children = c.children ?? [];
    for (const ch of children) {
      out.push({
        key:
          ch.user_observable_id != null && ch.user_observable_id !== ""
            ? `uo-${ch.user_observable_id}`
            : ch.user_linear_dataset_id != null && ch.user_linear_dataset_id !== ""
              ? `uld-${ch.user_linear_dataset_id}`
              : ch.user_symbolic_func_dataset_id != null && ch.user_symbolic_func_dataset_id !== ""
                ? `usfd-${ch.user_symbolic_func_dataset_id}`
                : `${c.id}-${ch.id}-${ch.label}`,
        nodeType: ch.id,
        label: ch.label,
        categoryId: c.id,
        categoryLabel: c.label,
        child: ch,
      });
    }
  }
  return out;
}

type AddNodeSearchModalProps = {
  open: boolean;
  onClose: () => void;
  /** Screen-space point where the node should appear (passed to addNode). */
  placeAt: { x: number; y: number } | null;
  onAdd: (nodeType: string, screenPos: { x: number; y: number }, options?: AddNodeOptions) => void;
};

export function AddNodeSearchModal({ open, onClose, placeAt, onAdd }: AddNodeSearchModalProps) {
  const categories = useNodeCategories();
  const combinedModelEntries = useCombinedModelLibraryTemplates();
  const catalogFlat = useMemo(() => flattenCategories(categories), [categories]);
  const combinedFlat = useMemo<FlatEntry[]>(
    () =>
      combinedModelEntries.map((row) => ({
        key: `cm-${row.entry.id}`,
        nodeType: "combined_model",
        label: row.entry.name,
        categoryId: "model",
        categoryLabel: "model",
        child: { id: "combined_model", label: row.entry.name },
        combinedTemplate: row.entry,
      })),
    [combinedModelEntries],
  );
  const flat = useMemo(() => [...catalogFlat, ...combinedFlat], [catalogFlat, combinedFlat]);
  const [query, setQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const q = query.trim().toLowerCase();
  const combinedFiltered = useMemo(
    () =>
      q === ""
        ? combinedModelEntries
        : combinedModelEntries.filter((row) => row.entry.name.toLowerCase().includes(q)),
    [combinedModelEntries, q],
  );

  const filteredFlat = useMemo(() => {
    let rows = flat;
    if (activeCategory != null) {
      rows = rows.filter((e) => e.categoryId === activeCategory);
    }
    if (!q) return rows;
    return rows.filter(
      (e) =>
        e.label.toLowerCase().includes(q) ||
        e.nodeType.toLowerCase().includes(q) ||
        e.categoryLabel.toLowerCase().includes(q),
    );
  }, [flat, q, activeCategory]);

  useEffect(() => {
    if (!open) {
      setQuery("");
      setActiveCategory(null);
      setSelectedKey(null);
      return;
    }
    const id = window.setTimeout(() => inputRef.current?.focus(), 10);
    return () => clearTimeout(id);
  }, [open]);

  useEffect(() => {
    if (
      activeCategory != null &&
      !categories.some((c) => c.id === activeCategory)
    ) {
      setActiveCategory(null);
    }
  }, [activeCategory, categories]);

  useEffect(() => {
    if (!open || filteredFlat.length === 0) {
      setSelectedKey(null);
      return;
    }
    setSelectedKey((k) => {
      if (k && filteredFlat.some((e) => e.key === k)) return k;
      return filteredFlat[0]?.key ?? null;
    });
  }, [open, filteredFlat]);

  const selectedEntry = useMemo(
    () => filteredFlat.find((e) => e.key === selectedKey) ?? null,
    [filteredFlat, selectedKey],
  );

  const commitAdd = useCallback(
    (entry: FlatEntry) => {
      if (!placeAt) return;
      const tmpl = entry.combinedTemplate;
      if (tmpl) {
        const n = tmpl.document?.nodes;
        const sourceNodeCount = Array.isArray(n) ? n.length : 0;
        onAdd("combined_model", placeAt, {
          combinedModelTemplateId: tmpl.id,
          combinedModelDisplayName: tmpl.name,
          combinedModelSourceNodeCount: sourceNodeCount,
          combinedModelTemplateDocument: tmpl.document,
        });
        onClose();
        return;
      }
      const ch = entry.child;
      if (entry.nodeType === "observable_user" && ch.user_observable_id) {
        onAdd(
          "observable_user",
          placeAt,
          {
            userObservableId: ch.user_observable_id,
            label: ch.label,
            tensorVizNodeId: ch.tensor_viz_node_id,
            tensorSelectorNodeId: ch.tensor_selector_node_id,
          },
        );
      } else if (entry.nodeType === "linear_dataset" && ch.user_linear_dataset_id) {
        onAdd("linear_dataset", placeAt, { userLinearDatasetId: ch.user_linear_dataset_id });
      } else if (entry.nodeType === "symbolic_func_dataset" && ch.user_symbolic_func_dataset_id) {
        onAdd("symbolic_func_dataset", placeAt, { userSymbolicFuncDatasetId: ch.user_symbolic_func_dataset_id });
      } else {
        onAdd(entry.nodeType, placeAt);
      }
      onClose();
    },
    [onAdd, onClose, placeAt],
  );

  const onBackdropPointerDown = useCallback(
    (e: ReactMouseEvent) => {
      if (e.target === e.currentTarget) onClose();
    },
    [onClose],
  );

  const onSearchKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        if (filteredFlat.length === 0) return;
        const i = Math.max(
          0,
          filteredFlat.findIndex((x) => x.key === selectedKey) + 1,
        );
        setSelectedKey(filteredFlat[Math.min(i, filteredFlat.length - 1)]!.key);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        if (filteredFlat.length === 0) return;
        const i = filteredFlat.findIndex((x) => x.key === selectedKey);
        const next = i <= 0 ? 0 : i - 1;
        setSelectedKey(filteredFlat[next]!.key);
        return;
      }
      if (e.key === "Enter") {
        e.preventDefault();
        const entry = selectedEntry ?? filteredFlat[0];
        if (entry) commitAdd(entry);
      }
    },
    [commitAdd, filteredFlat, onClose, selectedEntry, selectedKey],
  );

  if (!open || !placeAt) return null;

  const selectedNodeType = selectedEntry?.nodeType ?? "";
  const hint =
    nodeRegistryHint(selectedNodeType) ?? "Adds this node to the graph at the click location.";

  return (
    <div
      className="cr-add-node-modal-backdrop"
      role="presentation"
      onMouseDown={onBackdropPointerDown}
    >
      <div
        className="cr-add-node-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="cr-add-node-modal-title"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="cr-add-node-modal__head">
          <input
            ref={inputRef}
            id="cr-add-node-modal-search"
            type="search"
            className="cr-add-node-modal__search"
            placeholder="Add a node…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onSearchKeyDown}
            aria-label="Search node types"
            autoComplete="off"
            spellCheck={false}
          />
          <span id="cr-add-node-modal-title" className="cr-sr-only">
            Add a node
          </span>
        </div>

        <div className="cr-add-node-modal__tabs" role="tablist">
          <button
            type="button"
            role="tab"
            className={`cr-add-node-modal__chip${activeCategory === null ? " cr-add-node-modal__chip--active" : ""}`}
            aria-selected={activeCategory === null}
            onClick={() => setActiveCategory(null)}
          >
            All <span className="cr-add-node-modal__chip-n">{flat.length}</span>
          </button>
          {categories.map((c) => (
            <button
              key={c.id}
              type="button"
              role="tab"
              className={`cr-add-node-modal__chip${activeCategory === c.id ? " cr-add-node-modal__chip--active" : ""}`}
              aria-selected={activeCategory === c.id}
              onClick={() => setActiveCategory((prev) => (prev === c.id ? null : c.id))}
            >
              <span
                className="cr-add-node-modal__chip-dot"
                style={{ background: accentVarForCategory(c.id) }}
              />
              {c.label}
              <span className="cr-add-node-modal__chip-n">
                {flat.filter((e) => e.categoryId === c.id).length}
              </span>
            </button>
          ))}
        </div>

        <div className="cr-add-node-modal__body">
          <div className="cr-add-node-modal__main">
            <ul className="cr-add-node-modal__results" role="listbox" aria-label="Matching nodes">
              {filteredFlat.length === 0 ? (
                <li className="cr-add-node-modal__empty">No nodes match “{query.trim() || "…"}”.</li>
              ) : (
                filteredFlat.map((e) => (
                  <li key={e.key}>
                    <button
                      type="button"
                      role="option"
                      aria-selected={e.key === selectedKey}
                      className={`cr-add-node-modal__row${e.key === selectedKey ? " cr-add-node-modal__row--active" : ""}`}
                      onMouseEnter={() => setSelectedKey(e.key)}
                      onClick={() => commitAdd(e)}
                    >
                      <LibraryNodeGlyph nodeType={e.nodeType} />
                      <span className="cr-add-node-modal__row-name">{e.label}</span>
                      <span className="cr-add-node-modal__row-tid">{e.nodeType}</span>
                    </button>
                  </li>
                ))
              )}
            </ul>
          </div>

          <aside className="cr-add-node-modal__detail" aria-live="polite">
            {selectedEntry ? (
              <>
                <div
                  className="cr-add-node-modal__preview"
                  style={{ "--accent": accentVarForCategory(GENERATED_NODE_SPECS[selectedEntry.nodeType]?.category) } as React.CSSProperties}
                >
                  <div className="cr-add-node-modal__preview-head">
                    <LibraryNodeGlyph nodeType={selectedEntry.nodeType} />
                    {selectedEntry.label}
                  </div>
                  <div className="cr-add-node-modal__preview-body">
                    {Object.entries(GENERATED_NODE_SPECS[selectedEntry.nodeType]?.defaults ?? {})
                      .slice(0, 3)
                      .map(([k, v]) => (
                        <div key={k} className="cr-add-node-modal__preview-row">
                          <span>{k}</span>
                          <span className="cr-add-node-modal__preview-v">{typeof v === "object" ? "…" : String(v)}</span>
                        </div>
                      ))}
                  </div>
                </div>
                <div className="cr-add-node-modal__detail-title">{selectedEntry.label}</div>
                <div className="cr-add-node-modal__detail-meta">{selectedEntry.categoryLabel}</div>
                <p className="cr-add-node-modal__detail-desc"><InfoRichText text={hint} /></p>
                <div className="cr-add-node-modal__detail-kv">
                  <span className="cr-add-node-modal__detail-k">Type id</span>
                  <span className="cr-add-node-modal__detail-v">{selectedEntry.nodeType}</span>
                </div>
              </>
            ) : (
              <p className="cr-add-node-modal__detail-placeholder">Select a node to see details.</p>
            )}
          </aside>
        </div>

        <div className="cr-add-node-modal__foot">
          <span className="cr-add-node-modal__key"><kbd>↑↓</kbd> select</span>
          <span className="cr-add-node-modal__key"><kbd>↵</kbd> add</span>
          <span className="cr-add-node-modal__key"><kbd>esc</kbd> close</span>
          <span className="cr-add-node-modal__count">
            {filteredFlat.length} / {flat.length} nodes
          </span>
        </div>
      </div>
    </div>
  );
}
