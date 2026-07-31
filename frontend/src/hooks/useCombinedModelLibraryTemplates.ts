import { useCallback, useEffect, useState } from "react";
import {
  GRAPH_COMBINED_MODEL_LIBRARY_CHANGED,
  GRAPH_TEMPLATE_LIBRARY_CHANGED,
} from "../dnd";
import { fetchSavedGraphLibrary, type SavedGraphEntry } from "../graph/savedGraphLibrary";

/** One row in the Nodes library → model list (workflow-backed; legacy rows may still use template storage). */
export type CombinedModelLibraryItem = {
  entry: SavedGraphEntry;
  storageKind: "workflow" | "template";
};

export function useCombinedModelLibraryTemplates(): CombinedModelLibraryItem[] {
  const [items, setItems] = useState<CombinedModelLibraryItem[]>([]);

  const load = useCallback(() => {
    void Promise.all([fetchSavedGraphLibrary("workflow"), fetchSavedGraphLibrary("template")])
      .then(([w, t]) => {
        const wf = w.filter((e) => e.libraryOrigin === "combined_model");
        const tf = t.filter((e) => e.libraryOrigin === "combined_model");
        const wfIds = new Set(wf.map((e) => e.id));
        const merged: CombinedModelLibraryItem[] = [
          ...wf.map((entry) => ({ entry, storageKind: "workflow" as const })),
          ...tf
            .filter((e) => !wfIds.has(e.id))
            .map((entry) => ({ entry, storageKind: "template" as const })),
        ];
        merged.sort((a, b) => b.entry.savedAt - a.entry.savedAt);
        setItems(merged);
      })
      .catch(() => setItems([]));
  }, []);

  useEffect(() => {
    load();
    const onLib = () => load();
    window.addEventListener(GRAPH_TEMPLATE_LIBRARY_CHANGED, onLib);
    window.addEventListener(GRAPH_COMBINED_MODEL_LIBRARY_CHANGED, onLib);
    return () => {
      window.removeEventListener(GRAPH_TEMPLATE_LIBRARY_CHANGED, onLib);
      window.removeEventListener(GRAPH_COMBINED_MODEL_LIBRARY_CHANGED, onLib);
    };
  }, [load]);

  return items;
}
