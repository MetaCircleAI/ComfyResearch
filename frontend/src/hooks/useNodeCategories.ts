import { useCallback, useEffect, useState } from "react";
import {
  USER_LINEAR_DATASETS_CHANGED,
  USER_OBSERVABLES_CHANGED,
  USER_SYMBOLIC_FUNC_DATASETS_CHANGED,
} from "../dnd";
import { GENERATED_NODE_SPECS } from "../generated/generatedNodeSpecs";

export type CatalogNodeChild = {
  id: string;
  label: string;
  user_observable_id?: string;
  user_linear_dataset_id?: string;
  user_symbolic_func_dataset_id?: string;
  tensor_viz_node_id?: string;
  tensor_selector_node_id?: string;
  deletable?: boolean;
};

export type CatalogNodeCategory = { id: string; label: string; children?: CatalogNodeChild[] };

type CatalogResponse = { version: number; categories: CatalogNodeCategory[] };

type CatalogMutationDetail = {
  deletedUserObservableId?: string;
  deletedLinearDatasetId?: string;
  deletedSymbolicFuncDatasetId?: string;
};

const CATALOG_CATEGORY_ORDER = [
  "dataset",
  "optimizer",
  "model",
  "loss",
  "observables",
  "training",
  "checkpoint",
  "visualization",
  "language",
  "analysis",
] as const;

/**
 * The Python API normally supplies the sidebar catalog. Keep a local copy for
 * UI-only work so the node palette remains usable when the backend is offline.
 */
const LOCAL_NODE_CATEGORIES: CatalogNodeCategory[] = CATALOG_CATEGORY_ORDER.map((category) => {
  const children = Object.entries(GENERATED_NODE_SPECS)
    .filter(([, spec]) => spec.category === category)
    .filter(([id]) => id !== "combined_model" && id !== "observable_user")
    .map(([id, spec]) => ({ id, label: spec.label }))
    .sort((a, b) => a.label.localeCompare(b.label));

  return { id: category, label: category, children };
}).filter((category) => category.children?.length);

function removeCatalogChildren(
  prev: CatalogNodeCategory[],
  match: (ch: CatalogNodeChild) => boolean,
): CatalogNodeCategory[] {
  return prev.map((cat) => ({
    ...cat,
    children: cat.children?.filter((ch) => !match(ch)),
  }));
}

export function useNodeCategories(): CatalogNodeCategory[] {
  const [categories, setCategories] = useState<CatalogNodeCategory[]>([]);

  const load = useCallback(() => {
    void fetch("/api/node-categories", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data: CatalogResponse) => {
        if (Array.isArray(data.categories)) {
          setCategories(data.categories);
        }
      })
      .catch(() => {
        setCategories(LOCAL_NODE_CATEGORIES);
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const onObservables = (ev: Event) => {
      const d = (ev as CustomEvent<CatalogMutationDetail>).detail;
      if (d?.deletedUserObservableId) {
        const id = d.deletedUserObservableId;
        setCategories((prev) =>
          removeCatalogChildren(prev, (ch) => ch.user_observable_id === id),
        );
        return;
      }
      void load();
    };
    const onLinear = (ev: Event) => {
      const d = (ev as CustomEvent<CatalogMutationDetail>).detail;
      if (d?.deletedLinearDatasetId) {
        const id = d.deletedLinearDatasetId;
        setCategories((prev) =>
          removeCatalogChildren(prev, (ch) => ch.user_linear_dataset_id === id),
        );
        return;
      }
      void load();
    };
    const onSymbolic = (ev: Event) => {
      const d = (ev as CustomEvent<CatalogMutationDetail>).detail;
      if (d?.deletedSymbolicFuncDatasetId) {
        const id = d.deletedSymbolicFuncDatasetId;
        setCategories((prev) =>
          removeCatalogChildren(prev, (ch) => ch.user_symbolic_func_dataset_id === id),
        );
        return;
      }
      void load();
    };
    window.addEventListener(USER_OBSERVABLES_CHANGED, onObservables);
    window.addEventListener(USER_LINEAR_DATASETS_CHANGED, onLinear);
    window.addEventListener(USER_SYMBOLIC_FUNC_DATASETS_CHANGED, onSymbolic);
    return () => {
      window.removeEventListener(USER_OBSERVABLES_CHANGED, onObservables);
      window.removeEventListener(USER_LINEAR_DATASETS_CHANGED, onLinear);
      window.removeEventListener(USER_SYMBOLIC_FUNC_DATASETS_CHANGED, onSymbolic);
    };
  }, [load]);

  return categories;
}
