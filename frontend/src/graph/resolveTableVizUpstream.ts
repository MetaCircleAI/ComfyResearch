import type { Edge, Node } from "@xyflow/react";
import type { SweepDataTableRow } from "../components/nodes/sweepDataTableDefaults";

export type ResolvedSweepTable = {
  rows: SweepDataTableRow[];
  /** Mirrors sweep data table selection. */
  selectedRowIds: string[] | null;
  /** Mirrors sweep data table column order for row sorting. */
  paramKeyOrder: string[] | null;
};

/**
 * Read rows from the sweep data table connected to this table viz (`table` → `table`).
 */
export function resolveTableVizUpstream(
  nodes: Node[],
  edges: Edge[],
  tableVizId: string,
): ResolvedSweepTable | null {
  const inc = edges.find((e) => e.target === tableVizId && (e.targetHandle ?? "") === "table");
  if (!inc?.source) return null;
  const src = nodes.find((n) => n.id === inc.source);
  if (!src) return null;
  const t = String(src.type);
  if (t !== "sweep_data_table" && t !== "sweep_viz") return null;
  const d = (src.data ?? {}) as {
    rows?: unknown;
    selectedRowIds?: unknown;
    plotSelectedRowIds?: unknown;
    paramKeyOrder?: unknown;
  };
  const rows = Array.isArray(d.rows) ? (d.rows as SweepDataTableRow[]) : [];
  const selectedRowIds =
    d.selectedRowIds !== undefined
      ? (d.selectedRowIds as string[] | null)
      : d.plotSelectedRowIds !== undefined
        ? (d.plotSelectedRowIds as string[] | null)
        : null;
  const paramKeyOrder = Array.isArray(d.paramKeyOrder) ? (d.paramKeyOrder as string[]) : null;
  return { rows, selectedRowIds, paramKeyOrder };
}
