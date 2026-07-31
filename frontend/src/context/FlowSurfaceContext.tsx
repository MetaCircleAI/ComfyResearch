import type { Edge, Node } from "@xyflow/react";
import { createContext, useContext, type Dispatch, type ReactNode, type SetStateAction } from "react";

/**
 * React Flow for one project canvas. Updates must target this surface's persisted graph even when the
 * user switches the active project tab while training streams results back.
 */
export type FlowSurfaceContextValue = {
  projectId: string;
  applyNodes: Dispatch<SetStateAction<Node[]>>;
  applyEdges: Dispatch<SetStateAction<Edge[]>>;
  /** Read persisted canvas graph — stays valid after React Flow unmounts (e.g. project tab switch). */
  getNodes: () => Node[];
  getEdges: () => Edge[];
};

const FlowSurfaceContext = createContext<FlowSurfaceContextValue | null>(null);

export function FlowSurfaceProvider({
  value,
  children,
}: {
  value: FlowSurfaceContextValue;
  children: ReactNode;
}) {
  return <FlowSurfaceContext.Provider value={value}>{children}</FlowSurfaceContext.Provider>;
}

/** Prefer this over ``useReactFlow().setNodes`` for mutations that can finish after a project switch. */
export function useFlowSurface(): FlowSurfaceContextValue | null {
  return useContext(FlowSurfaceContext);
}
