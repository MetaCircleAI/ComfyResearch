import type { Edge, Node } from "@xyflow/react";
import { createContext, useContext, Dispatch, ReactNode, SetStateAction } from "react";
import type { GraphDocument } from "../types/graph";

export type AddNodeOptions = {
  userObservableId?: string;
  label?: string;
  tensorVizNodeId?: string;
  tensorSelectorNodeId?: string;
  userLinearDatasetId?: string;
  userSymbolicFuncDatasetId?: string;
  combinedModelTemplateId?: string;
  combinedModelDisplayName?: string;
  combinedModelSourceNodeCount?: number;
  combinedModelTemplateDocument?: GraphDocument;
};

export type ResearchGraphActions = {
  addNode: (nodeType: string, screenPos?: { x: number; y: number }, options?: AddNodeOptions) => void;
  setFlowNodes?: Dispatch<SetStateAction<Node[]>>;
  setFlowEdges?: Dispatch<SetStateAction<Edge[]>>;
};

const ResearchGraphContext = createContext<ResearchGraphActions | null>(null);

export function ResearchGraphProvider({
  value,
  children,
}: {
  value: ResearchGraphActions;
  children: ReactNode;
}) {
  return <ResearchGraphContext.Provider value={value}>{children}</ResearchGraphContext.Provider>;
}

export function useResearchGraph(): ResearchGraphActions | null {
  return useContext(ResearchGraphContext);
}
