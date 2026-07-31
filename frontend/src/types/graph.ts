import type { RegisteredNodeKind } from "../graph/nodeRegistrySpec";

export type TransitionalNodeKind =
  | "dataset"
  | "loss"
  | "model"
  | "observable"
  | "observable_viz_embedding_trajectory"
  | "observable_viz_relu_nonlinear"
  | "observable_viz_user"
  | "observable_viz_weight_l1"
  | "observable_viz_weight_l2"
  | "optimizer";

export type NodeKind = RegisteredNodeKind | TransitionalNodeKind;

export type Position = { x: number; y: number };

export type GraphNode = {
  id: string;
  type: NodeKind;
  position: Position;
  data: Record<string, unknown>;
  /** React Flow: subgraph parent (e.g. combined model wrapper). */
  parentId?: string | null;
  extent?: "parent" | null;
  hidden?: boolean | null;
  /** React Flow node shell size (`width` / `height`). */
  style?: Record<string, unknown> | null;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
};

export type Viewport = { x: number; y: number; zoom: number };

export type GraphDocument = {
  version: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  viewport?: Viewport | null;
};
