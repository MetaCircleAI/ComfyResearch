import { useCallback } from "react";
import { useReactFlow, type NodeProps } from "@xyflow/react";
import type { CombinedModelNodeData } from "./combinedModelDefaults";
import { applyCombinedModelIoModeChange, defaultCombinedModelData } from "./combinedModelDefaults";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { ModelInitializationTargetRow } from "./ModelInitializationTargetRow";
import { CombinedModelIoStrip } from "./CombinedModelIoStrip";
import { NodeHeaderWithIoMode } from "./NodeCanvasIoModeSelect";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { readNodeCanvasIoMode } from "../../graph/nodeCanvasIoMode";
import type { NodeCanvasIoMode } from "../../graph/nodeCanvasIoMode";

export function CombinedModelNode({ id, data, selected }: NodeProps) {
  const defs = defaultCombinedModelData();
  const d = { ...defs, ...(data as Partial<CombinedModelNodeData>) } as CombinedModelNodeData;
  const { getNodes, getEdges, setNodes, setEdges } = useReactFlow();
  const ioMode = readNodeCanvasIoMode(d as Record<string, unknown>);

  const onIoModeChange = useCallback(
    (next: NodeCanvasIoMode, _prev: NodeCanvasIoMode) => {
      const { nodes: n2, edges: e2 } = applyCombinedModelIoModeChange(getNodes(), getEdges(), id, next);
      setNodes(() => n2);
      setEdges(() => e2);
    },
    [getNodes, getEdges, setNodes, setEdges, id],
  );

  return (
    <div
      className={`cr-node cr-node--combined-model${ioMode === "model" ? " cr-node--canvas-io-model" : ""}${selected ? " cr-node--selected" : ""}`}
      style={{
        ["--accent" as string]: "var(--cr-accent-model)",
        height: "100%",
        minHeight: "100%",
        boxSizing: "border-box",
      }}
    >
      <NodeHeaderWithIoMode id={id} data={d as Record<string, unknown>} onIoModeChange={onIoModeChange}>
        {readInstanceTitle(d, d.displayName?.trim() || "Combined model")}
      </NodeHeaderWithIoMode>
      <div className="cr-node__body">
        {ioMode === "model" ? (
          <ModelInitSourceSocketStrip sourceHandleId="tensor" sourceLabel="model" />
        ) : (
          <>
            <ModelInitializationTargetRow />
            <CombinedModelIoStrip />
          </>
        )}
      </div>
    </div>
  );
}
