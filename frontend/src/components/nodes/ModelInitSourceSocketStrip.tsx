import { useLayoutEffect } from "react";
import { Handle, Position, useNodeId, useUpdateNodeInternals } from "@xyflow/react";

/** One I/O band: left «initialization» target + configurable right source (e.g. «model», handle ``tensor``). */
export function ModelInitSourceSocketStrip({
  sourceHandleId,
  sourceLabel,
  isConnectable = true,
}: {
  sourceHandleId: string;
  sourceLabel: string;
  isConnectable?: boolean;
}) {
  const nodeId = useNodeId();
  const updateNodeInternals = useUpdateNodeInternals();
  useLayoutEffect(() => {
    if (nodeId) updateNodeInternals(nodeId);
  }, [nodeId, sourceHandleId, updateNodeInternals]);

  return (
    <div className="cr-trainer-io cr-model-init-source-strip" aria-label="Initialization input and model output">
      <div className="cr-trainer-io-pair-grid">
        <div className="cr-trainer-io-row__leftwrap cr-model-init-source-strip__left">
          <Handle
            type="target"
            position={Position.Left}
            id="initialization"
            className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset cr-model-init-target"
          />
          <span className="cr-trainer-socket-label">initialization</span>
        </div>
        <div className="cr-trainer-io-row__rightwrap cr-model-init-source-strip__right">
          <span className="cr-trainer-output-label">{sourceLabel}</span>
          <Handle
            type="source"
            position={Position.Right}
            id={sourceHandleId}
            isConnectable={isConnectable}
            className="cr-handle-source cr-handle-source--trainer-row"
          />
        </div>
      </div>
    </div>
  );
}
