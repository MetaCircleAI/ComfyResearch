import { useLayoutEffect } from "react";
import { Handle, Position, useNodeId, useUpdateNodeInternals } from "@xyflow/react";

/** Top I/O strip aligned with Trainer: label + right source handle, then gray rule via `.cr-trainer-io`. */
export function SourceSocketRow({
  handleId,
  label,
  isConnectable = true,
}: {
  handleId: string;
  label: string;
  isConnectable?: boolean;
}) {
  const nodeId = useNodeId();
  const updateNodeInternals = useUpdateNodeInternals();
  useLayoutEffect(() => {
    if (nodeId) updateNodeInternals(nodeId);
  }, [nodeId, handleId, updateNodeInternals]);

  return (
    <div className="cr-trainer-io" aria-label="Node output socket">
      <div className="cr-trainer-io-row cr-trainer-io-row--source-out">
        <div className="cr-trainer-io-row__rightwrap">
          <span className="cr-trainer-output-label">{label}</span>
          <Handle
            type="source"
            position={Position.Right}
            id={handleId}
            isConnectable={isConnectable}
            className="cr-handle-source cr-handle-source--trainer-row"
          />
        </div>
      </div>
    </div>
  );
}
