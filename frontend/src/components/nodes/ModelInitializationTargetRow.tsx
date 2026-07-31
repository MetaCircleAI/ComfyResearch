import { useLayoutEffect } from "react";
import { Handle, Position, useNodeId, useUpdateNodeInternals } from "@xyflow/react";

/** Left target «initialization» for μP init wiring (matches trainer socket styling). */
export function ModelInitializationTargetRow() {
  const nodeId = useNodeId();
  const updateNodeInternals = useUpdateNodeInternals();
  useLayoutEffect(() => {
    if (nodeId) updateNodeInternals(nodeId);
  }, [nodeId, updateNodeInternals]);

  return (
    <div className="cr-trainer-io" aria-label="Initialization input">
      <div className="cr-trainer-io-row">
        <div className="cr-trainer-io-row__leftwrap cr-trainer-io-row__leftwrap--full">
          <Handle
            type="target"
            position={Position.Left}
            id="initialization"
            className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset cr-model-init-target"
          />
          <span className="cr-trainer-socket-label">initialization</span>
        </div>
      </div>
    </div>
  );
}
