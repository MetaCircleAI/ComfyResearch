import { useLayoutEffect } from "react";
import { Handle, Position, useNodeId, useUpdateNodeInternals } from "@xyflow/react";

import { LAYER_STRIP_SOURCE_HANDLE, LAYER_STRIP_TARGET_HANDLE } from "../../graph/layerStripHandles";

/** Paired tensor in (left) and tensor out (right). Distinct handle ids are required so XYFlow registers both sockets. */
export function AtomicLayerIoStrip() {
  const nodeId = useNodeId();
  const updateNodeInternals = useUpdateNodeInternals();
  useLayoutEffect(() => {
    if (nodeId) updateNodeInternals(nodeId);
  }, [nodeId, updateNodeInternals]);

  return (
    <div className="cr-trainer-io" aria-label="Layer tensor in and tensor out">
      <div className="cr-trainer-io-row">
        <div className="cr-trainer-io-row__leftwrap">
          <Handle
            type="target"
            position={Position.Left}
            id={LAYER_STRIP_TARGET_HANDLE}
            className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
          />
          <span className="cr-trainer-socket-label">tensor</span>
        </div>
        <div className="cr-trainer-io-row__rightwrap">
          <span className="cr-trainer-output-label">tensor</span>
          <Handle
            type="source"
            position={Position.Right}
            id={LAYER_STRIP_SOURCE_HANDLE}
            className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
          />
        </div>
      </div>
    </div>
  );
}
