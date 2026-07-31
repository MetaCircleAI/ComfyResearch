import { useLayoutEffect } from "react";
import { Handle, Position, useNodeId, useUpdateNodeInternals } from "@xyflow/react";

import {
  COMBINED_MODEL_RETURN_TARGET_HANDLE,
  LAYER_STRIP_SOURCE_HANDLE,
  LAYER_STRIP_TARGET_HANDLE,
} from "../../graph/layerStripHandles";

/**
 * Paired tensor in/out (same ids as ``AtomicLayerIoStrip``), with extra sockets stacked invisibly
 * so XYFlow can keep auto-wired edges while the UI shows a single left and single right ``tensor`` dot.
 */
export function CombinedModelIoStrip({ isConnectable = true }: { isConnectable?: boolean }) {
  const nodeId = useNodeId();
  const updateNodeInternals = useUpdateNodeInternals();
  useLayoutEffect(() => {
    if (nodeId) updateNodeInternals(nodeId);
  }, [nodeId, updateNodeInternals]);

  return (
    <div className="cr-trainer-io" aria-label="Combined model tensor in, boundary out, and tensor out">
      <div className="cr-trainer-io-row cr-trainer-io-row--combined-model-io">
        <div className="cr-trainer-io-row__leftwrap cr-combined-model-io__left">
          <div
            className="cr-combined-model-left-handle-stack"
            title="Tensor in; inner chain is wired automatically on combine"
          >
            <Handle
              type="target"
              position={Position.Left}
              id={LAYER_STRIP_TARGET_HANDLE}
              isConnectable={isConnectable}
              className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
            />
            <Handle
              type="source"
              position={Position.Left}
              id="tensor_boundary"
              isConnectable={isConnectable}
              className="cr-handle-source cr-combined-model-boundary-handle cr-combined-model-boundary-handle--stacked"
            />
          </div>
          <span className="cr-trainer-socket-label">tensor</span>
        </div>
        <div className="cr-trainer-io-row__rightwrap cr-combined-model-io__right">
          <span className="cr-trainer-output-label">tensor</span>
          <div
            className="cr-combined-model-right-handle-stack"
            title="Tensor out; inner chain is wired automatically on combine"
          >
            <Handle
              type="source"
              position={Position.Right}
              id={LAYER_STRIP_SOURCE_HANDLE}
              isConnectable={isConnectable}
              className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--tensor"
            />
            <Handle
              type="target"
              position={Position.Right}
              id={COMBINED_MODEL_RETURN_TARGET_HANDLE}
              isConnectable={isConnectable}
              className="cr-handle-target cr-combined-model-return-handle cr-combined-model-return-handle--stacked"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
