import { Handle, Position } from "@xyflow/react";

/** Single dataset output: train and test splits share the same node config (sizes in ``data``). */
export function DatasetSourceSockets() {
  return (
    <div className="cr-trainer-io" aria-label="Dataset output">
      <div className="cr-trainer-io-row cr-trainer-io-row--source-out">
        <div className="cr-trainer-io-row__rightwrap">
          <span className="cr-trainer-output-label">dataset</span>
          <Handle
            type="source"
            position={Position.Right}
            id="dataset"
            className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--dataset"
          />
        </div>
      </div>
    </div>
  );
}
