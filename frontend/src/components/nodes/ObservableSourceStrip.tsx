import { Handle, Position } from "@xyflow/react";

/**
 * Standard layout for metric observables: top socket row and gray divider via `.cr-trainer-io`.
 */
export function ObservableSourceStrip({
  sourceHandleId = "observables",
  withTargetObservable = false,
}: {
  sourceHandleId?: string;
  withTargetObservable?: boolean;
}) {
  return (
    <div className="cr-trainer-io" aria-label="Observable sockets">
      <div className="cr-trainer-io-row">
        {withTargetObservable ? (
          <div className="cr-trainer-io-row__leftwrap">
            <Handle
              type="target"
              position={Position.Left}
              id="observable_in"
              className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--tensor"
            />
            <span className="cr-trainer-socket-label">observable</span>
          </div>
        ) : (
          <div />
        )}
        <div className="cr-trainer-io-row__rightwrap">
          <span className="cr-trainer-output-label">observable</span>
          <Handle
            type="source"
            position={Position.Right}
            id={sourceHandleId}
            className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--observable"
          />
        </div>
      </div>
    </div>
  );
}
