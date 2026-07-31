import { useLayoutEffect } from "react";
import { Handle, Position, useNodeId, useUpdateNodeInternals } from "@xyflow/react";

/**
 * Left: one «lr schedule» label with a single visible target; wiring picks ``lr_schedule`` vs
 * ``mup_lr_schedule`` by source node type. Ghost handles keep legacy edge anchors; right: «optimizer».
 */
export function OptimizerLrScheduleSocketRows() {
  const nodeId = useNodeId();
  const updateNodeInternals = useUpdateNodeInternals();
  useLayoutEffect(() => {
    if (nodeId) updateNodeInternals(nodeId);
  }, [nodeId, updateNodeInternals]);

  return (
    <div className="cr-trainer-io cr-optimizer-lr-io-strip" aria-label="Optimizer LR schedule inputs and output">
      <div className="cr-trainer-io-pair-grid">
        <div className="cr-trainer-io-row__leftwrap cr-model-init-source-strip__left cr-optimizer-lr-schedule-left">
          <div
            className="cr-optimizer-lr-schedule-merged-target"
            aria-label="LR schedule input (lr_schedule and mup_lr_schedule nodes)"
          >
            <Handle
              type="target"
              position={Position.Left}
              id="lr_schedule"
              className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--lr-schedule cr-optimizer-lr-schedule-ghost-target"
            />
            <Handle
              type="target"
              position={Position.Left}
              id="mup_lr_schedule"
              className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--lr-schedule cr-optimizer-lr-schedule-ghost-target"
            />
            <Handle
              type="target"
              position={Position.Left}
              id="optimizer_lr_schedule"
              title="Time-dependent LR: wire an lr_schedule node. μP multipliers (Adam): wire a mup_lr_schedule node. Both can attach here."
              className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--lr-schedule cr-optimizer-lr-schedule-main-target"
            />
          </div>
          <span className="cr-trainer-socket-label">lr schedule</span>
        </div>
        <div className="cr-trainer-io-row__rightwrap cr-model-init-source-strip__right">
          <span className="cr-trainer-output-label">optimizer</span>
          <Handle
            type="source"
            position={Position.Right}
            id="optimizer"
            className="cr-handle-source cr-handle-source--trainer-row"
          />
        </div>
      </div>
    </div>
  );
}
