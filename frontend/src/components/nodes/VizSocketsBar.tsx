import { Handle, Position } from "@xyflow/react";
import type { ReactNode } from "react";

export type VizSocketsMode = "tensor" | "tensor_list";

/**
 * Viz nodes: left target (single `tensor` or list `tensor_list`), right sources
 * passthrough (`out_tensor` / `out_tensor_list`) and `comment` (→ Comment node).
 * Optional `middleRight` renders between tensor-out and comment (e.g. user observable on 0D viz).
 * Optional `annotatorSource` adds a Curve annotator source handle (viz → annotator).
 */
export function VizSocketsBar({
  mode = "tensor",
  middleRight,
  annotatorSource = false,
  streamSource = false,
  compareSource = false,
}: {
  mode?: VizSocketsMode;
  middleRight?: ReactNode;
  annotatorSource?: boolean;
  /** Training viz: rank-1 series out for curve series table. */
  streamSource?: boolean;
  /** Scalar observable line visualizations expose their resolved series to Metric compare. */
  compareSource?: boolean;
}) {
  const leftId = mode === "tensor_list" ? "tensor_list" : "tensor";
  const leftLabel = mode === "tensor_list" ? "tensor list" : "tensor";
  const outId = mode === "tensor_list" ? "out_tensor_list" : "out_tensor";
  const outLabel = mode === "tensor_list" ? "tensor list" : "tensor";

  return (
    <div className="cr-tviz-socket-row cr-tviz-socket-row--split">
      <div className="cr-tviz-socket-row__left">
        <Handle
          type="target"
          position={Position.Left}
          id={leftId}
          className="cr-handle-target cr-handle-target--tviz cr-handle-target--tviz-socket"
        />
        <span className="cr-tviz-socket-label">{leftLabel}</span>
      </div>
      <div className="cr-tviz-socket-row__right cr-tviz-socket-row__right--dual">
        <div className="cr-tviz-socket-pair">
          <span className="cr-tviz-socket-label">{outLabel}</span>
          <Handle
            type="source"
            position={Position.Right}
            id={outId}
            className="cr-handle-source cr-handle-source--trainer-row cr-handle-source--tviz-tensor-out"
          />
        </div>
        {streamSource ? (
          <div className="cr-tviz-socket-pair">
            <span className="cr-tviz-socket-label">stream</span>
            <Handle
              type="source"
              position={Position.Right}
              id="stream"
              className="cr-handle-source cr-handle-source--trainer-row cr-handle-source--tviz-tensor-out"
            />
          </div>
        ) : null}
        {annotatorSource ? (
          <div className="cr-tviz-socket-pair">
            <span className="cr-tviz-socket-label">annotate</span>
            <Handle
              type="source"
              position={Position.Right}
              id="annotator"
              className="cr-handle-source cr-handle-source--trainer-row cr-handle-source--tviz-annotator"
            />
          </div>
        ) : null}
        {compareSource ? (
          <div className="cr-tviz-socket-pair">
            <span className="cr-tviz-socket-label">compare</span>
            <Handle
              type="source"
              position={Position.Right}
              id="compare"
              className="cr-handle-source cr-handle-source--trainer-row"
            />
          </div>
        ) : null}
        {middleRight ? <div className="cr-tviz-socket-pair">{middleRight}</div> : null}
        <div className="cr-tviz-socket-pair">
          <span className="cr-tviz-socket-label">comment</span>
          <Handle
            type="source"
            position={Position.Right}
            id="comment"
            className="cr-handle-source cr-handle-source--trainer-row cr-handle-source--tviz-comment"
          />
        </div>
      </div>
    </div>
  );
}
