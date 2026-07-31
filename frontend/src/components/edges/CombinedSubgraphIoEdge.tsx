import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from "@xyflow/react";
import { Position } from "@xyflow/react";

import { COMBINED_MODEL_RETURN_TARGET_HANDLE } from "../../graph/layerStripHandles";
import { SHAPE_CHECK_LABEL_DATA_KEY } from "./ResearchDefaultEdge";

/**
 * Auto-wired ``combined_model`` ↔ child tensor edges: XYFlow anchors use handle ``position`` (left/right
 * edge of the socket). Inward bezier tangents need the opposite ``Position``; we override only that here
 * so lines meet the true outer handles, not an offset “ghost” anchor.
 *
 * Edges that attach to a node with ``parentId`` get an elevated z-index above the shell parent; the default
 * wide interaction stroke (``BaseEdge``) would then sit on top of the shell’s tensor handles and block new
 * connections. We omit that layer (``interactionWidth={0}``); the visible stroke remains clickable.
 */
export function CombinedSubgraphIoEdge(props: EdgeProps) {
  const {
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    sourceHandleId,
    targetHandleId,
    label,
    labelStyle,
    labelShowBg,
    labelBgStyle,
    labelBgPadding,
    labelBgBorderRadius,
    style,
    markerStart,
    markerEnd,
    interactionWidth: _interactionWidth,
    data,
  } = props;

  const sourcePositionForPath =
    sourceHandleId === "tensor_boundary" ? Position.Right : sourcePosition;
  const targetPositionForPath =
    targetHandleId === COMBINED_MODEL_RETURN_TARGET_HANDLE ? Position.Left : targetPosition;

  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition: sourcePositionForPath,
    targetX,
    targetY,
    targetPosition: targetPositionForPath,
  });

  const raw = (data ?? {}) as Record<string, unknown>;
  const shapeLab =
    typeof raw[SHAPE_CHECK_LABEL_DATA_KEY] === "string" ? String(raw[SHAPE_CHECK_LABEL_DATA_KEY]) : null;

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        labelX={labelX}
        labelY={labelY}
        label={label}
        labelStyle={labelStyle}
        labelShowBg={labelShowBg}
        labelBgStyle={labelBgStyle}
        labelBgPadding={labelBgPadding}
        labelBgBorderRadius={labelBgBorderRadius}
        style={style}
        markerStart={markerStart}
        markerEnd={markerEnd}
        interactionWidth={0}
      />
      {shapeLab ? (
        <EdgeLabelRenderer>
          <div
            className="cr-shape-check-edge-label cr-shape-check-edge-label--combined-io nodrag nopan"
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: "none",
            }}
          >
            {shapeLab}
          </div>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}
