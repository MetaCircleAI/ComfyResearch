import type { NodeProps } from "@xyflow/react";

/**
 * Full-cell strike overlay when graph-assist training fails for one matrix slot.
 * Interaction is disabled via React Flow node flags and ``pointerEvents: none``.
 */
type FailureOverlayData = {
  phase?: string;
  reason?: string;
};

export function GraphAssistFailureOverlayNode(props: NodeProps) {
  const data = (props.data ?? {}) as FailureOverlayData;
  const phase = (data.phase ?? "").trim();
  const reason = (data.reason ?? "").trim();
  const footer = [phase, reason].filter((x) => x.length > 0).join(" - ");
  return (
    <div
      className="nodrag nopan"
      style={{
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        boxSizing: "border-box",
        position: "relative",
        borderRadius: 6,
        background: "rgba(12, 10, 18, 0.42)",
        boxShadow: "inset 0 0 0 2px rgba(220, 72, 72, 0.55)",
      }}
    >
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{ display: "block", position: "absolute", inset: 0 }}
        aria-hidden
      >
        <line
          x1="5"
          y1="5"
          x2="95"
          y2="95"
          stroke="rgba(255, 82, 82, 0.95)"
          strokeWidth="7"
          strokeLinecap="square"
        />
        <line
          x1="95"
          y1="5"
          x2="5"
          y2="95"
          stroke="rgba(255, 82, 82, 0.95)"
          strokeWidth="7"
          strokeLinecap="square"
        />
      </svg>
      {footer ? (
        <div
          style={{
            position: "absolute",
            left: 6,
            right: 6,
            bottom: 6,
            padding: "2px 6px",
            borderRadius: 4,
            fontSize: 10,
            lineHeight: 1.2,
            color: "#ffd4d4",
            background: "rgba(60, 20, 20, 0.78)",
            border: "1px solid rgba(255, 82, 82, 0.55)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
          title={footer}
        >
          {footer}
        </div>
      ) : null}
    </div>
  );
}
