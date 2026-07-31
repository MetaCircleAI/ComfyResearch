import { memo, useCallback, type ComponentType } from "react";
import { NodeResizeControl, useUpdateNodeInternals, type NodeProps } from "@xyflow/react";
import { useShapeCheckOverlay } from "../../context/shapeCheckOverlayContext";
import { readGraphNodeLoopCount } from "../../graph/nodeLoopCount";

/**
 * Wraps any custom node so it can be resized from the bottom-right only.
 * React Flow keeps the opposite corner (top-left) fixed while resizing.
 */
export function withNodeResize<P extends NodeProps>(Inner: ComponentType<P>): ComponentType<P> {
  const Wrapped = memo((props: P) => {
    const updateInternals = useUpdateNodeInternals();
    const id = props.id;
    const onResize = useCallback(() => {
      updateInternals(id);
    }, [id, updateInternals]);
    const onResizeEnd = useCallback(() => {
      updateInternals(id);
    }, [id, updateInternals]);

    const loopN = readGraphNodeLoopCount(props.data);
    const showLoopBadge = loopN != null && loopN >= 2;
    const shapeCheck = useShapeCheckOverlay();
    const showShapeCheckError = shapeCheck?.errorNodeIds.has(props.id) ?? false;

    return (
      <>
        {props.selected ? (
          <NodeResizeControl
            position="bottom-right"
            minWidth={120}
            minHeight={36}
            maxWidth={4096}
            maxHeight={4096}
            className="cr-node-resize-control nodrag nopan"
            onResize={onResize}
            onResizeEnd={onResizeEnd}
          />
        ) : null}
        <div className="cr-node-resize-root">
          {showLoopBadge ? (
            <div className="cr-node-loop-badge nodrag nopan" aria-hidden>
              <span className="cr-node-loop-badge__icon" title={`Loop ×${loopN}`}>
                ⟲
              </span>
              <span className="cr-node-loop-badge__times">×{loopN}</span>
            </div>
          ) : null}
          {showShapeCheckError ? (
            <div className="cr-shape-check-node-x nodrag nopan" role="img" aria-label="Shape check error">
              <span className="cr-shape-check-node-x__glyph" aria-hidden>
                ✕
              </span>
            </div>
          ) : null}
          <Inner {...props} />
        </div>
      </>
    );
  });
  const innerName = Inner.displayName ?? Inner.name ?? "Node";
  Wrapped.displayName = `Resizable(${innerName})`;
  return Wrapped as ComponentType<P>;
}
