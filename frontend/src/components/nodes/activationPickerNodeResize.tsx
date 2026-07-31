import { memo, useCallback, type ComponentType } from "react";
import { NodeResizeControl, useUpdateNodeInternals, type NodeProps } from "@xyflow/react";

/**
 * Picker-only resize wrapper: bottom-right handle, no loop badge / shape-check (unlike ``withNodeResize``).
 */
export function withActivationPickerNodeResize<P extends NodeProps>(Inner: ComponentType<P>): ComponentType<P> {
  const Wrapped = memo((props: P) => {
    const updateInternals = useUpdateNodeInternals();
    const id = props.id;
    const onResize = useCallback(() => {
      updateInternals(id);
    }, [id, updateInternals]);
    const onResizeEnd = useCallback(() => {
      updateInternals(id);
    }, [id, updateInternals]);

    return (
      <>
        {props.selected ? (
          <NodeResizeControl
            position="bottom-right"
            minWidth={72}
            minHeight={22}
            maxWidth={960}
            maxHeight={720}
            className="cr-node-resize-control nodrag nopan"
            onResize={onResize}
            onResizeEnd={onResizeEnd}
          />
        ) : null}
        <div className="cr-node-resize-root cr-activation-picker-node-resize-root">
          <Inner {...props} />
        </div>
      </>
    );
  });
  const innerName = Inner.displayName ?? Inner.name ?? "Node";
  Wrapped.displayName = `ActivationPickerResizable(${innerName})`;
  return Wrapped as ComponentType<P>;
}
