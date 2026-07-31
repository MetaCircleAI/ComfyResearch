import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  useStore,
  useViewport,
  type ConnectionLineComponentProps,
  type EdgeProps,
} from "@xyflow/react";
import { useContext } from "react";
import { ConnectionSnapContext } from "./connectionSnapContext";

/** Transient edge `data` key for fake-tensor shape check labels (not persisted in graph JSON). */
export const SHAPE_CHECK_LABEL_DATA_KEY = "__shapeCheckLabel" as const;

const DRAG_LIFT_Y = -4;
const DRAG_LIFT_SCALE = 1.012;

type LiftedNode = {
  dragging?: boolean;
  measured?: { width?: number; height?: number };
  internals?: { positionAbsolute?: { x: number; y: number } };
};

function getLiftedEndpoint(x: number, y: number, node: LiftedNode | undefined) {
  if (!node?.dragging) return { x, y };

  const width = node.measured?.width ?? 0;
  const height = node.measured?.height ?? 0;
  const position = node.internals?.positionAbsolute;
  if (!position || width <= 0 || height <= 0) return { x, y: y + DRAG_LIFT_Y };

  const centerX = position.x + width / 2;
  const centerY = position.y + height / 2;
  return {
    x: centerX + (x - centerX) * DRAG_LIFT_SCALE,
    y: centerY + (y - centerY) * DRAG_LIFT_SCALE + DRAG_LIFT_Y,
  };
}

function getRenderedHandleCenter(
  nodeId: string | undefined,
  handleId: string | null | undefined,
  handleType: string | undefined,
  viewport: { x: number; y: number; zoom: number },
) {
  if (typeof document === "undefined" || !nodeId || !handleType || viewport.zoom <= 0) return null;

  const node = [...document.querySelectorAll<HTMLElement>(".react-flow__node")].find(
    (element) => element.dataset.id === nodeId,
  );
  const expectedHandleId = handleId ?? undefined;
  const handle = node && [...node.querySelectorAll<HTMLElement>(".react-flow__handle")].find(
    (element) => element.dataset.handleid === expectedHandleId && element.classList.contains(handleType),
  );
  const flow = node?.closest<HTMLElement>(".react-flow");
  if (!handle || !flow) return null;

  const handleRect = handle.getBoundingClientRect();
  const flowRect = flow.getBoundingClientRect();
  return {
    x: (handleRect.left + handleRect.width / 2 - flowRect.left - viewport.x) / viewport.zoom,
    y: (handleRect.top + handleRect.height / 2 - flowRect.top - viewport.y) / viewport.zoom,
  };
}

/**
 * Wire tint family derived from the source handle id. Styled only under the
 * studio theme (studio.css); classic keeps the default uniform stroke.
 */
function edgeTypeClass(sourceHandleId: string | null | undefined): string {
  const h = (sourceHandleId ?? "").toLowerCase();
  if (h.includes("dataset")) return "cr-edge-t-dataset";
  if (h.includes("checkpoint") || h.includes("model")) return "cr-edge-t-model";
  if (h.includes("optimizer") || h.includes("lr") || h.includes("batch")) return "cr-edge-t-optimizer";
  if (h.includes("loss")) return "cr-edge-t-loss";
  if (h.includes("observable")) return "cr-edge-t-observable";
  if (h.includes("tensor") || h.includes("out")) return "cr-edge-t-tensor";
  return "cr-edge-t-default";
}

export function ResearchDefaultEdge(props: EdgeProps) {
  const {
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style,
    markerStart,
    markerEnd,
    interactionWidth,
    data,
    source,
    target,
  } = props;

  const sourceNode = useStore((state) => state.nodeLookup.get(source));
  const targetNode = useStore((state) => state.nodeLookup.get(target));
  const liftedSource = getLiftedEndpoint(sourceX, sourceY, sourceNode);
  const liftedTarget = getLiftedEndpoint(targetX, targetY, targetNode);

  const [path, labelX, labelY] = getBezierPath({
    sourceX: liftedSource.x,
    sourceY: liftedSource.y,
    sourcePosition,
    targetX: liftedTarget.x,
    targetY: liftedTarget.y,
    targetPosition,
  });

  // Signature detail: while either endpoint trainer is actively training,
  // overlay a flowing dash along the wire (styled only under studio; with
  // fill=none and no stroke rule the overlay is invisible in classic).
  type TrainDataShape = {
    trainUi?: { active?: boolean; loading?: boolean; paused?: boolean };
    hostTrainUi?: { active?: boolean };
  };
  const isTraining = (n: unknown): boolean => {
    const node = n as { type?: string; data?: TrainDataShape } | undefined;
    if (node?.type !== "trainer") return false;
    const ui = node.data?.trainUi;
    if (ui?.paused) return false;
    // Local/persisted runs set trainUi.active+loading; host-assisted (AutoDL)
    // runs may drive progress via hostTrainUi.active instead.
    return Boolean((ui?.active && ui?.loading) || node.data?.hostTrainUi?.active);
  };
  const flowing = isTraining(sourceNode) || isTraining(targetNode);

  const raw = (data ?? {}) as Record<string, unknown>;
  const lab = typeof raw[SHAPE_CHECK_LABEL_DATA_KEY] === "string" ? String(raw[SHAPE_CHECK_LABEL_DATA_KEY]) : null;

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={style}
        className={edgeTypeClass(props.sourceHandleId)}
        markerStart={markerStart}
        markerEnd={markerEnd}
        interactionWidth={interactionWidth}
      />
      {flowing ? (
        <>
          <path d={path} fill="none" className={`cr-edge-flow ${edgeTypeClass(props.sourceHandleId)}`} />
          <circle
            cx={liftedTarget.x}
            cy={liftedTarget.y}
            r={3}
            fill="none"
            className={`cr-edge-arrival ${edgeTypeClass(props.sourceHandleId)}`}
          />
        </>
      ) : null}
      {lab ? (
        <EdgeLabelRenderer>
          <div
            className="cr-shape-check-edge-label nodrag nopan"
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: "none",
            }}
          >
            {lab}
          </div>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}

/**
 * The first connection after a node is created can begin before XYFlow has
 * written that node's handle bounds into its store. Read the rendered handles
 * for the temporary preview only; established edges keep using store geometry.
 */
export function ResearchConnectionLine({
  fromNode,
  fromHandle,
  fromX,
  fromY,
  fromPosition,
  toNode,
  toHandle,
  toX,
  toY,
  toPosition,
  connectionStatus,
  connectionLineStyle,
}: ConnectionLineComponentProps) {
  const viewport = useViewport();
  const snappedTarget = useContext(ConnectionSnapContext);
  const source = getRenderedHandleCenter(fromNode.id, fromHandle.id, fromHandle.type, viewport) ?? { x: fromX, y: fromY };
  const target = snappedTarget
    ? { x: snappedTarget.x, y: snappedTarget.y }
    : connectionStatus === "valid" && toNode && toHandle
    ? getRenderedHandleCenter(toNode.id, toHandle.id, toHandle.type, viewport) ?? { x: toX, y: toY }
    : { x: toX, y: toY };
  const [path] = getBezierPath({
    sourceX: source.x,
    sourceY: source.y,
    sourcePosition: fromPosition,
    targetX: target.x,
    targetY: target.y,
    targetPosition: toPosition,
  });

  return <path d={path} fill="none" className="react-flow__connection-path" style={connectionLineStyle} />;
}
