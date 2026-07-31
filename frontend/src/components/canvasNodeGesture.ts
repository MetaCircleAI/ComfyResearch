/**
 * Press-time gesture classification for canvas nodes.
 *
 * Elements matching this selector are app interactions: a pointerdown inside
 * them is never a node drag/select gesture. Everything else inside a node —
 * including bare `.nodrag` parameter-label shells, which XYFlow declines but
 * the canvas moves through its manual drag fallback — remains a node gesture.
 *
 * `.cr-tviz-chart` is the drag-to-zoom plot surface shared by every line
 * chart; without this exemption the manual `.nodrag` fallback turns it back
 * into a drag handle and the zoom selection can never complete (issue #169).
 */
export const NODE_PRESS_INTERACTIVE_SELECTOR = [
  "input",
  "textarea",
  "select",
  "[contenteditable='true']",
  ".react-flow__handle",
  ".react-flow__resize-control",
  ".cr-node__header button",
  ".cr-tviz-chart",
].join(", ");

export function isInteractiveNodePressTarget(target: Element): boolean {
  return Boolean(target.closest(NODE_PRESS_INTERACTIVE_SELECTOR));
}
