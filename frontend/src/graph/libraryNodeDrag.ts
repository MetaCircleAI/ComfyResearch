const DRAGGING_CLASS = "cr-library-node-dragging";
const DELETE_TARGET_CLASS = "cr-node-over-library-delete";
const ACTIVE_NODE_CLASS = "cr-library-active-drag";
let isTrackingPointer = false;
let clipFrame: number | null = null;
let activeDragNodeId: string | null = null;
let mirrorFrame: number | null = null;
let dragNodeMirror: HTMLElement | null = null;
let mirrorSourceId: string | null = null;
let hiddenMirrorSource: HTMLElement | null = null;
let hiddenMirrorSourceVisibility: string | null = null;
let latestDragPointer: { x: number; y: number } | null = null;

function root() {
  return document.documentElement;
}

function trackPointer(event: PointerEvent) {
  updateLibraryNodeDragTarget(event.clientX, event.clientY);
}

/** Keep unrelated canvas nodes behind the rail while the dragged node crosses it. */
function clipNonDraggedNodesToCanvas() {
  clipFrame = null;
  const canvas = document.querySelector<HTMLElement>(".cr-canvas-wrap");
  if (!canvas) return;

  const canvasLeft = canvas.getBoundingClientRect().left;
  document.querySelectorAll<HTMLElement>(".react-flow__node").forEach((node) => {
    const isActive =
      node.dataset.id === activeDragNodeId ||
      node.classList.contains(ACTIVE_NODE_CLASS) ||
      node.classList.contains("cr-library-drag-draft");
    if (isActive) {
      node.style.removeProperty("clip-path");
      return;
    }

    const rect = node.getBoundingClientRect();
    const hiddenLeft = Math.min(Math.max(canvasLeft - rect.left, 0), rect.width);
    // clip-path uses the unscaled node box, whereas getBoundingClientRect()
    // includes the current React Flow zoom.
    const zoom = rect.width > 0 ? rect.width / node.offsetWidth : 1;
    const localInset = zoom > 0 ? hiddenLeft / zoom : hiddenLeft;
    if (localInset > 0) node.style.clipPath = `inset(0 0 0 ${localInset}px)`;
    else node.style.removeProperty("clip-path");
  });
}

function scheduleNodeClipping() {
  if (clipFrame !== null) return;
  clipFrame = window.requestAnimationFrame(clipNonDraggedNodesToCanvas);
}

function removeDragNodeMirror() {
  dragNodeMirror?.remove();
  dragNodeMirror = null;
  mirrorSourceId = null;
  if (hiddenMirrorSource) {
    if (hiddenMirrorSourceVisibility) hiddenMirrorSource.style.setProperty("visibility", hiddenMirrorSourceVisibility);
    else hiddenMirrorSource.style.removeProperty("visibility");
  }
  hiddenMirrorSource = null;
  hiddenMirrorSourceVisibility = null;
}

function copyInheritedFlowVariables(source: HTMLElement, mirror: HTMLElement) {
  const flow = source.closest<HTMLElement>(".react-flow");
  if (!flow) return;
  const styles = getComputedStyle(flow);
  for (let index = 0; index < styles.length; index += 1) {
    const name = styles[index]!;
    if (!name.startsWith("--")) continue;
    mirror.style.setProperty(name, styles.getPropertyValue(name));
  }
}

function syncMirrorControlState(source: HTMLElement, mirror: HTMLElement) {
  const sourceControls = source.querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(
    "input, textarea, select",
  );
  const mirrorControls = mirror.querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(
    "input, textarea, select",
  );
  sourceControls.forEach((control, index) => {
    const copy = mirrorControls[index];
    if (!copy) return;
    if (control instanceof HTMLInputElement && copy instanceof HTMLInputElement) {
      copy.value = control.value;
      copy.checked = control.checked;
    } else if (control instanceof HTMLTextAreaElement && copy instanceof HTMLTextAreaElement) {
      copy.value = control.value;
    } else if (control instanceof HTMLSelectElement && copy instanceof HTMLSelectElement) {
      copy.selectedIndex = control.selectedIndex;
    }
  });

  const sourceCanvases = source.querySelectorAll<HTMLCanvasElement>("canvas");
  const mirrorCanvases = mirror.querySelectorAll<HTMLCanvasElement>("canvas");
  sourceCanvases.forEach((canvas, index) => {
    const copy = mirrorCanvases[index];
    const context = copy?.getContext("2d");
    if (!copy || !context) return;
    copy.width = canvas.width;
    copy.height = canvas.height;
    try {
      context.drawImage(canvas, 0, 0);
    } catch {
      // A tainted third-party canvas cannot be copied; leave its clone blank.
    }
  });
}

/** The active real node has a hidden in-canvas view and a matching top-layer view. */
function syncDragNodeMirror() {
  mirrorFrame = null;
  if (!activeDragNodeId) {
    removeDragNodeMirror();
    return;
  }

  const source = [...document.querySelectorAll<HTMLElement>(".react-flow__node")].find(
    (node) => node.dataset.id === activeDragNodeId,
  );
  if (!source && dragNodeMirror && latestDragPointer) {
    Object.assign(dragNodeMirror.style, {
      left: `${latestDragPointer.x}px`,
      top: `${latestDragPointer.y}px`,
    });
    return;
  }
  if (!source) {
    removeDragNodeMirror();
    return;
  }

  const rect = source.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    removeDragNodeMirror();
    return;
  }

  if (!dragNodeMirror || mirrorSourceId !== activeDragNodeId || hiddenMirrorSource !== source) {
    removeDragNodeMirror();
    dragNodeMirror = source.cloneNode(true) as HTMLElement;
    dragNodeMirror.classList.add("cr-library-drag-node-mirror");
    dragNodeMirror.classList.remove(ACTIVE_NODE_CLASS);
    dragNodeMirror.classList.remove("cr-library-drag-source-hidden");
    dragNodeMirror.style.setProperty("visibility", "visible");
    dragNodeMirror.querySelectorAll("[id]").forEach((element) => element.removeAttribute("id"));
    copyInheritedFlowVariables(source, dragNodeMirror);
    syncMirrorControlState(source, dragNodeMirror);
    document.body.append(dragNodeMirror);
    // Let the mirror enter at its resting geometry, then animate only its
    // visual lift on the following frame. Its fixed-position tracking can
    // continue immediately while that animation runs.
    window.requestAnimationFrame(() => {
      dragNodeMirror?.classList.add("cr-library-drag-node-mirror--lifted");
    });
    mirrorSourceId = activeDragNodeId;
    hiddenMirrorSource = source;
    hiddenMirrorSourceVisibility = source.style.getPropertyValue("visibility") || null;
  }
  source.style.setProperty("visibility", "hidden");

  const sourceWidth = source.offsetWidth;
  const sourceHeight = source.offsetHeight;
  const viewportScale = sourceWidth > 0 ? rect.width / sourceWidth : 1;

  Object.assign(dragNodeMirror.style, {
    position: "fixed",
    left: `${rect.left}px`,
    top: `${rect.top}px`,
    width: `${sourceWidth}px`,
    height: `${sourceHeight}px`,
    margin: "0",
    maxWidth: "none",
    maxHeight: "none",
    transform: `scale(${viewportScale})`,
    transformOrigin: "top left",
    pointerEvents: "none",
    zIndex: "2147483647",
  });
}

function scheduleDragNodeMirror() {
  if (mirrorFrame !== null) return;
  mirrorFrame = window.requestAnimationFrame(syncDragNodeMirror);
}

export function beginLibraryNodeDrag() {
  root().classList.add(DRAGGING_CLASS);
  root().classList.remove(DELETE_TARGET_CLASS);
  scheduleNodeClipping();
  scheduleDragNodeMirror();
  if (!isTrackingPointer) {
    window.addEventListener("pointermove", trackPointer, { passive: true });
    isTrackingPointer = true;
  }
}

export function isOverNodesLibrary(clientX: number, clientY: number): boolean {
  const rail = document.querySelector<HTMLElement>(".cr-nodes-panel");
  if (!rail) return false;
  const rect = rail.getBoundingClientRect();
  return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom;
}

export function updateLibraryNodeDragTarget(clientX: number, clientY: number): boolean {
  latestDragPointer = { x: clientX, y: clientY };
  const overLibrary = isOverNodesLibrary(clientX, clientY);
  const pageRoot = root();
  pageRoot.classList.toggle(DELETE_TARGET_CLASS, overLibrary);
  if (overLibrary) {
    pageRoot.style.setProperty("--cr-library-delete-x", `${clientX}px`);
    pageRoot.style.setProperty("--cr-library-delete-y", `${clientY}px`);
  }
  scheduleNodeClipping();
  scheduleDragNodeMirror();
  return overLibrary;
}

/** Mark the one existing canvas node that is allowed to cross the Nodes rail. */
export function markLibraryDragNode(nodeId: string) {
  activeDragNodeId = nodeId;
  document.querySelectorAll<HTMLElement>(".react-flow__node").forEach((node) => {
    node.classList.toggle(ACTIVE_NODE_CLASS, node.dataset.id === nodeId);
  });
  scheduleNodeClipping();
  scheduleDragNodeMirror();
}

export function endLibraryNodeDrag() {
  const pageRoot = root();
  pageRoot.classList.remove(DRAGGING_CLASS, DELETE_TARGET_CLASS);
  pageRoot.style.removeProperty("--cr-library-delete-x");
  pageRoot.style.removeProperty("--cr-library-delete-y");
  activeDragNodeId = null;
  latestDragPointer = null;
  if (clipFrame !== null) {
    window.cancelAnimationFrame(clipFrame);
    clipFrame = null;
  }
  if (mirrorFrame !== null) {
    window.cancelAnimationFrame(mirrorFrame);
    mirrorFrame = null;
  }
  removeDragNodeMirror();
  document.querySelectorAll<HTMLElement>(".react-flow__node").forEach((node) => {
    node.classList.remove(ACTIVE_NODE_CLASS);
    node.style.removeProperty("clip-path");
  });
  if (isTrackingPointer) {
    window.removeEventListener("pointermove", trackPointer);
    isTrackingPointer = false;
  }
}
