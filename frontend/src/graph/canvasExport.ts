import { toPng } from "html-to-image";
import { jsPDF } from "jspdf";

/**
 * Keep ``html-to-image`` on **1.11.11** in package.json: 1.11.12+ drops React Flow / xyflow SVG edges
 * (marker defs) in ``toPng`` output — see https://github.com/bubkoo/html-to-image/issues/496
 */

/** Matches the main React Flow instance in the workbench (excludes minimap / controls panels). */
const FLOW_RENDERER_SELECTOR = ".cr-canvas-wrap .react-flow__renderer";

function queryFlowRenderer(): HTMLElement | null {
  return document.querySelector(FLOW_RENDERER_SELECTOR);
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error ?? new Error("read failed"));
    r.readAsDataURL(blob);
  });
}

function imageNaturalSizeFromDataUrl(dataUrl: string): Promise<{ w: number; h: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => reject(new Error("Could not read exported image dimensions."));
    img.src = dataUrl;
  });
}

/** Rasterize the flow renderer (nodes, edges, dotted background) to PNG bytes. */
export async function captureFlowRendererToPngBlob(): Promise<Blob> {
  const el = queryFlowRenderer();
  if (!el) {
    throw new Error("Canvas is not ready to export (flow renderer not found).");
  }
  const pixelRatio = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
  const dataUrl = await toPng(el, {
    pixelRatio,
    cacheBust: true,
    backgroundColor: "#18181f",
  });
  const res = await fetch(dataUrl);
  if (!res.ok) throw new Error("PNG export failed.");
  return await res.blob();
}

/** Single-page PDF with the PNG scaled to one page (size matches image). */
export async function pngBlobToSinglePagePdfBlob(pngBlob: Blob): Promise<Blob> {
  const dataUrl = await blobToDataUrl(pngBlob);
  const { w, h } = await imageNaturalSizeFromDataUrl(dataUrl);
  if (w < 1 || h < 1) throw new Error("Exported image has no size.");
  const MM_PER_PX = 25.4 / 96;
  const fmtW = w * MM_PER_PX;
  const fmtH = h * MM_PER_PX;
  const pdf = new jsPDF({
    orientation: fmtW >= fmtH ? "landscape" : "portrait",
    unit: "mm",
    format: [fmtW, fmtH],
  });
  pdf.addImage(dataUrl, "PNG", 0, 0, fmtW, fmtH);
  return pdf.output("blob");
}

type ShowSaveFilePickerFn = (options?: {
  suggestedName?: string;
  types?: Array<{ description?: string; accept: Record<string, string[]> }>;
}) => Promise<FileSystemFileHandle>;

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export type BlobSaveOutcome = "saved" | "cancelled";

export async function saveBlobWithUserLocation(
  blob: Blob,
  suggestedName: string,
  description: string,
  accept: Record<string, string[]>,
): Promise<BlobSaveOutcome> {
  const showSave = (window as unknown as { showSaveFilePicker?: ShowSaveFilePickerFn }).showSaveFilePicker;
  if (typeof showSave === "function") {
    try {
      const handle = await showSave({
        suggestedName,
        types: [{ description, accept }],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return "saved";
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return "cancelled";
      downloadBlob(blob, suggestedName);
      return "saved";
    }
  }
  downloadBlob(blob, suggestedName);
  return "saved";
}
