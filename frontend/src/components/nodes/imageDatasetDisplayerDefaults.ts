export type VisionSplit = "train" | "test";

export type ImageDatasetDisplayerNodeData = {
  instanceTitle?: string;
  split: VisionSplit;
  /** Inclusive range, e.g. `10-19` or a single index `5`. */
  indexRange: string;
  /** Thumbnails per row in the grid. */
  columnsPerRow: number;
};

export function defaultImageDatasetDisplayerData(
  partial?: Partial<ImageDatasetDisplayerNodeData>,
): ImageDatasetDisplayerNodeData {
  return {
    split: "train",
    indexRange: "0-9",
    columnsPerRow: 5,
    ...partial,
  };
}

export function parseInclusiveIndexRange(raw: string): { start: number; end: number } | null {
  const t = String(raw ?? "").trim();
  if (!t) return null;
  const dash = t.match(/^(\d+)\s*-\s*(\d+)$/);
  if (dash) {
    return { start: Number.parseInt(dash[1]!, 10), end: Number.parseInt(dash[2]!, 10) };
  }
  const one = t.match(/^(\d+)$/);
  if (one) {
    const v = Number.parseInt(one[1]!, 10);
    return { start: v, end: v };
  }
  return null;
}
