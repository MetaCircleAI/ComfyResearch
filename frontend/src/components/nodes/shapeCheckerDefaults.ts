export type ShapeCheckerNodeData = {
  shapeText: string | null;
  sourceSummary: string | null;
  lastError: string | null;
};

export function defaultShapeCheckerData(partial?: Partial<ShapeCheckerNodeData>): ShapeCheckerNodeData {
  return {
    shapeText: partial?.shapeText ?? null,
    sourceSummary: partial?.sourceSummary ?? null,
    lastError: partial?.lastError ?? null,
  };
}
