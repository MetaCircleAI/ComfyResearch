/** Canvas-level structural detail: single block vs expanded layer wiring (UI only for most nodes). */
export type NodeCanvasLevelMode = "high" | "low";

export function readNodeCanvasLevelMode(data: Record<string, unknown> | undefined): NodeCanvasLevelMode {
  const v = data?.levelMode;
  return v === "low" ? "low" : "high";
}
