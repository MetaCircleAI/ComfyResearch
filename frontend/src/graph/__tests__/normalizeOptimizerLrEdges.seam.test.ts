/** optimizer_lr_schedule 归一化的持久 handle 语义 seam。
 * family 只判"可接";persisted targetHandle 由本归一化显式决定——
 * cyclic 与 plain 同归 lr_schedule,mup 独归 mup_lr_schedule。 */
import { describe, expect, it } from "vitest";

import { normalizeOptimizerLrScheduleEdgeTargets } from "../normalizeOptimizerLrEdges";

const N = (id: string, type: string) => ({ id, type, position: { x: 0, y: 0 }, data: {} });
const E = (source: string, sh: string) => ({
  id: `e-${source}`, source, target: "opt", sourceHandle: sh, targetHandle: "optimizer_lr_schedule",
});

describe("normalizeOptimizerLrScheduleEdgeTargets", () => {
  it("cyclic and plain lr schedules persist as lr_schedule; mup stays mup", () => {
    const nodes = [
      N("a", "lr_schedule"), N("b", "cyclic_lr_schedule"), N("c", "mup_lr_schedule"), N("opt", "sgd_optimizer"),
    ] as never;
    const edges = [E("a", "lr_schedule"), E("b", "lr_schedule"), E("c", "mup_lr_schedule")] as never;
    const out = normalizeOptimizerLrScheduleEdgeTargets(edges, nodes);
    expect(out.map((e: { targetHandle?: string }) => e.targetHandle)).toEqual([
      "lr_schedule", "lr_schedule", "mup_lr_schedule",
    ]);
  });
});
