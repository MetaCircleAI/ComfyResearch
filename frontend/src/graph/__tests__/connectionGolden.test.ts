/** 连接矩阵金标门——canvas 波的 differential 等价物。
 * matrix:全型 × 72 (sh,th) 组合 × ioMode 变体(450 万探针, 扩针)sha 恒等;
 * cases:图态/副作用结构化快照恒等(id 归一 $newN)。
 * 更新口径:行为修复必须申报 expected-changed 并重跑
 * `npm run generate:connection-golden`;静默漂移即回归。 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { computeGolden, GOLDEN_PATH } from "../../../scripts/generate-connection-golden";

describe("connection rules golden", () => {
  it("matches the committed matrix sha and structured cases", { timeout: 60_000 }, () => {
    const committed = JSON.parse(readFileSync(resolve(GOLDEN_PATH), "utf8"));
    const current = computeGolden();
    expect(current.matrix.sha256).toBe(committed.matrix.sha256);
    expect(current.matrix.total).toBe(committed.matrix.total);
    expect(current.matrix.trueCount).toBe(committed.matrix.trueCount);
    expect(current.matrix.buckets).toEqual(committed.matrix.buckets);
    expect(current.cases).toEqual(committed.cases);
  });
});
