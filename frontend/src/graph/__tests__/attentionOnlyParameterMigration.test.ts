import { describe, expect, it } from "vitest";
import { migrateAttentionOnlyParameterKey, migrateAttentionOnlyTensorSelectorData, migrateAttentionOnlyWeightTensorPayloads } from "../attentionOnlyParameterMigration";

describe("attention-only parameter migration", () => {
  it("updates legacy selector and payload keys", () => {
    expect(migrateAttentionOnlyParameterKey("w_q.weight")).toBe("block.w_q.weight");
    expect(migrateAttentionOnlyParameterKey("unembed.weight")).toBe("lm_head.weight");
    expect(migrateAttentionOnlyTensorSelectorData({
      selectedTensorKey: "w_o.bias",
      selectedTensorKeys: ["w_q.weight", "unembed.weight"],
    })).toMatchObject({
      selectedTensorKey: "block.w_o.bias",
      selectedTensorKeys: ["block.w_q.weight", "lm_head.weight"],
    });
    const migrated = migrateAttentionOnlyWeightTensorPayloads({
      weightTensorPayloads: {
        "w_q.weight": { shape: [2, 2] },
        "unembed.weight": { shape: [4, 2] },
      },
    });
    expect(migrated.weightTensorPayloads).toEqual({
      "block.w_q.weight": { shape: [2, 2] },
      "lm_head.weight": { shape: [4, 2] },
    });
  });
});
