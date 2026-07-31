import { describe, expect, it } from "vitest";
import { appendResearchNode, readInstanceTitle } from "../nodeInstanceTitle";

describe("attention relation score instance titles", () => {
  it("pairs score0/score1 with viz0/viz1", () => {
    const score0 = appendResearchNode([], "observable_attention_relation_score", { x: 0, y: 0 }, {});
    const score1 = appendResearchNode([score0], "observable_attention_relation_score", { x: 0, y: 0 }, {});
    const viz0 = appendResearchNode([score0, score1], "observable_viz", { x: 0, y: 0 }, { pairedObservableId: score0.id });
    const viz1 = appendResearchNode([score0, score1, viz0], "observable_viz", { x: 0, y: 0 }, { pairedObservableId: score1.id });

    expect(readInstanceTitle(score0.data, "")).toBe("score0");
    expect(readInstanceTitle(score1.data, "")).toBe("score1");
    expect(readInstanceTitle(viz0.data, "")).toBe("viz0");
    expect(readInstanceTitle(viz1.data, "")).toBe("viz1");
  });
});
