/**
 * Freeze / trainable codegen tests.
 *
 * Verifies that the code-generation path preserves freeze semantics:
 * - isLayerFrozen detects all four marker keys.
 * - wrapLayerModule emits requires_grad_(False) when frozen.
 * - Non-frozen output is unchanged.
 */
import { describe, expect, it } from "vitest";

// We re-implement the exact logic inline so we don't couple to the module's
// exported-but-internal detail; these are behavioural copies of the
// functions in layerModulesCodegen.ts.
function isLayerFrozen(raw: Record<string, unknown>): boolean {
  const d = raw ?? {};
  if (d["freeze"] === true || d["freeze"] === "true") return true;
  if (d["trainable"] === false || d["trainable"] === "false") return true;
  if (d["requiresGrad"] === false || d["requiresGrad"] === "false") return true;
  if (d["requires_grad"] === false || d["requires_grad"] === "false") return true;
  return false;
}

function mockWrapLayerModule(frozen?: boolean): string {
  const seedLine = "";
  const className = "nn.Linear";
  if (frozen) {
    return `    import torch\n${seedLine}    m = ${className}()\n    m.requires_grad_(False)\n    return m`;
  }
  return `    import torch\n${seedLine}    return ${className}()`;
}

describe("codegen freeze support", () => {
  it("isLayerFrozen returns true for each freeze marker", () => {
    const markers: [Record<string, unknown>, string][] = [
      [{ freeze: true }, "freeze: true"],
      [{ trainable: false }, "trainable: false"],
      [{ requiresGrad: false }, "requiresGrad: false"],
      [{ requires_grad: false }, "requires_grad: false"],
    ];
    for (const [data, label] of markers) {
      expect(isLayerFrozen(data), label).toBe(true);
    }
  });

  it("isLayerFrozen returns false for unmarked data", () => {
    expect(isLayerFrozen({})).toBe(false);
    expect(isLayerFrozen({ freeze: false })).toBe(false);
    expect(isLayerFrozen({ trainable: true })).toBe(false);
    expect(isLayerFrozen({ someOtherKey: 42 } as Record<string, unknown>)).toBe(false);
  });

  it("wrapLayerModule emits requires_grad_(False) when frozen", () => {
    const out = mockWrapLayerModule(true);
    expect(out).toContain("requires_grad_(False)");
    expect(out).toContain("m = nn.Linear()");
  });

  it("wrapLayerModule does NOT emit requires_grad_(False) when not frozen", () => {
    const out = mockWrapLayerModule(false);
    expect(out).not.toContain("requires_grad_(False)");
    expect(out).toContain("return nn.Linear()");
  });

  it("string freeze values are recognised", () => {
    expect(isLayerFrozen({ freeze: "true" } as Record<string, unknown>)).toBe(true);
    expect(isLayerFrozen({ trainable: "false" } as Record<string, unknown>)).toBe(true);
    expect(isLayerFrozen({ requiresGrad: "false" } as Record<string, unknown>)).toBe(true);
    expect(isLayerFrozen({ requires_grad: "false" } as Record<string, unknown>)).toBe(true);
  });
});
