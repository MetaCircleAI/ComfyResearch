import type { Edge, Node } from "@xyflow/react";
import { defaultMlpModelData } from "../src/components/nodes/mlpModelDefaults";
import { reconcileMlpLowExpansion } from "../src/graph/mlpLowLevelExpansion";

function assert(cond: unknown, message: string) {
  if (!cond) throw new Error(message);
}

const mlpId = "mlp-test";
const data = { ...defaultMlpModelData(), levelMode: "low" as const, ioMode: "input-output" as const };
const shell: Node = {
  id: mlpId,
  type: "mlp_model",
  position: { x: 0, y: 0 },
  data,
};

const first = reconcileMlpLowExpansion([shell], [], mlpId, data);
assert(first.nodes.length > 1, "first reconcile should expand graph");
assert(first.edges.length > 0, "first reconcile should add edges");

const second = reconcileMlpLowExpansion(first.nodes, first.edges, mlpId, data);
assert(second.nodes === first.nodes, "second reconcile should return same nodes ref");
assert(second.edges === first.edges, "second reconcile should return same edges ref");

console.log("verify-mlp-low-expansion-idempotent: ok");
