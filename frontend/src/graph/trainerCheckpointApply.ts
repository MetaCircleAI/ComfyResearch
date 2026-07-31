import type { Node } from "@xyflow/react";
import type { Dispatch, SetStateAction } from "react";
import type { CrlTrainerNodeData } from "../components/nodes/crlTrainerDefaults";
import type { ModelCheckpointNodeData } from "../components/nodes/modelCheckpointDefaults";
import type { TrainerNodeData } from "../components/nodes/trainerDefaults";

export type TrainEdgeWire = {
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
};

/** Same wiring rules as ``TrainerNode`` checkpoint fan-out. */
export function applyCheckpointToModelCheckpointNodes(
  setNodes: Dispatch<SetStateAction<Node[]>>,
  trainerNodeId: string,
  checkpointB64: string,
  edges: TrainEdgeWire[],
): void {
  if (!checkpointB64) return;
  const targets = new Set<string>();
  for (const e of edges) {
    if (e.source !== trainerNodeId) continue;
    const sh = e.sourceHandle ?? null;
    if (sh != null && sh !== "checkpoint") continue;
    if ((e.targetHandle ?? null) !== "model_checkpoint") continue;
    targets.add(e.target);
  }
  setNodes((prev) =>
    prev.map((n) => {
      if (n.id === trainerNodeId && n.type === "trainer") {
        const prevData = (n.data ?? {}) as Partial<TrainerNodeData>;
        return {
          ...n,
          data: { ...prevData, memoryCheckpoint_b64: checkpointB64 },
        };
      }
      if (!targets.has(n.id) || n.type !== "model_checkpoint") return n;
      const prevData = (n.data ?? {}) as Partial<ModelCheckpointNodeData>;
      return {
        ...n,
        data: {
          ...prevData,
          checkpoint_b64: checkpointB64,
          memoryCheckpoint_b64: checkpointB64,
          checkpointSource: "memory",
          checkpointFileName: "",
        },
      };
    }),
  );
}

/** Same wiring rules as ``CrlTrainerNode`` checkpoint fan-out. */
export function applyCrlCheckpointToCheckpointNodes(
  setNodes: Dispatch<SetStateAction<Node[]>>,
  trainerNodeId: string,
  checkpointB64: string,
  edges: TrainEdgeWire[],
): void {
  if (!checkpointB64) return;
  const targets = new Set<string>();
  for (const e of edges) {
    if (e.source !== trainerNodeId) continue;
    const sh = e.sourceHandle ?? null;
    if (sh != null && sh !== "checkpoint") continue;
    if ((e.targetHandle ?? null) !== "model_checkpoint") continue;
    targets.add(e.target);
  }
  setNodes((prev) =>
    prev.map((n) => {
      if (n.id === trainerNodeId && n.type === "crl_trainer") {
        const prevData = (n.data ?? {}) as Partial<CrlTrainerNodeData>;
        return { ...n, data: { ...prevData, memoryCheckpoint_b64: checkpointB64 } };
      }
      if (!targets.has(n.id) || n.type !== "model_checkpoint") return n;
      const prevData = (n.data ?? {}) as Partial<ModelCheckpointNodeData>;
      return {
        ...n,
        data: {
          ...prevData,
          checkpoint_b64: checkpointB64,
          memoryCheckpoint_b64: checkpointB64,
          checkpointSource: "memory",
          checkpointFileName: "",
        },
      };
    }),
  );
}
