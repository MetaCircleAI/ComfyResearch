import { addEdge, type Edge, type Node } from "@xyflow/react";
import { appendResearchNode } from "./nodeInstanceTitle";
import { spawnConfigFor } from "./observableVizTrainerSpawn";
import { defaultTrainingVisualizationData } from "../components/nodes/trainingVisualizationDefaults";
import { nodeRegistryTypesWithFamily } from "./nodeRegistrySpec";

/** Right of trainer; training viz is tall — keep it high so it does not cover the observable viz below. */
export const TRAINER_AUTO_VIZ_DX = 300;
export const TRAINER_AUTO_TRAINING_VIZ_DY = -520;
export const TRAINER_AUTO_OBSERVABLE_VIZ_DY = 260;

const TRAINER_LOSS_VIZ_SPAWN_TYPES = new Set<string>(nodeRegistryTypesWithFamily("trainer_loss_viz_spawn"));

/**
 * Ensures training / observable viz nodes exist when loss or observable is wired to a trainer,
 * matching manual ``onConnect`` behavior (idempotent).
 */
function safeVizSuffix(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_");
}

export function ensureTrainerAutoVizes(
  nodes: Node[],
  edges: Edge[],
  trainerId: string,
  observableVizPresetId?: string,
  trainingVizPresetId?: string,
): { nodes: Node[]; edges: Edge[] } {
  let outNodes = nodes;
  let outEdges = edges;
  const trainer = outNodes.find((n) => n.id === trainerId);
  if (!trainer) return { nodes: outNodes, edges: outEdges };

  const obsWires = outEdges.filter(
    (e) =>
      e.target === trainerId &&
      e.targetHandle === "observables" &&
      (e.sourceHandle === "observables" || e.sourceHandle === "observable"),
  );
  const usePresetForSingleObs =
    obsWires.length === 1 &&
    typeof observableVizPresetId === "string" &&
    observableVizPresetId.length > 0;

  obsWires.forEach((obsWire, obsIdx) => {
    const hasOvForSource = outEdges.some((e) => {
      if (e.source !== trainerId || e.sourceHandle !== "observable_results" || e.targetHandle !== "tensor") {
        return false;
      }
      const vn = outNodes.find((x) => x.id === e.target);
      if (vn?.type !== "observable_viz") return false;
      const d = (vn.data ?? {}) as { pairedObservableId?: string };
      return d.pairedObservableId === obsWire.source;
    });
    if (hasOvForSource) return;

    const obs = outNodes.find((n) => n.id === obsWire.source);
    const spawn = obs?.type ? spawnConfigFor(String(obs.type)) : undefined;
    if (!spawn) return;

    const vizId = usePresetForSingleObs
      ? observableVizPresetId!
      : `${trainerId}__ov__${safeVizSuffix(obsWire.source)}`;
    const pos = {
      x: trainer.position.x + TRAINER_AUTO_VIZ_DX,
      y: trainer.position.y + TRAINER_AUTO_OBSERVABLE_VIZ_DY + obsIdx * 44,
    };
    const data = spawn.defaultData(obsWire.source, trainerId, obs);
    const vizNode = appendResearchNode(
      outNodes,
      "observable_viz",
      pos,
      data as Record<string, unknown>,
      vizId,
    );
    outNodes = [...outNodes, vizNode];
    outEdges = addEdge(
      {
        source: trainerId,
        target: vizId,
        sourceHandle: "observable_results",
        targetHandle: "tensor",
      },
      outEdges,
    );
  });

  const lossWire = outEdges.find(
    (e) => e.target === trainerId && e.targetHandle === "loss" && e.sourceHandle === "loss",
  );
  if (lossWire) {
    const hasTvEdge = outEdges.some((e) => {
      if (e.source !== trainerId || e.sourceHandle !== "loss_results" || e.targetHandle !== "tensor_list") {
        return false;
      }
      const vn = outNodes.find((x) => x.id === e.target);
      return vn?.type === "training_visualization";
    });
    if (!hasTvEdge) {
      const lossNode = outNodes.find((n) => n.id === lossWire.source);
      const ok = TRAINER_LOSS_VIZ_SPAWN_TYPES.has(String(lossNode?.type));
      if (ok) {
        const tvId =
          typeof trainingVizPresetId === "string" && trainingVizPresetId.length > 0
            ? trainingVizPresetId
            : `${trainerId}__train_viz`;
        const pos = {
          x: trainer.position.x + TRAINER_AUTO_VIZ_DX,
          y: trainer.position.y + TRAINER_AUTO_TRAINING_VIZ_DY,
        };
        const vizNode = appendResearchNode(
          outNodes,
          "training_visualization",
          pos,
          defaultTrainingVisualizationData() as Record<string, unknown>,
          tvId,
        );
        outNodes = [...outNodes, vizNode];
        outEdges = addEdge(
          {
            source: trainerId,
            target: tvId,
            sourceHandle: "loss_results",
            targetHandle: "tensor_list",
          },
          outEdges,
        );
      }
    }
  }

  return { nodes: outNodes, edges: outEdges };
}
