import type { Node } from "@xyflow/react";
import type { TrainerTrainUi } from "../components/nodes/trainerDefaults";
import { getTrainerRunSession } from "./trainerRunSession";

export type ProjectTrainerActivity = {
  trainerId: string;
  trainUi: TrainerTrainUi;
};

export type ProjectTrainSummary = {
  hasActiveTraining: boolean;
  progressPct: number;
  loading: boolean;
  paused: boolean;
  trainers: ProjectTrainerActivity[];
};

type CanvasSlice = { nodes: Node[] };

function readTrainUi(node: Node): TrainerTrainUi | undefined {
  if (node.type !== "trainer") return undefined;
  const ui = (node.data as { trainUi?: TrainerTrainUi } | undefined)?.trainUi;
  if (!ui?.active) return undefined;
  return ui;
}

/** Scan a project's canvas for in-flight or paused trainer runs (``trainUi`` on nodes). */
export function summarizeProjectTrainActivity(canvas: CanvasSlice): ProjectTrainSummary {
  const trainers: ProjectTrainerActivity[] = [];
  for (const node of canvas.nodes) {
    const trainUi = readTrainUi(node);
    if (!trainUi) continue;
    if (!trainUi.loading && !trainUi.paused) continue;
    trainers.push({ trainerId: node.id, trainUi });
  }
  const loading = trainers.some((t) => t.trainUi.loading);
  const paused = trainers.some((t) => t.trainUi.paused && !t.trainUi.loading);
  const progressPct = trainers.reduce((max, t) => Math.max(max, t.trainUi.progressPct), 0);
  return {
    hasActiveTraining: trainers.length > 0,
    progressPct,
    loading,
    paused,
    trainers,
  };
}

export async function abortTrainerOnServer(trainerId: string): Promise<void> {
  try {
    await fetch("/api/train/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trainer_node_id: trainerId, action: "abort" }),
    });
  } catch {
    /* best-effort */
  }
}

export async function abortProjectTraining(trainers: ProjectTrainerActivity[]): Promise<void> {
  await Promise.all(
    trainers.map((t) => {
      getTrainerRunSession(t.trainerId)?.abort();
      return abortTrainerOnServer(t.trainerId);
    }),
  );
}
