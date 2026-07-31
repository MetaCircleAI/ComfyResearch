import type { Node } from "@xyflow/react";
import type { Dispatch, SetStateAction } from "react";
import type { TrainerHostTrainUi, TrainerNodeData } from "../components/nodes/trainerDefaults";
import { defaultTrainerData } from "../components/nodes/trainerDefaults";

/** Push or clear host-driven training progress on the trainer node (graph assist, etc.). */
export function patchTrainerHostUi(
  trainerId: string,
  hostTrainUi: TrainerHostTrainUi | null,
  setNodes: Dispatch<SetStateAction<Node[]>>,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== trainerId || n.type !== "trainer") return n;
      const def = defaultTrainerData();
      const cur = (n.data ?? {}) as Partial<TrainerNodeData>;
      const prev: TrainerNodeData = {
        trainingSteps: cur.trainingSteps ?? def.trainingSteps,
        logFrequency: cur.logFrequency ?? def.logFrequency,
        computeDevice: cur.computeDevice ?? def.computeDevice,
        batchSize: cur.batchSize ?? def.batchSize,
        gradClipMaxNorm: cur.gradClipMaxNorm ?? def.gradClipMaxNorm,
        lossHistory: cur.lossHistory,
        testLossHistory: cur.testLossHistory,
        stepTicks: cur.stepTicks,
        observableMetricHistories: cur.observableMetricHistories,
        memoryCheckpoint_b64: cur.memoryCheckpoint_b64,
        hostTrainUi: cur.hostTrainUi,
        targetCurveStepTicks: cur.targetCurveStepTicks,
        targetCurveLossHistory: cur.targetCurveLossHistory,
        lastTrainLoopSeconds: cur.lastTrainLoopSeconds,
        lastAutoTuneSummary: cur.lastAutoTuneSummary,
        autoTuneComparisonResult: cur.autoTuneComparisonResult,
      };
      if (hostTrainUi === null) {
        const { hostTrainUi: _h, ...rest } = prev;
        return { ...n, data: rest };
      }
      return { ...n, data: { ...prev, hostTrainUi } };
    }),
  );
}
