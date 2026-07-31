import type { Node } from "@xyflow/react";
import type { ObservableVizVariant } from "./observableVizVariant";
import { defaultObservableVizEmbeddingTrajectoryData } from "../components/nodes/observableVizEmbeddingTrajectoryDefaults";
import { defaultObservableVizHessianEigenvaluesData } from "../components/nodes/observableVizHessianEigenvaluesDefaults";
import { defaultObservableVizNeuronTrajectory2dData } from "../components/nodes/observableVizNeuronTrajectory2dDefaults";
import { defaultObservableVizUserData } from "../components/nodes/observableVizUserDefaults";
import { defaultAttentionMapVizData } from "../components/nodes/attentionMapVizDefaults";
import { GENERATED_NODE_SPECS } from "../generated/generatedNodeSpecs";

export type ObservableVizSpawnConfig = {
  vizVariant: ObservableVizVariant;
  defaultData: (
    pairedObservableId: string,
    pairedTrainerId: string,
    sourceNode: Node | undefined,
  ) => Record<string, unknown>;
};

/** Generated-only spawn lookup：配置由 generatedNodeSpecs 的 spawn 块逐 node
 * 构造，不按 variant 泛化。手写兜底表已随 kan_reg 接入而删除。 */
export function spawnConfigFor(nodeType: string): ObservableVizSpawnConfig | undefined {
  const g = GENERATED_NODE_SPECS[nodeType];
  if (g?.observable?.spawnsVizNode && g.observable.vizVariant && g.spawn) {
    const variant = g.observable.vizVariant as ObservableVizVariant;
    const spawn = g.spawn;
    const vizTitle = g.observable.vizTitle ?? g.label;
    const fieldDefault = (key: string): number => {
      const f = (g.fields ?? []).find((x) => x.key === key);
      const n = f ? Number((f as { defaultValue?: unknown }).defaultValue) : Number.NaN;
      return Number.isFinite(n) ? n : 1;
    };
    // titleFromField uses trimmed source data and falls back to title or vizTitle.
    const resolveTitle = (sourceNode: Node | undefined): string => {
      const fallback = spawn.title ?? vizTitle;
      if (spawn.titleFromField == null) return fallback;
      const od = (sourceNode?.data ?? {}) as Record<string, unknown>;
      const raw = od[spawn.titleFromField];
      return typeof raw === "string" && raw.trim() ? raw.trim() : fallback;
    };
    return {
      vizVariant: variant,
      defaultData: (pairedObservableId, pairedTrainerId, sourceNode) => {
        if (spawn.kind === "user_scalar") {
          // Omitting unit also omits vizYAxisLabel; do not default it to "value".
          const spawnTitle = resolveTitle(sourceNode);
          return {
            ...(spawn.unit != null
              ? defaultObservableVizUserData(pairedObservableId, pairedTrainerId, spawnTitle, spawn.unit)
              : defaultObservableVizUserData(pairedObservableId, pairedTrainerId, spawnTitle)),
            vizVariant: variant,
          };
        }
        if (spawn.kind === "neuron_trajectory") {
          return {
            ...defaultObservableVizNeuronTrajectory2dData(pairedObservableId, pairedTrainerId),
            vizVariant: variant,
          };
        }
        if (spawn.kind === "scalar_series") {
          // weight_l1 / relu_nonlinear 的专用 builder 输出逐字段相同(实测):
          // 与 defaultObservableVizWeightL1Data / defaultObservableVizReluNonlinearData
          // 的字面等价由 seam 测试对 builder deepEqual 锚定。
          return {
            pairedObservableId,
            pairedTrainerId,
            logScaleX: false,
            logScaleY: false,
            showSeries: true,
            vizVariant: variant,
          };
        }
        if (spawn.kind === "embedding_trajectory") {
          return {
            ...defaultObservableVizEmbeddingTrajectoryData(pairedObservableId, pairedTrainerId, resolveTitle(sourceNode)),
            vizVariant: variant,
          };
        }
        if (spawn.kind === "attention_map") {
          return {
            ...defaultAttentionMapVizData(pairedObservableId, pairedTrainerId, resolveTitle(sourceNode)),
            vizVariant: variant,
          };
        }
        const od = (sourceNode?.data ?? {}) as Record<string, unknown>;
        const raw = spawn.topKFromField != null ? Number(od[spawn.topKFromField]) : Number.NaN;
        const topK =
          spawn.fixedTopK ??
          (Number.isFinite(raw) ? Math.max(1, Math.floor(raw)) : fieldDefault(spawn.topKFromField ?? ""));
        // Only an explicit "ascending" source value selects ascending order.
        const order =
          spawn.orderFromField != null && od[spawn.orderFromField] === "ascending"
            ? ("ascending" as const)
            : ("descending" as const);
        return {
          ...defaultObservableVizHessianEigenvaluesData(pairedObservableId, pairedTrainerId, topK, order),
          vizVariant: variant,
          ...(spawn.seriesLabels ? { seriesLabels: [...spawn.seriesLabels] } : {}),
        };
      },
    };
  }
  return undefined;
}
