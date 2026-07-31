import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField } from "./comfyMultiFields";
import { ObservableNodeHeader } from "./ObservableNodeHeader";
import { ObservableSourceStrip } from "./ObservableSourceStrip";
import {
  defaultKanRegData,
  KAN_REG_METRIC_IDS,
  type KanRegMetricId,
  type KanRegNodeData,
} from "./kanRegDefaults";
import { floatChoices, packFloatList } from "./multiValueUtils";

function patchData(
  id: string,
  prev: KanRegNodeData & ReturnType<typeof defaultKanRegData>,
  patch: Partial<KanRegNodeData>,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => (n.id === id ? { ...n, data: { ...prev, ...patch } } : n)),
  );
}

export function KanRegNode({ id, data, selected }: NodeProps) {
  const defs = defaultKanRegData();
  const d = { ...defs, ...(data as Partial<KanRegNodeData>) } as KanRegNodeData & typeof defs;
  const { setNodes } = useReactFlow();
  const update = (patch: Partial<KanRegNodeData>) => patchData(id, d, patch, setNodes);
  const metric = KAN_REG_METRIC_IDS.includes(d.regMetric as KanRegMetricId)
    ? d.regMetric
    : defs.regMetric;

  return (
    <div
      className={`cr-node cr-node--kan-reg${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-loss)" }}
    >
      <ObservableNodeHeader id={id} graphNodeType="kan_reg" title="KAN reg" />
      <div className="cr-node__body cr-node__body--compact cr-node__body--kan-reg">
        <ObservableSourceStrip />
        <label className="cr-kan-reg-metric nodrag nopan">
          <span className="cr-kan-reg-metric__label">reg metric</span>
          <select
            className="cr-kan-reg-metric__select"
            value={metric}
            onChange={(e) => update({ regMetric: e.target.value })}
          >
            {KAN_REG_METRIC_IDS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <ComfyFloatListField
          label="λ (scale)"
          values={floatChoices(d.lamb, defs.lamb)}
          title="Outer multiplier on the full pykan get_reg term (matches pykan fit lamb)"
          ariaLabel="KAN reg lambda scale"
          onCommit={(vals) => update({ lamb: packFloatList(vals) })}
        />
        <ComfyFloatListField
          label="λ_l1"
          values={floatChoices(d.lambL1, defs.lambL1)}
          title="pykan lamb_l1 — L1 on activation scale tensor"
          ariaLabel="KAN reg lamb l1"
          onCommit={(vals) => update({ lambL1: packFloatList(vals) })}
        />
        <ComfyFloatListField
          label="λ_entropy"
          values={floatChoices(d.lambEntropy, defs.lambEntropy)}
          title="pykan lamb_entropy"
          ariaLabel="KAN reg lamb entropy"
          onCommit={(vals) => update({ lambEntropy: packFloatList(vals) })}
        />
        <ComfyFloatListField
          label="λ_coef"
          values={floatChoices(d.lambCoef, defs.lambCoef)}
          title="pykan lamb_coef — spline coefficient sparsity"
          ariaLabel="KAN reg lamb coef"
          onCommit={(vals) => update({ lambCoef: packFloatList(vals) })}
        />
        <ComfyFloatListField
          label="λ_coef_diff"
          values={floatChoices(d.lambCoefDiff, defs.lambCoefDiff)}
          title="pykan lamb_coefdiff — smoothness across spline coeffs"
          ariaLabel="KAN reg lamb coef diff"
          onCommit={(vals) => update({ lambCoefDiff: packFloatList(vals) })}
        />
        <p className="cr-observable-hint">
          Adds λ·get_reg(metric, …) to the MSE loss for <strong>kan_model</strong> only. Wire to Trainer
          observables; a viz spawns automatically.
        </p>
      </div>
    </div>
  );
}
