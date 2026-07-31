import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { SourceSocketRow } from "./SourceSocketRow";
import {
  INPUT_DISTRIBUTION_OPTIONS,
  OUTPUT_DISTRIBUTION_OPTIONS,
} from "./linearDatasetDefaults";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import {
  defaultRandomInputDistributionData,
  type RandomInputDistributionNodeData,
} from "./randomInputDistributionDefaults";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import {
  DEFAULT_RANDOM_INPUT_DIST_PARAM_ORDER,
  DEFAULT_RANDOM_INPUT_DIST_SPEC_NAME,
  generateRandomInputDistributionSpecCode,
} from "../../graph/specCode/randomInputDistributionSpecCode";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";

function replaceNodeData(
  id: string,
  data: RandomInputDistributionNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: RandomInputDistributionNodeData,
  patch: Partial<RandomInputDistributionNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

function effectiveParamOrder(d: RandomInputDistributionNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_RANDOM_INPUT_DIST_PARAM_ORDER];
}

export function RandomInputDistributionNode({ id, data, selected }: NodeProps) {
  const defs = defaultRandomInputDistributionData();
  const d = { ...defs, ...(data as Partial<RandomInputDistributionNodeData>) } as RandomInputDistributionNodeData;
  const { setNodes } = useReactFlow();

  const update = useCallback(
    (patch: Partial<RandomInputDistributionNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const full = useMemo(() => d, [d]);
  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_RANDOM_INPUT_DIST_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateRandomInputDistributionSpecCode(d, order, specName),
    [d, order, specName],
  );

  return (
    <div
      className={`cr-node cr-node--random-input-distribution${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="random_input_distribution"
        nodeId={id}
        graphNodeType="random_input_distribution"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Random input distribution")}
        {d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
      </DatasetNodeHeaderWithInfo>
      <div className="cr-node__body">
        <SourceSocketRow handleId="input_distribution" label="input distribution" />

        <ComfyIntListField
          label={String.raw`input dim $d_{\mathrm in}$`}
          values={intChoices(full.inputDim, 10)}
          min={1}
          ariaLabel="Input dimension"
          onCommit={(vals) => update({ inputDim: packIntList(vals) })}
        />
        <DiscreteMultiSelect
          label="input distribution"
          options={INPUT_DISTRIBUTION_OPTIONS}
          value={full.inputDistribution}
          onCommit={(inputDistribution) => update({ inputDistribution })}
          ariaLabel="Base distribution for x"
        />
        <DiscreteMultiSelect
          label="noise on $x$"
          options={OUTPUT_DISTRIBUTION_OPTIONS}
          value={full.noiseDistribution}
          onCommit={(noiseDistribution) => update({ noiseDistribution })}
          ariaLabel="Whether to add Gaussian noise to x after sampling"
        />
        <ComfyFloatListField
          label="noise scale"
          values={floatChoices(full.noiseLevel, 0)}
          min={0}
          positiveOnly={false}
          ariaLabel="Gaussian noise scale on x when enabled"
          onCommit={(vals) => update({ noiseLevel: packFloatList(vals) })}
        />
        <ComfyIntListField
          label="sample seed"
          values={intChoices(full.seed, 0)}
          min={0}
          title="NumPy RNG seed for drawing x (and x-noise) on the server"
          ariaLabel="Random seed for input draws"
          onCommit={(vals) => update({ seed: packIntList(vals) })}
        />
      </div>
    </div>
  );
}
