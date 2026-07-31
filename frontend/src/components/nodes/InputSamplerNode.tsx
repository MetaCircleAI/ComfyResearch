import { useCallback } from "react";
import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { intChoices, packIntList } from "./multiValueUtils";
import { defaultInputSamplerData, type InputSamplerNodeData } from "./inputSamplerDefaults";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import {
  DEFAULT_INPUT_SAMPLER_SPEC_NAME,
  generateInputSamplerSpecCode,
} from "../../graph/specCode/inputSamplerSpecCode";

function replaceNodeData(
  id: string,
  data: InputSamplerNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function patchData(
  id: string,
  prev: InputSamplerNodeData,
  patch: Partial<InputSamplerNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  replaceNodeData(id, { ...prev, ...patch }, setNodes);
}

export function InputSamplerNode({ id, data, selected }: NodeProps) {
  const defs = defaultInputSamplerData();
  const d = { ...defs, ...(data as Partial<InputSamplerNodeData>) } as InputSamplerNodeData;
  const { setNodes } = useReactFlow();

  const update = useCallback(
    (patch: Partial<InputSamplerNodeData>) => patchData(id, d, patch, setNodes),
    [d, id, setNodes],
  );

  const specName = DEFAULT_INPUT_SAMPLER_SPEC_NAME;
  const generatedCode = generateInputSamplerSpecCode(d, specName);

  return (
    <div
      className={`cr-node cr-node--input-sampler${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="input_sampler"
        nodeId={id}
        graphNodeType="input_sampler"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Input sampler")}
      </DatasetNodeHeaderWithInfo>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Input sampler I/O">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="distribution"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">distribution</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">sample tensor</span>
              <Handle
                type="source"
                position={Position.Right}
                id="sample_tensor"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--dataset"
              />
            </div>
          </div>
        </div>

        <ComfyIntListField
          label="num samples"
          values={intChoices(d.numSamples, defs.numSamples)}
          min={1}
          ariaLabel="Number of samples to draw"
          onCommit={(vals) => update({ numSamples: packIntList(vals) })}
        />
      </div>
    </div>
  );
}
