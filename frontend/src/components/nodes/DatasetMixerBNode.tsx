import { useCallback } from "react";
import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField } from "./comfyMultiFields";
import { floatChoices, packFloatList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import { KatexMixedInline } from "./KatexMixedInline";
import { normalizeDatasetMixerBData, type DatasetMixerBNodeData } from "./datasetMixerBDefaults";

function replaceNodeData(
  id: string,
  data: DatasetMixerBNodeData,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

export function DatasetMixerBNode({ id, data, selected }: NodeProps) {
  const d = normalizeDatasetMixerBData((data ?? {}) as Record<string, unknown>);
  const { setNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<DatasetMixerBNodeData>) => replaceNodeData(id, { ...d, ...patch }, setNodes),
    [d, id, setNodes],
  );

  return (
    <div
      className={`cr-node cr-node--dataset-mixer-b${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo nodeType="dataset_mixer_b">
        {readInstanceTitle(d, "Dataset mixer B")}
      </DatasetNodeHeaderWithInfo>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Dataset mixer B I/O">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="dataset_a"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">dataset A</span>
            </div>
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="dataset_b"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">dataset B (input synced from dataset A)</span>
            </div>
            <div className="cr-trainer-io-row__rightwrap">
              <span className="cr-trainer-output-label">dataset</span>
              <Handle
                type="source"
                position={Position.Right}
                id="dataset"
                className="cr-handle-source cr-handle-source--trainer-row cr-trainer-out-handle cr-trainer-out-handle--dataset"
              />
            </div>
          </div>
        </div>
        <ComfyFloatListField
          label={String.raw`interpolation $\lambda$`}
          ariaLabel="dataset mixer B interpolation lambda"
          values={floatChoices(d.interpolationLambda, 0.5)}
          min={0}
          max={1}
          positiveOnly={false}
          title={String.raw`$O_\lambda = \lambda O_1 + (1-\lambda) O_2$`}
          onCommit={(vals) => update({ interpolationLambda: packFloatList(vals) })}
        />
        <div className="cr-node__hint">
          <KatexMixedInline
            text={String.raw`One synchronized x per row: drawn from dataset A's input law; B is evaluated on that same x (train and test). Mixed output: $O_\lambda = \lambda O_1 + (1-\lambda) O_2$.`}
          />
        </div>
      </div>
    </div>
  );
}
