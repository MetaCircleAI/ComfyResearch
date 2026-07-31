import { useCallback, useMemo } from "react";
import { Handle, Position, useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import {
  DATASET_SAMPLING_MODE_OPTIONS,
  type DatasetSamplingMode,
} from "./linearDatasetDefaults";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { defaultTeacherDatasetData, type TeacherDatasetNodeData } from "./teacherDatasetDefaults";
import {
  DEFAULT_TEACHER_DATASET_PARAM_ORDER,
  DEFAULT_TEACHER_DATASET_SPEC_NAME,
  generateTeacherDatasetSpecCode,
} from "../../graph/specCode/teacherDatasetSpecCode";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";

function effectiveParamOrder(d: TeacherDatasetNodeData): string[] {
  if (d.paramOrder?.length) return d.paramOrder;
  return [...DEFAULT_TEACHER_DATASET_PARAM_ORDER];
}

export function TeacherDatasetNode({ id, data, selected }: NodeProps) {
  const defs = defaultTeacherDatasetData();
  const d = { ...defs, ...(data as Partial<TeacherDatasetNodeData>) } as TeacherDatasetNodeData;
  const { setNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<TeacherDatasetNodeData>) => {
      setNodes((nodes: Node[]) =>
        nodes.map((n) => (n.id === id ? { ...n, data: { ...d, ...patch } as TeacherDatasetNodeData } : n)),
      );
    },
    [d, id, setNodes],
  );
  const order = useMemo(() => effectiveParamOrder(d), [d]);
  const specName = d.specCodeName ?? DEFAULT_TEACHER_DATASET_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateTeacherDatasetSpecCode(d, order, specName),
    [d, order, specName],
  );

  return (
    <div
      className={`cr-node cr-node--teacher-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType="teacher_dataset"
        nodeId={id}
        graphNodeType="teacher_dataset"
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d, "Teacher Dataset")}
        {d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
      </DatasetNodeHeaderWithInfo>
      <div className="cr-node__body">
        <div className="cr-trainer-io" aria-label="Teacher dataset I/O (paired inputs and outputs)">
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="model"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--model"
              />
              <span className="cr-trainer-socket-label">model</span>
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
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="train_input"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">train input</span>
            </div>
          </div>
          <div className="cr-trainer-io-row">
            <div className="cr-trainer-io-row__leftwrap">
              <Handle
                type="target"
                position={Position.Left}
                id="test_input"
                className="cr-handle-target cr-handle-target--trainer-row cr-trainer-handle cr-trainer-handle--dataset"
              />
              <span className="cr-trainer-socket-label">test input</span>
            </div>
          </div>
        </div>

        <DiscreteMultiSelect<DatasetSamplingMode>
          label="train data sampling"
          options={DATASET_SAMPLING_MODE_OPTIONS}
          value={d.samplingMode ?? "fixed"}
          onCommit={(v) =>
            update({
              samplingMode: (typeof v === "string" ? v : v[0] ?? "fixed") as DatasetSamplingMode,
            })
          }
          ariaLabel="Train data sampling mode for the trainer"
          singleSelect
        />

        {d.extras && Object.keys(d.extras).length > 0 ? (
          <p className="cr-node__hint cr-node__hint--extras">
            Extra params from spec (not used by training): {JSON.stringify(d.extras)}
          </p>
        ) : null}
      </div>
    </div>
  );
}
