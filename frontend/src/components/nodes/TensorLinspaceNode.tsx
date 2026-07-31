import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { useCallback, useEffect, useRef } from "react";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import { SourceSocketRow } from "./SourceSocketRow";
import {
  defaultTensorLinspaceData,
  type TensorLinspaceNodeData,
  type TensorLinspaceSpace,
} from "./tensorLinspaceDefaults";

const SPACE_OPTIONS: { id: TensorLinspaceSpace; label: string }[] = [
  { id: "linear", label: "linear" },
  { id: "log10", label: "log10" },
];

function buildLinspaceValues(start: number, end: number, numPoints: number, space: TensorLinspaceSpace): number[] {
  if (!Number.isFinite(start) || !Number.isFinite(end)) {
    throw new Error("start and end must be finite numbers.");
  }
  if (!Number.isInteger(numPoints) || numPoints < 1) {
    throw new Error("number of points must be an integer >= 1.");
  }
  if (numPoints === 1) {
    return [space === "log10" ? 10 ** start : start];
  }
  const out: number[] = new Array(numPoints);
  for (let i = 0; i < numPoints; i++) {
    const t = i / (numPoints - 1);
    const x = start + (end - start) * t;
    out[i] = space === "log10" ? 10 ** x : x;
  }
  return out;
}

function patchTensorLinspaceData(
  id: string,
  patch: Partial<TensorLinspaceNodeData>,
  setNodes: (updater: (nodes: Node[]) => Node[]) => void,
) {
  setNodes((nodes) =>
    nodes.map((n) => {
      if (n.id !== id) return n;
      const def = defaultTensorLinspaceData();
      const cur = (n.data ?? {}) as Partial<TensorLinspaceNodeData>;
      const prev: TensorLinspaceNodeData = {
        start: cur.start !== undefined ? cur.start : def.start,
        end: cur.end !== undefined ? cur.end : def.end,
        numPoints: cur.numPoints !== undefined ? cur.numPoints : def.numPoints,
        space: cur.space ?? def.space,
        outputTensor: cur.outputTensor ?? def.outputTensor,
        lastError: cur.lastError ?? def.lastError,
      };
      return { ...n, data: { ...prev, ...patch } };
    }),
  );
}

export function TensorLinspaceNode({ id, data, selected }: NodeProps) {
  const def = defaultTensorLinspaceData();
  const raw = (data ?? {}) as Partial<TensorLinspaceNodeData>;
  const d: TensorLinspaceNodeData = {
    start: raw.start !== undefined ? raw.start : def.start,
    end: raw.end !== undefined ? raw.end : def.end,
    numPoints: raw.numPoints !== undefined ? raw.numPoints : def.numPoints,
    space: raw.space ?? def.space,
    outputTensor: raw.outputTensor ?? def.outputTensor,
    lastError: raw.lastError ?? def.lastError,
  };

  const { setNodes } = useReactFlow();
  const update = useCallback(
    (patch: Partial<TensorLinspaceNodeData>) => patchTensorLinspaceData(id, patch, setNodes),
    [id, setNodes],
  );

  const start = floatChoices(d.start, 0)[0]!;
  const end = floatChoices(d.end, 1)[0]!;
  const numPoints = intChoices(d.numPoints, 8)[0]!;
  const regenSig = `${start}|${end}|${numPoints}|${d.space}`;
  const stableSig = useRef<string>("__never__");
  const lastErr = useRef<string | null>(null);

  useEffect(() => {
    try {
      const values = buildLinspaceValues(start, end, numPoints, d.space);
      const next = { shape: [numPoints], values };
      const prev = d.outputTensor;
      if (
        stableSig.current === regenSig &&
        prev &&
        prev.shape.length === 1 &&
        prev.shape[0] === numPoints &&
        prev.values.length === values.length &&
        prev.values.every((v, i) => v === values[i])
      ) {
        return;
      }
      stableSig.current = regenSig;
      lastErr.current = null;
      update({ outputTensor: next, lastError: null });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (lastErr.current === msg) return;
      lastErr.current = msg;
      stableSig.current = "";
      update({ outputTensor: null, lastError: msg });
    }
  }, [d.outputTensor, d.space, end, numPoints, regenSig, start, update]);

  return (
    <div
      className={`cr-node cr-node--tensor-linspace${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-tensor)" }}
    >
      <div className="cr-node__header">
        <div className="cr-node__header-row cr-node__header-row--io-mode">
          <div className="cr-node__header-title">{readInstanceTitle(data as Record<string, unknown>, "Tensor linspace")}</div>
        </div>
      </div>
      <div className="cr-node__body">
        <SourceSocketRow handleId="tensor" label="tensor" />
        <ComfyFloatListField
          label="start"
          values={floatChoices(d.start, 0)}
          onCommit={(vals) => update({ start: packFloatList(vals) })}
          ariaLabel="Tensor linspace start"
        />
        <ComfyFloatListField
          label="end"
          values={floatChoices(d.end, 1)}
          onCommit={(vals) => update({ end: packFloatList(vals) })}
          ariaLabel="Tensor linspace end"
        />
        <ComfyIntListField
          label="number of points"
          values={intChoices(d.numPoints, 8)}
          min={1}
          onCommit={(vals) => update({ numPoints: packIntList(vals) })}
          ariaLabel="Tensor linspace number of points"
        />
        <DiscreteMultiSelect<TensorLinspaceSpace>
          label="space"
          options={SPACE_OPTIONS}
          value={d.space}
          onCommit={(next) =>
            update({
              space: (typeof next === "string" ? next : next[0] ?? d.space) as TensorLinspaceSpace,
            })
          }
          ariaLabel="Tensor linspace space mode"
          singleSelect
        />
        {d.lastError ? <p className="cr-trainer-train-err">{d.lastError}</p> : null}
      </div>
    </div>
  );
}
