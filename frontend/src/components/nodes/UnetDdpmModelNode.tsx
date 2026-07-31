import { useCallback } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyIntListField } from "./comfyMultiFields";
import { ModelInitSourceSocketStrip } from "./ModelInitSourceSocketStrip";
import { intChoices, packIntList } from "./multiValueUtils";

type Data = { inChannels?: number; baseChannels?: number; channelMult?: string; timeEmbedDim?: number; diffusionTimesteps?: number; imageSize?: number; seed?: number };

export function UnetDdpmModelNode({ id, data, selected }: NodeProps) {
  const { setNodes } = useReactFlow();
  const d = { inChannels: 3, baseChannels: 64, channelMult: "1,2,2", timeEmbedDim: 128, diffusionTimesteps: 1000, imageSize: 32, seed: 0, ...(data as Data) };
  const patch = useCallback((next: Partial<Data>) => setNodes((nodes: Node[]) => nodes.map((node) => node.id === id ? { ...node, data: { ...d, ...next } } : node)), [d, id, setNodes]);
  return <div className={`cr-node cr-node--mlp-model${selected ? " cr-node--selected" : ""}`} style={{ ["--accent" as string]: "var(--cr-accent-model)" }}>
    <div className="cr-node__header"><div className="cr-node__header-title">UNet DDPM</div></div>
    <div className="cr-node__body"><ModelInitSourceSocketStrip sourceHandleId="model" sourceLabel="model" />
      <ComfyIntListField label="base channels" values={intChoices(d.baseChannels, 64)} min={8} onCommit={(values) => patch({ baseChannels: packIntList(values) as number })} />
      <label className="cr-comfy-widget"><span className="cr-comfy-widget__label">channel multipliers</span><input className="cr-input" value={d.channelMult} onChange={(event) => patch({ channelMult: event.target.value })} /></label>
      <ComfyIntListField label="time embed dim" values={intChoices(d.timeEmbedDim, 128)} min={8} onCommit={(values) => patch({ timeEmbedDim: packIntList(values) as number })} />
      <ComfyIntListField label="diffusion T" values={intChoices(d.diffusionTimesteps, 1000)} min={2} onCommit={(values) => patch({ diffusionTimesteps: packIntList(values) as number })} />
      <ComfyIntListField label="init seed" values={intChoices(d.seed, 0)} min={0} onCommit={(values) => patch({ seed: packIntList(values) as number })} />
    </div>
  </div>;
}
