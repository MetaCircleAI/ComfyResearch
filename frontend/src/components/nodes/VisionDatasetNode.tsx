import { useCallback, useMemo } from "react";
import { useReactFlow, type Node, type NodeProps } from "@xyflow/react";
import { ComfyFloatListField, ComfyIntListField } from "./comfyMultiFields";
import { DatasetSourceSockets } from "./DatasetSourceSockets";
import { DiscreteMultiSelect } from "./DiscreteMultiSelect";
import { DATASET_SAMPLING_MODE_OPTIONS, type DatasetSamplingMode } from "./linearDatasetDefaults";
import { DatasetNodeHeaderWithInfo } from "./DatasetNodeHeaderWithInfo";
import { floatChoices, intChoices, packFloatList, packIntList } from "./multiValueUtils";
import { readInstanceTitle } from "../../graph/nodeInstanceTitle";
import type { DatasetNodeInfoKind } from "./datasetNodeInfoContent";
import {
  CIFAR_INPUT_TRANSFORM_OPTIONS,
  CIFAR_PREPROCESSING_OPTIONS,
  defaultParamOrderFor,
  defaultVisionDatasetData,
  type CifarInputTransform,
  type CifarPreprocessing,
  type VisionDatasetKind,
  type VisionDatasetNodeData,
} from "./visionDatasetDefaults";
import { DEFAULT_VISION_DATASET_SPEC_NAME, generateVisionDatasetSpecCode } from "../../graph/specCode/visionDatasetSpecCode";

function replaceNodeData(
  id: string,
  data: VisionDatasetNodeData,
  setNodes: (updater: (nodes: Node[]) => void) => void,
) {
  setNodes((nodes) => nodes.map((n) => (n.id === id ? { ...n, data } : n)));
}

function inferKind(type: string | undefined): VisionDatasetKind {
  const t = (type ?? "mnist_dataset") as VisionDatasetKind;
  return t;
}

export function VisionDatasetNode({ id, data, selected, type }: NodeProps) {
  const kind = inferKind(type);
  const defs = defaultVisionDatasetData(kind);
  const d = { ...defs, ...(data as Partial<VisionDatasetNodeData>) } as VisionDatasetNodeData;
  const { setNodes } = useReactFlow();
  const infoKind = kind as DatasetNodeInfoKind;
  const order = d.paramOrder?.length ? d.paramOrder : defaultParamOrderFor(kind);
  const specName = (d.specCodeName ?? DEFAULT_VISION_DATASET_SPEC_NAME).trim() || DEFAULT_VISION_DATASET_SPEC_NAME;
  const generatedCode = useMemo(
    () => generateVisionDatasetSpecCode(kind, d, order, specName),
    [d, kind, order, specName],
  );
  const seedVal = intChoices(d.initSeed ?? d.seed ?? 0, 0)[0];
  const cifarWhitening = kind === "cifar10_dataset" && d.preprocessing === "center_crop_28_per_image_whiten";
  const update = useCallback(
    (patch: Partial<VisionDatasetNodeData>) => {
      const next = { ...d, ...patch } as VisionDatasetNodeData;
      if (patch.seed != null) next.initSeed = patch.seed;
      if (patch.initSeed != null) next.seed = patch.initSeed;
      replaceNodeData(id, next, setNodes);
    },
    [d, id, setNodes],
  );

  const title =
    kind === "mnist_dataset"
      ? "MNIST dataset"
      : kind === "cifar10_dataset"
        ? "CIFAR-10 dataset"
      : kind === "gaussian_blob_dataset"
        ? "Gaussian blob dataset"
        : kind === "shape_world_dataset"
          ? "Shape world dataset"
          : "Hole counting dataset";

  return (
    <div
      className={`cr-node cr-node--linear-dataset${selected ? " cr-node--selected" : ""}`}
      style={{ ["--accent" as string]: "var(--cr-accent-dataset)" }}
    >
      <DatasetNodeHeaderWithInfo
        nodeType={infoKind}
        nodeId={id}
        graphNodeType={kind}
        specPythonCode={generatedCode}
      >
        {readInstanceTitle(d as Record<string, unknown>, title)}
        {d.specCodeName ? <span className="cr-node__header-sub">{d.specCodeName}</span> : null}
      </DatasetNodeHeaderWithInfo>
      <div className="cr-node__body">
        <DatasetSourceSockets />
        <div className="cr-comfy-field">
          <label className="cr-comfy-widget cr-comfy-widget--flush nodrag nopan">
            <span
              className="cr-comfy-widget__label"
              title="Reshape [N,1,H,W] to [N,C·H·W] for mlp_model + cross_entropy_loss. Leave off for resnet_model / vit_model."
            >
              flatten to vector (MLP + CE)
            </span>
            <input
              type="checkbox"
              className="cr-checkbox-inline"
              checked={!!d.flattenOutput}
              onChange={(e) => update({ flattenOutput: e.target.checked })}
              aria-label="Flatten images to vectors for MLP classification"
            />
          </label>
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
        {(kind === "mnist_dataset" || kind === "cifar10_dataset") ? (
          <div className="cr-comfy-field">
            <label className="cr-comfy-widget cr-comfy-widget--flush">
              <span
                className="cr-comfy-widget__label"
                title={
                  kind === "cifar10_dataset"
                    ? "Override bundled data/cifar10 (run scripts/fetch_bundled_cifar10.py). Empty uses bundled path or URL cache."
                    : "Override bundled data/mnist. Empty uses data/mnist or ~/.cache/comfy_research_mnist."
                }
              >
                download cache dir
              </span>
              <input
                className="cr-input cr-comfy-widget__control"
                type="text"
                value={d.downloadCacheDir ?? ""}
                onChange={(e) => update({ downloadCacheDir: e.target.value })}
                aria-label={
                  kind === "cifar10_dataset"
                    ? "CIFAR-10 download cache directory"
                    : "MNIST download cache directory"
                }
                placeholder={
                  kind === "cifar10_dataset"
                    ? "default: data/cifar10 (bundled)"
                    : "default: data/mnist (bundled)"
                }
              />
            </label>
          </div>
        ) : null}
        {kind === "cifar10_dataset" ? (
          <>
            <ComfyIntListField
              label="subset seed"
              values={intChoices(d.subsetSeed ?? 0, 0)}
              min={0}
              ariaLabel="CIFAR-10 subset seed"
              onCommit={(vals) => update({ subsetSeed: packIntList(vals) })}
            />
            <DiscreteMultiSelect<CifarInputTransform>
              label="input transform"
              options={CIFAR_INPUT_TRANSFORM_OPTIONS}
              value={d.inputTransform ?? "none"}
              onCommit={(v) =>
                update({
                  inputTransform: (typeof v === "string" ? v : v[0] ?? "none") as CifarInputTransform,
                })
              }
              ariaLabel="CIFAR-10 input transform"
              singleSelect
            />
            <DiscreteMultiSelect<CifarPreprocessing>
              label="preprocessing"
              options={CIFAR_PREPROCESSING_OPTIONS}
              value={d.preprocessing ?? "none"}
              onCommit={(v) => {
                const preprocessing = (typeof v === "string" ? v : v[0] ?? "none") as CifarPreprocessing;
                update({
                  preprocessing,
                  ...(preprocessing === "center_crop_28_per_image_whiten" && d.normalize === "minus_one_to_one"
                    ? { normalize: "zero_one" }
                    : {}),
                });
              }}
              ariaLabel="CIFAR-10 preprocessing"
              singleSelect
            />
            <ComfyFloatListField
              label="label corruption"
              ariaLabel="Fraction of CIFAR-10 training labels randomized"
              values={floatChoices(d.labelCorruption ?? 0, 0)}
              min={0}
              max={1}
              title="Training labels are replaced once using the dataset seed and remain fixed across epochs. Test labels stay unchanged."
              onCommit={(vals) => update({ labelCorruption: packFloatList(vals) })}
            />
            <div className="cr-comfy-field">
              <label className="cr-comfy-widget cr-comfy-widget--flush">
                <span className="cr-comfy-widget__label">class balanced</span>
                <input type="checkbox" className="cr-checkbox-inline" checked={d.classBalanced ?? true} onChange={(event) => update({ classBalanced: event.target.checked })} />
              </label>
            </div>
            <label className="cr-comfy-widget cr-comfy-widget--flush">
              <span className="cr-comfy-widget__label">normalization</span>
              <select className="cr-input cr-comfy-widget__control" value={d.normalize ?? "zero_one"} onChange={(event) => update({ normalize: event.target.value as VisionDatasetNodeData["normalize"] })}>
                <option value="zero_one">[0, 1]</option>
                <option value="minus_one_to_one" disabled={cifarWhitening}>[-1, 1] (DDPM)</option>
              </select>
              {cifarWhitening ? (
                <span className="cr-comfy-widget__hint">
                  Per-image whitening cannot be combined with DDPM [-1, 1] normalization; the backend rejects this combination.
                </span>
              ) : null}
            </label>
          </>
        ) : null}
        {kind === "gaussian_blob_dataset" ? (
          <>
            <ComfyIntListField
              label="# classes"
              ariaLabel="Number of classes"
              values={intChoices(d.numClasses ?? 10, 10)}
              min={2}
              max={64}
              title="Integer labels are 0 … numClasses − 1; each class is a Gaussian blob on a grid inside the image."
              onCommit={(vals) => update({ numClasses: packIntList(vals) })}
            />
            <ComfyIntListField
              label={String.raw`image size (px)`}
              ariaLabel="Image size"
              values={intChoices(d.imageSize ?? 28, 28)}
              min={8}
              max={64}
              onCommit={(vals) => update({ imageSize: packIntList(vals) })}
            />
            <ComfyFloatListField
              label="noise level"
              ariaLabel="Per-image Gaussian noise standard deviation"
              values={floatChoices(d.noiseLevel ?? 0.15, 0.15)}
              min={0}
              max={2}
              title="Std dev of Gaussian noise added to each sample (0 = clean class templates)."
              onCommit={(vals) => update({ noiseLevel: packFloatList(vals) })}
            />
          </>
        ) : null}
        {kind === "shape_world_dataset" ? (
          <>
            <ComfyIntListField
              label={String.raw`image size (px)`}
              ariaLabel="Image size"
              values={intChoices(d.imageSize ?? 32, 32)}
              min={16}
              max={96}
              onCommit={(vals) => update({ imageSize: packIntList(vals) })}
            />
            <ComfyFloatListField
              label="noise level (σ)"
              ariaLabel="Shape world additive noise scale"
              title="Gaussian noise added after drawing the shape; 0 = no noise."
              values={floatChoices(d.noiseLevel ?? 0.04, 0.04)}
              min={0}
              max={0.5}
              onCommit={(vals) => update({ noiseLevel: packFloatList(vals) })}
            />
          </>
        ) : null}
        {kind === "hole_counting_dataset" ? (
          <>
            <ComfyIntListField
              label={String.raw`image size (px)`}
              ariaLabel="Image size"
              values={intChoices(d.imageSize ?? 48, 48)}
              min={24}
              max={96}
              onCommit={(vals) => update({ imageSize: packIntList(vals) })}
            />
            <ComfyIntListField
              label={String.raw`max holes $k$ (classes $0\ldots k$)`}
              ariaLabel="Max holes"
              values={intChoices(d.maxHoles ?? 3, 3)}
              min={0}
              max={8}
              onCommit={(vals) => update({ maxHoles: packIntList(vals) })}
            />
          </>
        ) : null}
        <ComfyIntListField
          label="train size"
          values={intChoices(d.trainSize, 2048)}
          min={1}
          ariaLabel="Train size"
          onCommit={(vals) => update({ trainSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="test size"
          values={intChoices(d.testSize, 512)}
          min={0}
          ariaLabel="Test size"
          onCommit={(vals) => update({ testSize: packIntList(vals) })}
        />
        <ComfyIntListField
          label="init seed"
          values={intChoices(seedVal, 0)}
          min={0}
          title="Stochastic dataset generation and splits."
          ariaLabel="Init seed"
          onCommit={(vals) => {
            const s = packIntList(vals);
            update({ seed: s, initSeed: s });
          }}
        />
      </div>
    </div>
  );
}
