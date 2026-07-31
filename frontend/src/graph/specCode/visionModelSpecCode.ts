import type { ResnetModelNodeData } from "../../components/nodes/resnetModelDefaults";
import type { VitModelNodeData } from "../../components/nodes/vitModelDefaults";
import type { ListOr1 } from "../../components/nodes/multiValueUtils";

function first<T>(v: ListOr1<T>, d: T): T {
  if (Array.isArray(v)) return (v[0] ?? d) as T;
  return (v ?? d) as T;
}

export function buildResnetModelNotebookPython(
  slug: string,
  title: string,
  raw: Record<string, unknown>,
): string {
  const d = raw as Partial<ResnetModelNodeData>;
  const variant = first(d.variant ?? "resnet18", "resnet18");
  const seed = first(d.seed ?? 0, 0);
  const baseChannels = first(d.baseChannels ?? 32, 32);
  const blocksStage1 = first(d.blocksStage1 ?? 2, 2);
  const blocksStage2 = first(d.blocksStage2 ?? 2, 2);
  const blocksStage3 = first(d.blocksStage3 ?? 2, 2);
  const blocksStage4 = first(d.blocksStage4 ?? 2, 2);
  const kernelSize = first(d.kernelSize ?? 3, 3);
  const mdInner =
    variant === "self_defined"
      ? `{
        "variant": ${JSON.stringify(variant)},
        "seed": ${JSON.stringify(seed)},
        "baseChannels": ${JSON.stringify(baseChannels)},
        "blocksStage1": ${JSON.stringify(blocksStage1)},
        "blocksStage2": ${JSON.stringify(blocksStage2)},
        "blocksStage3": ${JSON.stringify(blocksStage3)},
        "blocksStage4": ${JSON.stringify(blocksStage4)},
        "kernelSize": ${JSON.stringify(kernelSize)},
    }`
      : `{"variant": ${JSON.stringify(variant)}, "seed": ${JSON.stringify(seed)}}`;
  return `# === ${title} (resnet_model) ===
import torch
from comfy_research.engine.vision_models import build_resnet_from_md

def model_${slug}(num_classes: int, in_channels: int = 1):
    md = ${mdInner}
    torch.manual_seed(int(md["seed"]))
    return build_resnet_from_md(md, in_channels=in_channels, num_classes=num_classes)
`;
}

export function buildVitModelNotebookPython(slug: string, title: string, raw: Record<string, unknown>): string {
  const d = raw as Partial<VitModelNodeData>;
  const variant = first(d.variant ?? "tiny", "tiny");
  const patchSize = first(d.patchSize ?? 4, 4);
  const hiddenDim = first(d.hiddenDim ?? 128, 128);
  const depth = first(d.depth ?? 3, 3);
  const numHeads = first(d.numHeads ?? 4, 4);
  const seed = first(d.seed ?? 0, 0);
  return `# === ${title} (vit_model) ===
import torch
from comfy_research.engine.vision_models import build_vit_from_md

def model_${slug}(num_classes: int, image_size: int, in_channels: int = 1):
    md = {
        "variant": ${JSON.stringify(variant)},
        "patchSize": ${JSON.stringify(patchSize)},
        "hiddenDim": ${JSON.stringify(hiddenDim)},
        "depth": ${JSON.stringify(depth)},
        "numHeads": ${JSON.stringify(numHeads)},
        "seed": ${JSON.stringify(seed)},
    }
    torch.manual_seed(int(md["seed"]))
    return build_vit_from_md(md, in_channels=in_channels, num_classes=num_classes, image_size=image_size)
`;
}
