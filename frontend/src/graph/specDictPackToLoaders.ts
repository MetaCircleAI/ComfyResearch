/** Wrap a Python ``def SpecName(...): ... return {"x_train": ...}`` into ``fn_<slug>_loaders`` for the Code tab. */

export type TorchDtypeLiteral = "float32" | "float64" | "int64";

export type DictPackLoaderPlan = {
  xTrain: TorchDtypeLiteral;
  yTrain: TorchDtypeLiteral;
  xTest: TorchDtypeLiteral;
  yTest: TorchDtypeLiteral;
};

const TORCH_DTYPE: Record<TorchDtypeLiteral, string> = {
  float32: "torch.float32",
  float64: "torch.float64",
  int64: "torch.int64",
};

function packBranch(name: string, dt: TorchDtypeLiteral): string {
  const td = TORCH_DTYPE[dt];
  return `    ${name} = pack["${name}"]\n    if ${name} is None:\n        ${name}_t = None\n    else:\n        ${name}_t = torch.as_tensor(${name}, device=device, dtype=${td})`;
}

/** ``specBody`` must define ``def ${specFnName}(...):`` returning dict with optional None test arrays. */
export function wrapDictReturningSpecAsLoaders(opts: {
  title: string;
  nodeType: string;
  pySym: string;
  specBody: string;
  specFnName: string;
  plan: DictPackLoaderPlan;
  extraHeaderLines?: string[];
}): string {
  const { title, nodeType, pySym, specBody, specFnName, plan, extraHeaderLines = [] } = opts;
  const xh = extraHeaderLines.length ? extraHeaderLines.map((l) => `# ${l}`).join("\n") + "\n" : "";
  return `# === ${title} (${nodeType}) ===
${xh}import torch
from torch.utils.data import DataLoader, TensorDataset

${specBody}

def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    pack = ${specFnName}()
${packBranch("x_train", plan.xTrain)}
${packBranch("y_train", plan.yTrain)}
${packBranch("x_test", plan.xTest)}
${packBranch("y_test", plan.yTest)}
    train_ds = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    if x_test_t is not None and y_test_t is not None and int(x_test_t.shape[0]) > 0:
        test_ds = TensorDataset(x_test_t, y_test_t)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
    else:
        test_loader = None
    return train_loader, test_loader
`;
}
