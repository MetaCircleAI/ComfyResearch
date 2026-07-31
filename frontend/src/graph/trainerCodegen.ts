import type { Node as RFNode } from "@xyflow/react";
import { defaultTrainerData, type TrainerNodeData } from "../components/nodes/trainerDefaults";
import { VISION_DATASET_KINDS } from "../components/nodes/visionDatasetDefaults";
import type { CodegenContext } from "./codegenContext";
import { pySlugForNode } from "./codegenContext";
import { TOY_LANGUAGE_DATASET_KINDS } from "./extraDatasetCodegen";

function upstreamFromTrainer(
  trainerId: string,
  targetHandle: string,
  ctx: CodegenContext | undefined,
): RFNode | undefined {
  if (!ctx) return undefined;
  const e = ctx.edges.find((x) => x.target === trainerId && (x.targetHandle ?? "") === targetHandle);
  if (!e) return undefined;
  return ctx.nodes.find((n) => n.id === e.source);
}

export function buildTrainerTorch(
  pySym: string,
  title: string,
  trainerNodeId: string,
  raw: Record<string, unknown>,
  ctx: CodegenContext | undefined,
): string {
  const defs = defaultTrainerData();
  const d = { ...defs, ...(raw as Partial<TrainerNodeData>) } as TrainerNodeData;
  const steps = Number.isFinite(d.trainingSteps) ? Math.max(1, Math.floor(d.trainingSteps)) : 1000;
  const logEvery = Number.isFinite(d.logFrequency) ? Math.max(1, Math.floor(d.logFrequency)) : 10;
  const trainerBatch = Number.isFinite(d.batchSize) ? Math.floor(d.batchSize as number) : -1;

  const allNodes = ctx?.nodes ?? [];

  const ds =
    upstreamFromTrainer(trainerNodeId, "dataset", ctx) ??
    upstreamFromTrainer(trainerNodeId, "train_dataset", ctx);
  const modelN = upstreamFromTrainer(trainerNodeId, "model", ctx);
  const optN = upstreamFromTrainer(trainerNodeId, "optimizer", ctx);
  const lossN = upstreamFromTrainer(trainerNodeId, "loss", ctx);

  const dsSym = ds ? pySlugForNode(ds.id, allNodes) : "";
  const modelSym = modelN ? pySlugForNode(modelN.id, allNodes) : "";
  const optSym = optN ? pySlugForNode(optN.id, allNodes) : "";
  const lossSym = lossN ? pySlugForNode(lossN.id, allNodes) : "";

  const exportLoadersDatasets = new Set<string>([
    "linear_dataset",
    "random_noise_dataset",
    "memorization_a_dataset",
    "memorization_b_dataset",
    "unigram_dataset",
    "modular_addition_dataset",
    "symbolic_func_dataset",
    "token_prediction_dataset",
    "circle_random_walk_dataset",
    "circular_motion_dataset",
    "kepler_2d_dataset",
    "uniform_linear_motion_dataset",
    "in_context_associative_recall_dataset",
    "bigram_low_rank_dataset",
    "teacher_dataset",
    "diffusion_pde_dataset",
    "reaction_diffusion_dataset",
    "advection_dataset",
    "random_input_distribution",
    "input_sampler",
    "dataset_mixer",
    "dataset_mixer_b",
    ...VISION_DATASET_KINDS,
    ...TOY_LANGUAGE_DATASET_KINDS,
  ]);
  const dsOk = Boolean(ds && exportLoadersDatasets.has(String(ds.type)));
  const trainerWiredModelTypes = new Set<string>([
    "mlp_model",
    "mlp_token_model",
    "gated_mlp_token_model",
    "moe_mlp_token_model",
  ]);
  const modelOk = Boolean(modelN && trainerWiredModelTypes.has(String(modelN.type)));
  const optOk =
    optN &&
    (optN.type === "adam_optimizer" || optN.type === "sgd_optimizer" || optN.type === "muon_optimizer");
  const lossOk =
    lossN &&
    (lossN.type === "mse_loss" || lossN.type === "cross_entropy_loss" || lossN.type === "kan_reg");

  const wired = Boolean(dsOk && modelN && optN && lossN && modelOk && optOk && lossOk && dsSym && modelSym && optSym && lossSym);

  if (wired) {
    const wiredTypes = [ds?.type, modelN?.type, optN?.type, lossN?.type].filter(Boolean).join(" → ");
    return `# === ${title} (trainer) ===
# Wired from graph: ${wiredTypes}. Run the cells for those node types above (same kernel) so the \`fn_*\` helpers exist.
# Matches canvas /api/train for full-batch runs: set trainer \`batchSize\` to -1 on the node (or mirror the loop below).
# Mini-batch: trainer \`batchSize\` ≥ 1 samples random indices each step; \`batch_size\` here still selects DataLoader batch for the exported loaders.
# Logging follows trainer logFrequency (multiples + step 0).
import torch


def fn_${pySym}_run(batch_size: int = 64, device: str | torch.device = "cpu"):
    train_loader, test_loader = fn_${dsSym}_loaders(batch_size=batch_size, device=device)
    x_train, y_train = train_loader.dataset.tensors
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    if test_loader is not None and len(test_loader.dataset) > 0:
        x_test, y_test = test_loader.dataset.tensors
        x_test = x_test.to(device)
        y_test = y_test.to(device)
    else:
        x_test = y_test = None

    model = fn_${modelSym}_model().to(device)
    opt = fn_${optSym}_optimizer((p for p in model.parameters() if p.requires_grad))
    loss_fn = fn_${lossSym}_criterion()
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)

    train_losses: list[float] = []
    test_losses: list[float] = []
    step_axis: list[int] = []

    def eval_test() -> float:
        if x_test is None or y_test is None:
            return float("nan")
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return float(loss_fn(model(x_test), y_test).item())
        finally:
            if was_training:
                model.train()

    model.eval()
    with torch.no_grad():
        train_losses.append(float(loss_fn(model(x_train), y_train).item()))
    step_axis.append(0)
    test_losses.append(eval_test())

    model.train()
    trainer_batch = ${trainerBatch}
    n_train = int(x_train.shape[0])
    for step in range(${steps}):
        opt.zero_grad(set_to_none=True)
        if trainer_batch > 0 and trainer_batch < n_train:
            g_step = torch.Generator(device=x_train.device)
            g_step.manual_seed((0x51ED7A77 + int(step)) & 0x7FFFFFFF)
            idx = torch.randperm(n_train, generator=g_step, device=x_train.device)[:trainer_batch]
            xb = x_train.index_select(0, idx)
            yb = y_train.index_select(0, idx)
        else:
            xb, yb = x_train, y_train
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        done_steps = step + 1
        if done_steps % ${logEvery} == 0:
            train_losses.append(float(loss.item()))
            step_axis.append(done_steps)
            test_losses.append(eval_test())

    return {"steps": step_axis, "train_loss": train_losses, "test_loss": test_losses, "model": model}


def fn_${pySym}_plot(result: dict):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    # ComfyResearch notebook: plt.show() is captured and shown under this cell (no extra window).
    steps_axis = result["steps"]
    plt.figure(figsize=(4.6, 2.6))
    plt.plot(steps_axis, result["train_loss"], label="train", color="#e74c3c")
    plt.plot(steps_axis, result["test_loss"], label="test", color="#3498db")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("${title}")
    plt.gca().margins(x=0.05, y=0.05)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
`;
  }

  return `# === ${title} (trainer) ===
# Standalone fallback (graph not wired when this cell was generated, or unsupported node mix).
# Wire dataset (including vision / unigram / modular addition / bigram low-rank) → MLP or MLP_token family → Adam/SGD/Muon → MSE/CE/KAN-reg → Trainer, then refresh this cell so it calls your dataset's \`fn_*_loaders\`.
import torch
import torch.nn as nn


def fn_${pySym}_run(
    *,
    device: str | torch.device = "cpu",
    training_steps: int = ${steps},
    log_every: int = ${logEvery},
    batch_size: int = 64,
):
    g = torch.Generator(device="cpu").manual_seed(0)
    x_train = torch.randn(800, 10, generator=g, device=device)
    y_train = torch.randn(800, 1, generator=g, device=device)
    x_test = torch.randn(200, 10, generator=g, device=device)
    y_test = torch.randn(200, 1, generator=g, device=device)

    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_losses: list[float] = []
    test_losses: list[float] = []
    steps_out: list[int] = []

    def eval_test() -> float:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return float(loss_fn(model(x_test), y_test).item())
        finally:
            if was_training:
                model.train()

    model.eval()
    with torch.no_grad():
        train_losses.append(float(loss_fn(model(x_train), y_train).item()))
    steps_out.append(0)
    test_losses.append(eval_test())

    model.train()
    for step in range(training_steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
        done_steps = step + 1
        if done_steps % log_every == 0:
            train_losses.append(float(loss.item()))
            steps_out.append(done_steps)
            test_losses.append(eval_test())

    return {"steps": steps_out, "train_loss": train_losses, "test_loss": test_losses, "model": model}


def fn_${pySym}_plot(result: dict):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    # ComfyResearch notebook: plt.show() is captured and shown under this cell (no extra window).
    steps_axis = result["steps"]
    plt.figure(figsize=(4.6, 2.6))
    plt.plot(steps_axis, result["train_loss"], label="train", color="#e74c3c")
    plt.plot(steps_axis, result["test_loss"], label="test", color="#3498db")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("${title}")
    plt.gca().margins(x=0.05, y=0.05)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
`;
}
