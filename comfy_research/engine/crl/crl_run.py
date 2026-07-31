"""PyTorch CRL training loop (contrastive RL) — NDJSON compatible with ``/api/train``."""

from __future__ import annotations

import base64
import io
import random
from dataclasses import dataclass
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from fastapi import HTTPException

from comfy_research.engine.crl.crl_buffer import TrajectoryChunk, TrajectoryReplay, flatten_crl_torch
from comfy_research.engine.crl.crl_networks import CrlResidualAgent, parse_crl_residual_activation
from comfy_research.engine.crl.crl_point_u4_env import PointU4MazeEnv
from comfy_research.engine.runs.train_control import get_control, register_trainer, unregister_trainer
from comfy_research.engine.runs.trainer_run import find_loss_visualization_targets, observable_viz_metric_updates
from comfy_research.generated.node_capabilities import has_capability
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _node_map(nodes: list[Node]) -> dict[str, Node]:
    return {n.id: n for n in nodes}


def _incoming(edges: list[Edge], nodes: dict[str, Node], trainer_id: str, handle: str) -> Node | None:
    for e in edges:
        if e.target == trainer_id and e.targetHandle == handle:
            return nodes.get(e.source)
    return None


def _incoming_all(edges: list[Edge], nodes: dict[str, Node], trainer_id: str, handle: str) -> list[Node]:
    out: list[Node] = []
    seen: set[str] = set()
    for e in edges:
        if e.target == trainer_id and e.targetHandle == handle:
            n = nodes.get(e.source)
            if n is not None and n.id not in seen:
                seen.add(n.id)
                out.append(n)
    return out


def _scalar_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _scalar_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _weight_l2_norm(module: torch.nn.Module) -> float:
    s = 0.0
    with torch.no_grad():
        for p in module.parameters():
            s += float(p.detach().float().pow(2).sum().item())
    return float(s**0.5)


def _weight_l1_norm(module: torch.nn.Module) -> float:
    s = 0.0
    with torch.no_grad():
        for p in module.parameters():
            s += float(p.detach().float().abs().sum().item())
    return float(s)


def _pack_crl_checkpoint_b64(agent: CrlResidualAgent, optimizer: torch.optim.Optimizer) -> str:
    """Same container keys as supervised trainer checkpoints (``model`` / ``optimizer``)."""
    buf = io.BytesIO()
    torch.save({"model": agent.state_dict(), "optimizer": optimizer.state_dict()}, buf)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _loss_plot_png_b64(step_ticks: list[int], train_hist: list[float], aux_hist: list[float]) -> str:
    fig, ax = plt.subplots(figsize=(3.2, 1.35), dpi=120)
    ax.plot(step_ticks, train_hist, color="#c084fc", linewidth=1.6, label="critic")
    if aux_hist and len(aux_hist) == len(step_ticks):
        ax.plot(step_ticks, aux_hist, color="#60a5fa", linewidth=1.4, label="success")
    ax.margins(x=0.05, y=0.05)
    ax.set_xlabel("step", fontsize=8, color="#9a9aa8")
    ax.set_ylabel("value", fontsize=8, color="#9a9aa8")
    ax.tick_params(colors="#9a9aa8", labelsize=7)
    ax.set_facecolor("#121218")
    fig.patch.set_facecolor("#1a1a1f")
    for spine in ax.spines.values():
        spine.set_color("#3a3a44")
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.standard_b64encode(b.getvalue()).decode("ascii")


@dataclass
class CrlRunContext:
    nodes: list[Node]
    edges: list[Edge]
    trainer_node_id: str
    device: torch.device
    agent: CrlResidualAgent
    optimizer: torch.optim.Optimizer
    env: PointU4MazeEnv
    replay: TrajectoryReplay
    nmap: dict[str, Node]
    observable_nodes: list[Node]
    observable_metric_histories: dict[str, list[float]]
    training_steps: int
    log_frequency: int
    batch_size: int
    unroll_length: int
    gamma: float
    logsumexp_penalty_coeff: float
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    entropy_param: float
    disable_entropy: bool
    num_sgd_batches_per_step: int
    obs_dim: int
    goal_start: int
    goal_end: int
    action_dim: int
    action_size: int
    target_entropy: float
    critic_loss_history: list[float]
    success_history: list[float]
    step_ticks: list[int]


def prepare_crl_run(
    nodes: list[Node],
    edges: list[Edge],
    trainer_node_id: str,
    resume: dict[str, Any] | None = None,
) -> CrlRunContext:
    if resume:
        raise HTTPException(status_code=400, detail="Resume is not supported for crl_trainer yet.")
    nmap = _node_map(nodes)
    if trainer_node_id not in nmap:
        raise HTTPException(status_code=400, detail="trainer_node_id not found in nodes.")
    tr = nmap[trainer_node_id]
    if str(tr.type) != str(NodeKind.crl_trainer):
        raise HTTPException(status_code=400, detail="trainer_node_id must refer to a crl_trainer node.")

    opt_n = _incoming(edges, nmap, trainer_node_id, "optimizer")
    model_n = _incoming(edges, nmap, trainer_node_id, "model")
    env_n = _incoming(edges, nmap, trainer_node_id, "env")
    if opt_n is None or model_n is None or env_n is None:
        raise HTTPException(status_code=400, detail="crl_trainer requires wired optimizer, model, and env.")
    if opt_n.type != NodeKind.adam_optimizer:
        raise HTTPException(status_code=400, detail="CRL trainer currently supports adam_optimizer only.")
    if model_n.type != NodeKind.crl_residual_mlp:
        raise HTTPException(status_code=400, detail="model must be crl_residual_mlp.")
    if env_n.type != NodeKind.crl_env_config:
        raise HTTPException(status_code=400, detail="env must be crl_env_config.")

    md_t = tr.data or {}
    md_m = model_n.data or {}
    md_e = env_n.data or {}
    md_o = opt_n.data or {}

    training_steps = max(1, _scalar_int(md_t.get("trainingSteps"), 50))
    log_frequency = max(1, _scalar_int(md_t.get("logFrequency"), 5))
    batch_size = max(8, _scalar_int(md_t.get("batchSize"), 64))
    unroll_length = max(4, _scalar_int(md_t.get("unrollLength"), 32))
    num_sgd = max(1, _scalar_int(md_t.get("sgdStepsPerTrainStep"), 8))
    gamma = _scalar_float(md_t.get("gamma"), 0.99)
    logsumexp_penalty_coeff = _scalar_float(md_t.get("logsumexpPenaltyCoeff"), 0.1)
    entropy_param = _scalar_float(md_t.get("entropyParam"), 0.5)
    disable_entropy = bool(md_t.get("disableEntropy", False))

    num_envs = max(2, _scalar_int(md_e.get("numEnvs"), 8))
    episode_length = max(32, _scalar_int(md_e.get("episodeLength"), 200))
    maze_scale = _scalar_float(md_e.get("mazeSizeScaling"), 4.0)
    preset = str(md_e.get("preset") or "point_u4_maze").strip()
    if preset not in ("point_u4_maze", "ant_u4_maze"):
        raise HTTPException(status_code=400, detail="crl_env_config preset must be point_u4_maze or ant_u4_maze.")
    # ant_u4_maze uses same lightweight env until Brax is integrated
    seed_env = _scalar_int(md_e.get("seed"), 0)
    rng_np = np.random.default_rng(seed_env)
    env = PointU4MazeEnv(num_envs, scale=maze_scale, episode_length=episode_length, rng=rng_np)

    state_dim = _scalar_int(md_m.get("stateDim"), env.obs_dim_state)
    action_dim = _scalar_int(md_m.get("actionDim"), env.action_dim)
    goal_dim = _scalar_int(md_m.get("goalDim"), env.goal_dim)
    obs_dim_full = state_dim + goal_dim
    if obs_dim_full != env.obs_dim_state + env.goal_dim:
        pass  # allow mismatch if user overrides for custom env — PointU4 still uses env dims
    actor_w = max(8, _scalar_int(md_m.get("actorWidth"), 128))
    critic_w = max(8, _scalar_int(md_m.get("criticWidth"), 128))
    actor_depth = max(4, _scalar_int(md_m.get("actorDepth"), 4))
    critic_depth = max(4, _scalar_int(md_m.get("criticDepth"), 4))
    if actor_depth % 4 or critic_depth % 4:
        raise HTTPException(status_code=400, detail="actorDepth and criticDepth must be multiples of 4.")
    embed_dim = max(8, _scalar_int(md_m.get("embedDim"), 64))
    activation = parse_crl_residual_activation(md_m)
    model_seed = _scalar_int(md_m.get("seed"), 0)
    torch.manual_seed(model_seed)

    agent = CrlResidualAgent(
        state_dim=env.obs_dim_state,
        action_dim=env.action_dim,
        goal_dim=env.goal_dim,
        obs_dim_full=env.obs_dim_state + env.goal_dim,
        actor_width=actor_w,
        critic_width=critic_w,
        actor_depth=actor_depth,
        critic_depth=critic_depth,
        embed_dim=embed_dim,
        activation=activation,
    )

    actor_lr = _scalar_float(md_o.get("learningRate"), 3e-4)
    # allow separate critic LR via optional field on optimizer node
    critic_lr = _scalar_float(md_o.get("criticLearningRate"), actor_lr)
    alpha_lr = _scalar_float(md_o.get("alphaLearningRate"), actor_lr)

    optimizer = torch.optim.Adam(
        [
            {"params": agent.actor.parameters(), "lr": actor_lr},
            {"params": agent.sa_encoder.parameters(), "lr": critic_lr},
            {"params": agent.g_encoder.parameters(), "lr": critic_lr},
            {"params": [agent.log_alpha], "lr": alpha_lr},
        ]
    )

    dev_s = str(md_t.get("computeDevice") or "cpu").lower()
    device = torch.device("cuda" if dev_s == "cuda" and torch.cuda.is_available() else "cpu")
    agent = agent.to(device)

    observable_nodes = _incoming_all(edges, nmap, trainer_node_id, "observables")
    observable_metric_histories: dict[str, list[float]] = {on.id: [] for on in observable_nodes}

    goal_start = 0
    goal_end = env.goal_dim
    action_size = env.action_dim
    target_entropy = -entropy_param * float(action_size)

    ctx = CrlRunContext(
        nodes=nodes,
        edges=edges,
        trainer_node_id=trainer_node_id,
        device=device,
        agent=agent,
        optimizer=optimizer,
        env=env,
        replay=TrajectoryReplay(max_chunks=_scalar_int(md_t.get("maxReplayChunks"), 300)),
        nmap=nmap,
        observable_nodes=observable_nodes,
        observable_metric_histories=observable_metric_histories,
        training_steps=training_steps,
        log_frequency=log_frequency,
        batch_size=batch_size,
        unroll_length=unroll_length,
        gamma=gamma,
        logsumexp_penalty_coeff=logsumexp_penalty_coeff,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        entropy_param=entropy_param,
        disable_entropy=disable_entropy,
        num_sgd_batches_per_step=num_sgd,
        obs_dim=env.obs_dim_state,
        goal_start=goal_start,
        goal_end=goal_end,
        action_dim=action_dim,
        action_size=action_size,
        target_entropy=target_entropy,
        critic_loss_history=[],
        success_history=[],
        step_ticks=[],
    )
    return ctx


def _observable_snapshot(ctx: CrlRunContext) -> None:
    model = ctx.agent
    for on in ctx.observable_nodes:
        if not has_capability(str(on.type), "observable"):
            continue
        if on.type == NodeKind.observable_weight_l2:
            ctx.observable_metric_histories[on.id].append(_weight_l2_norm(model))
        elif on.type == NodeKind.observable_weight_l1:
            ctx.observable_metric_histories[on.id].append(_weight_l1_norm(model))


def iter_crl_events_from_context(ctx: CrlRunContext) -> Iterator[dict[str, Any]]:
    trainer_node_id = ctx.trainer_node_id
    register_trainer(trainer_node_id)
    nodes = ctx.nodes
    edges = ctx.edges
    nmap = ctx.nmap
    device = ctx.device
    agent = ctx.agent
    optimizer = ctx.optimizer
    env = ctx.env
    replay = ctx.replay
    rng_py = random.Random(_scalar_int((ctx.nmap[trainer_node_id].data or {}).get("seed"), 0))

    try:
        st = env.reset()
        obs_np = env.observe(st)
        total = ctx.training_steps
        ctx.step_ticks.append(0)
        ctx.critic_loss_history.append(0.0)
        ctx.success_history.append(0.0)
        _observable_snapshot(ctx)

        for outer in range(total):
            ctrl = get_control(trainer_node_id)
            if ctrl is not None and ctrl.abort_requested:
                yield {"type": "aborted"}
                return

            # collect unroll
            obs_hist = [obs_np]
            acts: list[np.ndarray] = []
            rews: list[np.ndarray] = []
            discs: list[np.ndarray] = []
            seeds_hist = [st.seed.copy()]
            success_roll = 0.0
            n_roll = 0
            for _t in range(ctx.unroll_length):
                o_t = obs_hist[-1]
                obs_t = torch.from_numpy(o_t).to(device)
                state_t = obs_t[:, : ctx.obs_dim]
                goal_t = obs_t[:, ctx.obs_dim : ctx.obs_dim + (ctx.goal_end - ctx.goal_start)]
                obs_actor = torch.cat([state_t, goal_t], dim=-1)
                with torch.no_grad():
                    mean, log_std = agent.actor(obs_actor)
                    std = log_std.clamp(-5, 2).exp()
                    noise = torch.randn_like(mean)
                    x_t = mean + std * noise
                    act_t = torch.tanh(x_t)
                act_np = act_t.cpu().numpy()
                st, onext, rew, disc = env.step(st, act_np)
                dist = np.linalg.norm(st.pos - st.goal, axis=-1)
                success_roll += float((dist < 0.35 * env.scale).mean())
                n_roll += 1
                obs_hist.append(onext)
                acts.append(act_np.astype(np.float32))
                rews.append(rew.astype(np.float32))
                discs.append(disc.astype(np.float32))
                seeds_hist.append(st.seed.copy())
                obs_np = onext

            obs_chunk = np.stack(obs_hist, axis=0)
            act_chunk = np.stack(acts, axis=0)
            rew_chunk = np.stack(rews, axis=0)
            disc_chunk = np.stack(discs, axis=0)
            seed_chunk = np.stack(seeds_hist, axis=0)
            replay.insert(
                TrajectoryChunk(
                    obs=obs_chunk,
                    act=act_chunk,
                    reward=rew_chunk,
                    discount=disc_chunk,
                    seed=seed_chunk,
                )
            )

            mean_critic = 0.0
            if len(replay) > 0:
                for _ in range(ctx.num_sgd_batches_per_step):
                    win = replay.sample_windows(rng_py, batch_size=ctx.batch_size, unroll=ctx.unroll_length)
                    if win is None:
                        break
                    # win.obs: [U+1, B, D], win.act: [U, B, A]
                    obs_bt = torch.from_numpy(np.asarray(win.obs, dtype=np.float32)).to(device).permute(1, 0, 2).contiguous()
                    act_bt = torch.from_numpy(np.asarray(win.act, dtype=np.float32)).to(device).permute(1, 0, 2).contiguous()
                    new_obs, new_act = flatten_crl_torch(
                        obs_bt,
                        act_bt,
                        gamma=ctx.gamma,
                        obs_dim=ctx.obs_dim,
                        goal_start=ctx.goal_start,
                        goal_end=ctx.goal_end,
                    )
                    b2, lm1, dtot = new_obs.shape
                    state = new_obs[:, :, : ctx.obs_dim].reshape(b2 * lm1, ctx.obs_dim)
                    goal = new_obs[:, :, ctx.obs_dim :].reshape(b2 * lm1, ctx.goal_end - ctx.goal_start)
                    act2 = new_act.reshape(b2 * lm1, ctx.action_dim)
                    sa = agent.sa_encoder(state, act2)
                    gg = agent.g_encoder(goal)
                    logits = -torch.sqrt(torch.sum((sa[:, None, :] - gg[None, :, :]) ** 2, dim=-1) + 1e-8)
                    critic_loss = -(torch.diagonal(logits) - torch.logsumexp(logits, dim=1)).mean()
                    lse = torch.logsumexp(logits + 1e-6, dim=1)
                    critic_loss = critic_loss + ctx.logsumexp_penalty_coeff * (lse**2).mean()
                    obs_pi = new_obs.reshape(b2 * lm1, dtot)
                    mean, log_std = agent.actor(obs_pi)
                    std = log_std.clamp(-5, 2).exp()
                    z = torch.randn_like(mean)
                    x_pre = mean + std * z
                    a_pi = torch.tanh(x_pre)
                    log_prob = torch.distributions.Normal(mean, std).log_prob(z) - torch.log(1 - a_pi.pow(2) + 1e-6)
                    log_prob = log_prob.sum(-1)
                    sa_pi = agent.sa_encoder(state, a_pi)
                    qf_pi = -torch.sqrt(torch.sum((sa_pi - gg) ** 2, dim=-1) + 1e-8)
                    if ctx.disable_entropy:
                        actor_loss = -qf_pi.mean()
                        alpha_loss = torch.zeros((), device=device)
                    else:
                        actor_loss = (agent.alpha() * log_prob - qf_pi).mean()
                        alpha_loss = (agent.alpha() * (-log_prob.detach() - ctx.target_entropy)).mean()

                    optimizer.zero_grad(set_to_none=True)
                    (critic_loss + actor_loss + alpha_loss).backward()
                    optimizer.step()
                    mean_critic += float(critic_loss.detach().cpu())

            done_steps = outer + 1
            if ctx.num_sgd_batches_per_step > 0 and len(replay) > 0:
                mean_critic /= ctx.num_sgd_batches_per_step

            if done_steps % ctx.log_frequency == 0 or done_steps == 1:
                ctx.step_ticks.append(done_steps)
                ctx.critic_loss_history.append(float(mean_critic))
                ctx.success_history.append(float(success_roll / max(1, n_roll)))
                _observable_snapshot(ctx)
                yield {"type": "progress", "step": done_steps, "total": total}

        plot_b64 = _loss_plot_png_b64(ctx.step_ticks, ctx.critic_loss_history, ctx.success_history)
        viz_ids = find_loss_visualization_targets(edges, nodes, trainer_node_id)
        obs_updates = observable_viz_metric_updates(
            edges,
            nmap,
            trainer_node_id,
            ctx.observable_metric_histories,
            {},
        )
        yield {
            "type": "complete",
            "checkpoint_b64": _pack_crl_checkpoint_b64(agent, optimizer),
            "loss_history": ctx.critic_loss_history,
            "test_loss_history": ctx.success_history,
            "step_ticks": ctx.step_ticks,
            "plot_png_base64": plot_b64,
            "visualization_node_ids": viz_ids,
            "observable_viz_updates": obs_updates,
            "observable_metric_histories": ctx.observable_metric_histories,
            "observable_embedding_histories": {},
        }
    finally:
        unregister_trainer(trainer_node_id)
