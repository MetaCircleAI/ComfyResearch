"""Cross-walk to `wang-kevin3290/scaling-crl` (JAX / Flax) for CRL reproduction in PyTorch.

Reference (public repo, no vendoring required):
  https://github.com/wang-kevin3290/scaling-crl

Key mappings used by ComfyResearch `crl_run` / `crl_networks` / `crl_buffer`:

- **Ant U4 maze env id:** `ant_u4_maze` → `train.py` `make_env`: ``AntMaze(..., maze_layout_name=env_id[4:])`` i.e. ``u4_maze`` layout in ``envs/ant_maze.py`` (`U4_MAZE` grid, Brax ``spring`` backend in upstream code).
- **Obs / goal indices (ant maze):** `obs_dim=29`, `goal_start_idx=0`, `goal_end_idx=2` (xy goal slice inside processed states; see ``buffer.TrajectoryUniformSamplingQueue.flatten_crl_fn``).
- **Residual stack:** ``train.py`` `residual_block`: four ``Dense → LayerNorm → activation`` inside a residual skip; ``network_depth`` counts **Dense** layers: one initial ``Dense(width)``, then ``network_depth // 4`` blocks × 4 Denses each. ``SA_encoder`` / ``G_encoder`` / ``Actor`` share this pattern; output embedding dim **64**.
- **InfoNCE (critic):** logits ``[B,B]`` with ``logits[i,j] = -||φ(s_i,a_i) - ψ(g_j)||_2``; loss ``-mean(diag(logits) - logsumexp(logits, axis=1))`` plus optional ``logsumexp_penalty_coeff * mean(logsumexp(logits+1e-6)^2)``.
- **Actor (+ entropy):** Gaussian policy with ``tanh`` squashing; ``qf_pi = -||φ(s,π(s)) - ψ(g)||_2``; loss ``mean(exp(log_alpha)*log_prob - qf_pi)`` unless entropy disabled.
- **Replay + HER:** ``buffer.flatten_crl_fn`` — discounted future mask within same trajectory seed, categorical sample of future index, build ``obs = concat(state, goal)`` for policy; critic uses ``state`` from first ``obs_dim`` and ``goal`` from appended dims.

ComfyResearch uses a **PyTorch** trainer (`crl_run.py`) and a **lightweight ``point_u4_maze``** preset (continuous 2D particle, same ``U4_MAZE`` topology) when full Brax/MuJoCo is not installed; hparams default toward ``scaling-crl`` ``Args`` (``gamma``, ``batch_size``, LRs, ``logsumexp_penalty_coeff``, etc.) with smaller ``num_envs`` / SGD counts for local CPU runs.
"""
