---
doc_type: reproduction
doc_status: phenomenon
template_id: e399fd7d-e107-44d0-94b6-7e2159392253
online_template_id: de684a36-a2d5-440f-bb2b-3c249abb8270
---

:::{div} cr-eyebrow
Learning Phases and Feature Formation · Phenomenon reproduction
:::

# Exact Solution for On-Line Learning in Multilayer Neural Networks

:::{div} cr-article-lead
On-line gradient-descent learning in a teacher-student soft-committee machine
exhibits a plateau in the generalization error when the student has more hidden
nodes than the teacher ($K > M$), before a transition onto specialization
eliminates the unnecessary node.
:::

::::::{div} cr-article-meta
:::::{div} cr-meta-person
::::{div} cr-avatar
ZH
::::
::::{div} cr-meta-copy
:::{div} cr-meta-label
Author
:::
**赵浩然 (Zhao Haoran)**
::::
:::::
:::::{div} cr-meta-scope
::::{div} cr-meta-copy
:::{div} cr-meta-label
Scope
:::
**Phenomenon reproduction**
::::
:::::
::::::

:::{admonition} Abstract
:class: cr-abstract

On-line gradient-descent learning in a teacher-student soft-committee machine
exhibits trapping in a symmetric subspace when the student has more hidden
nodes than the teacher ($K > M$). The generalization error exhibits a plateau  -- 
remaining nearly constant over an extended range of the normalized number of
examples $\alpha$  --  before a transition onto specialization, after which the
unnecessary student node is eliminated and the generalization error drops
toward zero. This is a qualitative reproduction of the over-realizable learning
dynamics reported in Saad & Solla (1995).
:::

**Paper:** [Exact Solution for On-Line Learning in Multilayer Neural
Networks](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.74.4337),
Saad & Solla, Phys. Rev. Lett. 74, 4337-4340 (1995).

**Template:** `Saad & Solla Over-Realizable (CPU)`

**Template ID:** `e399fd7d-e107-44d0-94b6-7e2159392253`

## Reproduction Goal

Saad & Solla (1995) derived coupled first-order differential equations
describing the dynamics of on-line gradient-descent learning in a two-layer
soft-committee machine in the thermodynamic limit $N \to \infty$. In the
**over-realizable** case ($K > M$, more student hidden nodes than teacher
hidden nodes), the order parameters $Q_{ik}$ and $R_{in}$ evolve through the
following stages:

1. **Undifferentiated symmetric solution**  --  all student weight vectors grow
   with no differentiation among hidden nodes; the generalization error
   decreases rapidly at first, then stalls.
2. **Trapping in the symmetric subspace (plateau)**  --  the student remains
   trapped near a symmetric fixed point of the dynamical equations. The
   generalization error exhibits a plateau, staying nearly constant over an
   extended interval of the normalized number of examples $\alpha = P/N$.
3. **Transition onto specialization**  --  the symmetric solution becomes
   unstable; student hidden nodes begin to specialize, each aligning to a
   different teacher node.
4. **Elimination of unnecessary nodes**  --  the redundant ($K - M$) student node
   decays toward zero norm ($Q_{33} \to 0$).
5. **Perfect generalization**  --  the surviving student nodes perfectly match
   the teacher nodes, and the generalization error $\to 0$.

The reproduction graph records the test error ($\frac{1}{2}$ × MSE) over the
normalized number of examples $\alpha$, using tanh activations with the
paper's overlap-aware initialization and fixed $+1$ readout couplings.

## Paper Experiment and Reproduction Boundary

| Item | Paper | Current Comfy Research template |
| --- | --- | --- |
| Claim | Over-realizable on-line learning ($K > M$) exhibits a plateau in the generalization error (trapping in a symmetric subspace), followed by transition onto specialization and elimination of unnecessary nodes | Same qualitative signature: plateau → transition → specialization → elimination |
| Activation | $\operatorname{erf}(x / \sqrt{2})$ | $\tanh$ with gain $\sqrt{2/\pi}$ (slope-matched to erf at zero) |
| Architecture | Soft-committee machine: single hidden layer, output couplings fixed to $+1$, only input-to-hidden couplings are adaptive | Same: single hidden layer, fixed $+1$ readout, only input-to-hidden weights trainable |
| Data & Limit | Gaussian i.i.d. inputs, $N \to \infty$ thermodynamic limit | Gaussian i.i.d. inputs, $N = 300$ |
| Optimizer | On-line gradient descent, learning rate $\eta$ scaled with $1/N$ | SGD, batch size 1, learning rate $\eta/N$ |
| Error | $\frac{1}{2} \times$ quadratic deviation (Eq. 1) | Same: `mse_loss` with `lossScale = 0.5` |
| Initialization | $R_{in} = 0$ for all $i, n$, $Q_{ik} = 0$ for all $i \neq k$, $Q_{ii}$ drawn independently from uniform distribution in $[0, 0.5]$ | Same overlap geometry enforced via explicit weight payloads |
| Teacher overlaps | $T_{nm} = n\,\delta_{nm}$, i.e. $\operatorname{diag}(1, 2)$ | Same, encoded in teacher's first-layer weights |
| Observables | Order parameters $Q_{ik}$, $R_{in}$, and generalization error $\varepsilon_g$ | Test error ($\frac{1}{2}$ × MSE), train error, weight L2 norm, and train-test gap |

This graph tests a **mechanism-level pattern**, not a point-by-point match to
the paper's Fig. 1. The fixed seed and overlap-aware initialization make the
dynamics reproducible and easy to inspect, but do not establish robustness
across random seeds.

## Experiment Configuration

| Item | Template setting |
| --- | --- |
| Dataset | Fixed: &nbsp; training samples drawn from $\mathcal{N}(0, I_N)$, sampled with replacement each step. Test evaluation uses a separate fixed pool of 30,000 samples. Seed 23. |
| Input dimension | $N = 300$ |
| Teacher network | Soft-committee machine, $M = 2$ hidden nodes, tanh activation, fixed $+1$ readout, weights frozen (`freeze: true`). Teacher overlaps: $T = \operatorname{diag}(1, 2)$. |
| Student network | Soft-committee machine, $K = 3$ hidden nodes, tanh activation, fixed $+1$ readout. Only the input-to-hidden couplings are adaptive. Initialized with $R_{in} = 0$, off-diagonal $Q_{ik} = 0$. |
| Activation scaling | `tanh(gain * x)` with `gain = sqrt(2/π)`, slope-matched to $\operatorname{erf}(x/\sqrt{2})$ at zero. The gain is folded into the first-layer weight payloads. |
| Objective | $\frac{1}{2} \times$ MSE (`mse_loss.lossScale = 0.5`) |
| Optimizer | SGD, learning rate $\eta/N= (2/\pi)/300 \approx 0.0021$, momentum 0, weight decay 0 |
| Batch size | 1 |
| Device | CPU |
| Training | 60000 steps, log every 100 steps, $\alpha = P/N = 200$ |
| Observables | Test error ($\frac{1}{2}$ × MSE), train error, train-test gap, weight L2 norm |

A separate large test dataset (wired via a second `teacher_dataset` as
`test_dataset`) keeps the test-error curve smooth enough to resolve the
trapping regime without single-sample noise.

The teacher weights are frozen (`trainable: false`, `freeze: true`,
`requiresGrad: false`), as are both the teacher and student readout couplings
(fixed at $+1$). Only the student's input-to-hidden weight matrix is adaptive
 --  matching the paper's soft-committee machine in which "all the hidden units
are connected to the output unit with positive couplings of unit strength, and
only the input-to-hidden couplings are adaptive."

## Run in ComfyResearch

1. Open **Templates** and load **`Saad & Solla Over-Realizable (CPU)`** (or
   import the workspace JSON).
2. Confirm Trainer settings: CPU, batch size 1, fixed sampling mode on the
   teacher dataset.
3. Verify that both the Teacher and Student `combined_model` nodes contain the
   atomic layers: `linear_layer` (input-to-hidden) → `activation_layer` (tanh)
   → `linear_layer` (fixed $+1$ readout).
4. Confirm that only the student's first `linear_layer` is marked trainable;
   teacher layers and both readout layers should show `freeze: true` /
   `trainable: false`.
5. Click **Train** on the `Trainer` node.
6. Inspect the test error curve in **Training Viz**. The plateau should be
   visible as a long, nearly flat segment.
7. Inspect the weight L2 norm in **Observable viz** to see the overall
   magnitude of the student's adaptive couplings.
8. Optionally, inspect the train-test gap.

## Results

The test error curve exhibits the qualitative signature predicted by Saad &
Solla (1995) for the over-realizable case ($K = 3 > M = 2$):

1. **Initial decay**  --  the generalization error drops rapidly at small
   $\alpha$.
2. **Trapping in the symmetric subspace (plateau)**  --  the error enters a long,
   nearly flat plateau where the student is trapped near a symmetric fixed
   point. All three student hidden nodes have approximately equal norms
   ($Q_{11} \approx Q_{22} \approx Q_{33}$), with no differentiation among
   them. Because no student node has committed to a particular teacher node,
   the gradient signal is weak and the generalization error decreases only very
   slowly. This corresponds to the "small $\alpha$ behavior dominated by an
   undifferentiated symmetric solution" described in the paper.
3. **Transition onto specialization**  --  the symmetric solution gives way to
   differentiation: student hidden nodes begin to specialize, each aligning to
   a different teacher node. The generalization error resumes a sustained
   decrease.
4. **Elimination of the unnecessary node**  --  the redundant third student node
   decays toward zero ($Q_{33} \to 0$), while the two surviving nodes imitate
   the two teacher nodes. Asymptotically, $Q_{11} = R_{11} = T_{11}$ and
   $Q_{22} = R_{22} = T_{22}$, with vanishing correlations between the
   surviving student vectors ($Q_{12} \to 0$).
5. **Perfect generalization**  --  the generalization error approaches zero as the
   student perfectly matches the teacher after eliminating the unnecessary
   node. This is the defining signature of the over-realizable regime: the
   student has sufficient resources ($K > M$) to implement the task exactly.

:::{figure} ../../_images/app/1995exp.png
:alt: Comfy Research graph showing test error over the normalized number of examples, with a visible plateau corresponding to trapping in the symmetric subspace, followed by a transition onto specialization and convergence toward perfect generalization.
:class: cr-reproduction-screenshot
:::

## Interpretation

The observed curve is qualitatively consistent with the dynamical picture of
Saad & Solla (1995):

- The **trapping in the symmetric subspace** (visible as a **plateau** in the
  error curve) reflects the dynamics near a symmetric fixed point of the
  order-parameter ODEs (Eq. 8), where all student weight vectors have
  approximately equal overlaps with each teacher node. In this regime the
  gradient signal is weak because no student node has committed to a specific
  teacher direction.
- The **transition onto specialization** corresponds to the destabilization of
  the symmetric fixed point: small fluctuations (amplified by the finite input
  dimension and the stochasticity of on-line sampling) eventually push one
  student node to align more strongly with a particular teacher node,
  triggering a positive-feedback cascade of differentiation.
- The **convergence to perfect generalization** is expected in the
  over-realizable case: the student has more hidden nodes than the teacher
  ($K = 3 > M = 2$) and, after eliminating the unnecessary node, can represent
  the teacher exactly.

Key design choices that were necessary to reproduce the paper's qualitative
behavior:

- **tanh activation** (with erf-matched gain): ReLU lacks saturation and does
  not support the symmetric fixed point required for trapping. The original
  paper uses erf; tanh with slope-matching gain $\sqrt{2/\pi}$ at zero is the
  closest available option in Comfy Research.
- **Fixed $+1$ readout**: the paper's soft-committee machine connects all
  hidden units to the output with "positive couplings of unit strength."
  Allowing the readout to train would give the student additional adaptive
  degrees of freedom beyond what the paper's theory describes.
- **Overlap-aware initialization**: the paper's initial conditions
  ($R_{in} = 0$, off-diagonal $Q_{ik} = 0$) are essential for the dynamics to
  pass through the symmetric subspace. Random initialization would mix these
  overlaps and obscure the trapping and transition.
- **Sufficiently large input dimension $N$**: smaller $N$ introduces strong
  $O(1/\sqrt{N})$ fluctuations that can prematurely destabilize the symmetric
  solution, shortening or eliminating the trapping regime.
- **Appropriate learning rate $\eta$**: too small a learning rate produces
  smooth descent with no discernible trapping; the discrete-time dynamics need
  a sufficiently large step for the student to remain trapped near the
  symmetric fixed point.

## Limitations

- The tanh activation is an approximation to the paper's erf; while the slope
  at zero is matched via the gain factor $\sqrt{2/\pi}$, the saturation
  behavior differs for large inputs, which affects the exact duration of the
  trapping regime and the location of the transition.
- The finite input dimension introduces $O(1/\sqrt{N})$ fluctuations not
  present in the thermodynamic limit $N \to \infty$. These fluctuations both
  enable the escape from the symmetric subspace (which would be perfectly
  stable in the limit) and add noise to the error curve.
- One fixed seed and one set of initial $Q_{ii}$ values do not measure
  initialization-to-initialization variation. Different initial conditions can
  produce trapping regimes of different lengths, and some seeds may escape the
  symmetric subspace earlier or later.
