---
doc_type: reproduction
doc_status: phenomenon
template_id: b38ae9dd-735b-46c4-973f-a850a2a55544
---

:::{div} cr-eyebrow
Loss Geometry and Parameter Symmetry · Phenomenon reproduction
:::

# NEURAL MECHANICS: SYMMETRY AND BROKEN CONSERVATION LAWS IN DEEP LEARNING DYNAMICS

:::{div} cr-article-lead
Finite learning rates and weight decay break the conservation laws of gradient
flow. A VGG-11 on CIFAR-10 traces the full spectrum of Eq. 19: monotonic
growth, non-monotonic crossover, and monotonic decay of the weight norm.
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

Neural network architectures embed continuous differentiable symmetries  -- 
translation (softmax), scale (batch normalization), and rescale (ReLU)  --  that
impose geometric constraints on gradients and Hessians. Under gradient flow
($\eta \to 0$, $\lambda = 0$), each symmetry gives rise to a strict
conservation law. Finite learning rates and weight decay break these
conservation laws, yielding exact integral expressions for the dynamics of the
previously conserved quantities. This reproduction trains a VGG-11 on CIFAR-10
with SGD to verify the qualitative predictions of the scale symmetry dynamics
(Eq. 19) across three weight decay regimes.
:::

**Paper:** [Neural Mechanics: Symmetry and Broken Conservation Laws in Deep
Learning Dynamics](https://arxiv.org/abs/2012.04728), Kunin, Sagastuy-Brena,
Ganguli, Yamins & Tanaka, ICLR 2021 (arXiv:2012.04728).

**Template:** `Symmetry & Broken Conservation Laws (GPU)`

**Template ID:** `b38ae9dd-735b-46c4-973f-a850a2a55544`

## Reproduction Goal

Kunin et al. (2021) showed that any continuous differentiable symmetry of the
training loss imposes geometric constraints on gradients and Hessians, leading
to an associated conservation law in the continuous-time limit of SGD (gradient
flow), akin to Noether's theorem in physics. Finite learning rates and weight
decay break these conservation laws, and the resulting dynamics can be
described by exact integral expressions.

Three symmetries are identified in standard architectures:

| Symmetry | Origin | Conserved quantity (gradient flow) | Broken dynamics | Paper Eq. |
| --- | --- | --- | --- | --- |
| Translation | Softmax (classifier weights and bias) | $\langle \theta(t), \mathbf{1}_A \rangle$ | $\langle \theta(t), \mathbf{1}_A \rangle = e^{-\lambda t} \langle \theta(0), \mathbf{1}_A \rangle$ | (18) |
| Scale | BatchNorm ($\gamma, \beta$) | $\|\theta_A(t)\|^2$ | $\|\theta_A(t)\|^2 = e^{-2\lambda t} \|\theta_A(0)\|^2 + \eta \int_0^t e^{-2\lambda(t-\tau)} \|g_A\|^2 \mathrm{d}\tau$ | (19) |
| Rescale | ReLU-connected consecutive layers | $\|\theta_{A1}(t)\|^2 - \|\theta_{A2}(t)\|^2$ | same form with difference of gradient norms in the integral | (20) |

The time derivative of Eq. 19,

$$
\frac{d}{dt}\|\theta_A(t)\|^2 = -2\lambda \|\theta_A(t)\|^2 + \eta \|g_A\|^2,
$$

reveals a competition between two forces: a **centripetal effect** due to
weight decay ($-2\lambda \|\theta_A\|^2$) and a **centrifugal effect** due to
discretization ($\eta \|g_A\|^2$). At $\lambda = 0$, only the centrifugal term
operates and the norm grows monotonically. At sufficiently large $\lambda$, the
centripetal term dominates and the norm decays. At intermediate $\lambda$, the
two terms trade dominance as the gradient norm evolves through training,
producing non-monotonic dynamics.

This reproduction focuses on **scale symmetry** (Eq. 19) in the batch
normalization layers of a VGG-11 model, tracking the global weight L2 norm
$\|\theta(t)\|$ as a proxy for the aggregate of all BN scale parameter norms.
Three weight decay settings span the full spectrum: zero, intermediate, and
strong.

## Paper Experiment and Reproduction Boundary

| Item | Paper | Current Comfy Research template |
| --- | --- | --- |
| Claim | Finite learning rate and weight decay break the conservation laws of gradient flow; scale symmetry dynamics follow Eq. 19 with a competition between centripetal and centrifugal effects | Same qualitative signature: monotonic growth ($\lambda = 0$), non-monotonic ($\lambda$ small), monotonic decay ($\lambda$ large) |
| Model | VGG-16 with batch normalization, ~138M parameters | VGG-11 with batch normalization, ~9.7M parameters |
| Dataset | Tiny ImageNet (200 classes, 64×64, 100k samples) | CIFAR-10 (10 classes, 32×32,  | train /  | test) |
| Optimizer | SGD, $\eta = 0.1$, momentum 0, batch size 256 | Same settings |
| Weight decay | $\lambda \in \{0, 10^{-4}, 10^{-3}\}$ | $\lambda \in \{0, 5\times10^{-5}, 0.01\}$ |
| Training | 100 epochs | 8000 steps (fixed step budget) |
| Observable | Per-layer squared channel norms $\|W_l\|^2$ for specific conv layers | Global weight L2 norm (all BN layers aggregated) and gradient norm |
| Device | (not specified) | Remote GPU |
| Runtime goal | Empirical validation of Eq. 18-20 on large-scale models | Qualitative demonstration of the centripetal-centrifugal competition |

This graph tests a **mechanism-level pattern**  --  the qualitative dependence of
the weight norm trajectory on $\lambda$  --  rather than a point-by-point match
to the paper's Fig. 5. The use of global weight L2 norm (rather than
per-BN-layer norms) and a smaller model/dataset means the curves are not
directly comparable to the paper's, but the qualitative signatures of Eq. 19
are preserved.

## Experiment Configuration

| Item | Template setting |
| --- | --- |
| Dataset | CIFAR-10 (32×32 RGB, 10 classes),  | train /  | test samples |
| Model | VGG-11 CIFAR: 8 Conv2D layers each followed by BatchNorm2d + ReLU, with MaxPool after blocks 2/4/6/8; 3 FC layers (512→512→10); ~9.7M parameters |
| Loss | Cross-entropy (with implicit softmax → translation symmetry in classifier) |
| Optimizer | SGD, learning rate $\eta = 0.1$, momentum 0, weight decay $\lambda$ varied |
| Batch size | 256 |
| Device | Remote GPU |
| Training | 8000 steps, log every  | steps |
| Observables | Global weight L2 norm $\|\theta(t)\|$ (`observable_weight_l2`, `normAggregation=global`); Global gradient norm (`observable_gradient_norm`, normalized) |
| Experiment groups | **A**: $\eta = 0.1$, $\lambda = 0$ \| **B**: $\eta = 0.1$, $\lambda = 5\times10^{-5}$ \| **C**: $\eta = 0.1$, $\lambda = 0.01$ |

The global weight L2 norm $\|\theta(t)\| = \sqrt{\sum_p p^2}$ aggregates over
all trainable parameters (convolution kernels, BN $\gamma$ and $\beta$, FC
weights and biases). Since all BN layers share the same qualitative Eq. 19
dynamics, the global norm inherits the same qualitative dependence on $\eta$
and $\lambda$. The gradient norm serves as a diagnostic for the centrifugal
driving term: when $\|g\|^2$ is large, the integral term in Eq. 19 accumulates
rapidly; when it decays, the centripetal term may overtake the centrifugal term
if $\lambda > 0$.

## Run in Comfy Research

1. Open **Templates** and load **`Symmetry & Broken Conservation Laws (GPU)`**
   (or import the workspace JSON).
2. Confirm the graph: `cifar10_dataset` → `trainer`, `vgg11_cifar_model` →
   `trainer`, `cross_entropy_loss` → `trainer`, `sgd_optimizer` → `trainer`.
3. Verify Trainer settings: GPU, 8000 steps, batch size 256, momentum 0.
4. Verify observables: `observable_weight_l2` (global) and
   `observable_gradient_norm` (normalized) are registered in the Trainer's
   observables panel.
5. **Experiment A**: set SGD `weightDecay = 0`. Click **Train**.
6. **Experiment B**: set SGD `weightDecay = 5e-5`. Click **Train**.
7. **Experiment C**: set SGD `weightDecay = 0.01`. Click **Train**.
8. Compare the three runs in the Observables panel to see the full
   centripetal-centrifugal spectrum.

## Results

The global weight L2 norm $\|\theta(t)\|$ exhibits the three qualitative
regimes predicted by Eq. 19:

### Experiment A: $\lambda = 0$ (centrifugal effect only)

The weight L2 norm **rises monotonically** throughout training. With no weight
decay, the first term of Eq. 19 stays constant at $\|\theta(0)\|^2$ and the
integral term $\eta \int \|g\|^2 \mathrm{d}\tau$ accumulates monotonically,
driving the norm upward. The gradient norm peaks early (high initial loss) then
decays as training converges, but even the reduced late-training gradient
contributes positive accumulation to the integral  --  there is no countervailing
force.

:::{figure} ../../_images/app/vgg0.1_0.png
:alt: Comfy Research graph for Experiment A: VGG-11 on CIFAR-10, η = 0.1, λ = 0. Weight L2 norm rises monotonically.
:class: cr-reproduction-screenshot
:::

### Experiment B: $\lambda = 5 \times 10^{-5}$ (competitive regime)

The weight L2 norm **rises then falls**  --  a non-monotonic trajectory. Early in
training, the gradient norm is large and the centrifugal term
$\eta \|g\|^2$ exceeds the centripetal term $2\lambda \|\theta\|^2$, so the
norm grows. As training converges and $\|g\|^2$ decays, the centripetal term
gradually overtakes the centrifugal term, and the norm begins to decrease. The
peak marks the crossover where $2\lambda \|\theta\|^2 = \eta \|g\|^2$. This is
the most direct qualitative signature of the competition described by Eq. 19.

:::{figure} ../../_images/app/vgg0.1_0.00005.png
:alt: Comfy Research graph for Experiment B: VGG-11 on CIFAR-10, η = 0.1, λ = 5×10⁻⁵. Weight L2 norm rises then falls  --  non-monotonic dynamics.
:class: cr-reproduction-screenshot
:::

### Experiment C: $\lambda = 0.01$ (centripetal effect dominant)

The weight L2 norm **falls monotonically**. The exponential decay factor
$e^{-2\lambda t}$ in Eq. 19 suppresses both the memory term and the integral
term. With $\lambda = 0.01$, the centripetal effect dominates from the start  -- 
the decay rate outstrips any accumulation from the centrifugal term.

:::{figure} ../../_images/app/vgg0.1_0.01.png
:alt: Comfy Research graph for Experiment C: VGG-11 on CIFAR-10, η = 0.1, λ = 0.01. Weight L2 norm falls monotonically.
:class: cr-reproduction-screenshot
:::

### Summary

| Experiment | $\eta$ | $\lambda$ | $\|\theta(t)\|$ behavior | Dominant effect |
| --- | --- | --- | --- | --- |
| A | 0.1 | 0 | Monotonic rise | Centrifugal only |
| B | 0.1 | $5\times 10^{-5}$ | Rise then fall | Crossover: centrifugal → centripetal |
| C | 0.1 | 0.01 | Monotonic fall | Centripetal dominant |

## Interpretation

The three experiments span the full qualitative spectrum predicted by the scale
symmetry dynamics (Eq. 19):

- **Experiment A** ($\lambda = 0$): finite learning rate alone breaks the
  conservation law. Under gradient flow $\|\theta_A\|^2$ would be conserved, but
  discrete updates at finite $\eta$ introduce a centrifugal effect that pushes
  the norm monotonically upward.
- **Experiment B** ($\lambda = 5 \times 10^{-5}$): early in training, large
  gradients drive the centrifugal term to dominate; later, as gradients shrink,
  the centripetal term overtakes it. The crossover where the norm peaks is the
  dynamic balance $2\lambda \|\theta\|^2 = \eta \|g\|^2$.
- **Experiment C** ($\lambda = 0.01$): weight decay dominates the dynamics from
  the outset, pulling the norm down despite the centrifugal push.

## Limitations

- The **global weight L2 norm** aggregates all parameters  --  convolution
  kernels, BN $\gamma$/$\beta$, and FC weights  --  not solely the BN scale
  parameters that define the scale symmetry group. Each BN layer obeys Eq. 19
  independently; the global norm is a superposition and cannot be directly
  compared to the paper's per-layer curves.
- The **gradient norm** measures the full gradient across all parameters, not
  $\|g_A\|^2$ for a specific symmetry group. It serves as a qualitative proxy
  for the centrifugal driving term but cannot be used to quantitatively verify
  the integral in Eq. 19.
- This experiment only probes **scale symmetry** (Eq. 19). Translation symmetry
  (Eq. 18) and rescale symmetry (Eq. 20) are not directly measured  -- 
  translation because the `statistics` node in Comfy Research lacks a sum
  operation for computing $\langle \theta, \mathbf{1}_A \rangle$, and rescale
  because `statistics2` does not operate as a training-step observable.
- The model (VGG-11, ~9.7M parameters) and dataset (CIFAR-10) are smaller than
  the paper's VGG-16 + Tiny ImageNet setup, though the symmetry and its
  dynamic consequences are architectural properties that do not depend on
  scale.
- The three $\lambda$ values were chosen for clear qualitative separation
  rather than to match the paper's specific settings.
- Momentum and stochastic gradient noise  --  both modeled in the paper's full
  framework  --  are not explored here.

This result should be read as **a fast qualitative demonstration** of how
finite learning rate and weight decay break the scale symmetry conservation
law, producing the centripetal-centrifugal competition described by Eq. 19.
