"""Saxe (orthogonal) initialization for deep linear / MLP networks.

Saxe, McClelland & Ganguli (2013) show that gradient descent on a deep linear
network trained from **orthogonal** initial weights (scaled by a small amplitude ε)
produces *exact* sequential mode learning: the k-th singular value of W_eff is
learned only after the (k−1)-th has saturated.

The initialization is:
  - Each nn.Linear weight matrix W is set to ``amplitude * Q`` where Q is drawn
    from the Haar measure on the orthogonal group (rectangular: left-singular
    vectors of a random Gaussian matrix).
  - Biases are zeroed.
  - Non-Linear layers (LayerNorm, Embedding, …) are left at their PyTorch defaults.
"""

from __future__ import annotations

import torch.nn as nn


def apply_saxe_init(model: nn.Module, amplitude: float = 0.01) -> None:
    """Re-initialise all nn.Linear weights to ``amplitude * Q`` (Q orthogonal).

    Parameters
    ----------
    model:
        Any nn.Module.  Only nn.Linear submodules are touched.
    amplitude:
        Scalar ε that scales the orthogonal matrix.  The default 0.01 keeps
        the initial effective product W_eff ≈ 0 so that the theory in Saxe et al.
        applies cleanly.  Larger values (e.g. 0.1) speed up early learning but
        may weaken the sequential-learning property.
    """
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            nn.init.orthogonal_(mod.weight, gain=float(amplitude))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
