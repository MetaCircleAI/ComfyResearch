"""Device resolution + reproducibility setup for the trainer runtime (extracted from trainer_run)."""

import os

import torch
from fastapi import HTTPException

from comfy_research.engine.trainer.scalar import _scalar_int
from comfy_research.schemas.graph import Node


def _resolve_trainer_compute_device(spec: str) -> torch.device:
    """Trainer node ``computeDevice``: auto / cpu / cuda / cuda:N / mps."""
    s = (spec or "auto").strip().lower() or "auto"
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if s == "cpu":
        return torch.device("cpu")
    if s == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise HTTPException(
                status_code=400,
                detail="Trainer computeDevice is mps but MPS is not available in this PyTorch build.",
            )
        return torch.device("mps")
    if s == "cuda":
        if not torch.cuda.is_available():
            raise HTTPException(status_code=400, detail="Trainer computeDevice is cuda but CUDA is not available.")
        return torch.device("cuda")
    if s.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise HTTPException(
                status_code=400,
                detail="Trainer computeDevice requests CUDA but CUDA is not available.",
            )
        return torch.device(s)
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unknown trainer computeDevice {spec!r}; use auto, cpu, cuda, cuda:N (e.g. cuda:0), or mps."
        ),
    )


def _optimizer_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for group in opt.param_groups:
        for p in group.get("params", ()):
            if p is None:
                continue
            st = opt.state.get(p)
            if not st:
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.to(device)


def _trainer_force_single_thread_cpu() -> None:
    """Parallel CPU BLAS/OpenMP reorders reductions → bitwise drift; force single thread for stable reruns."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = False
    except Exception:
        pass


def _trainer_configure_reproducible_torch(device: torch.device, *, rng_seed: int) -> None:
    """Best-effort repeatable numerics for identical graphs + seeds (CUDA SDP / TF32 / cuDNN / CPU threads)."""
    seed_u64 = int(rng_seed) & 0xFFFFFFFFFFFFFFFF

    if device.type == "cpu":
        _trainer_force_single_thread_cpu()

    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass

    torch.manual_seed(seed_u64)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_u64)
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed_u64)
        except Exception:
            pass


def _trainer_master_rng_seed(dataset_seed: int, model_node: Node) -> int:
    """Stable 64-bit mix for global PyTorch RNG and minibatch shuffles.

    Dataset tensors keep using ``dataset_seed`` via ``np.random.default_rng(dataset_seed)`` and model init
    still uses ``model.data.seed`` when present (see branches that call ``torch.manual_seed`` before build).

    When the model node has **no** ``seed`` field, returns ``dataset_seed`` unchanged so prior runs match
    legacy behavior. When ``seed`` is set on the model, it is XOR-mixed with the dataset base seed so **both**
    knobs affect training-loop global RNG (e.g. dropout) and ``minibatch_perm_seed``.
    """
    ds = int(dataset_seed) & 0xFFFFFFFFFFFFFFFF
    md = model_node.data or {}
    if "seed" not in md:
        return int(ds)
    ms = int(_scalar_int(md.get("seed"), 0)) & 0xFFFFFFFFFFFFFFFF
    return int((ds ^ ms ^ ((ds << 18) & 0xFFFFFFFFFFFFFFFF) ^ (ms >> 11)) & 0xFFFFFFFFFFFFFFFF)
