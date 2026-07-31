from fastapi import APIRouter

from comfy_research.config.remote_train_config import get_effective_remote_train_config

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    cfg = get_effective_remote_train_config()
    return {
        "ok": True,
        "mode": "remote" if cfg.is_active() else "local",
        "remoteHost": cfg.host,
        "remoteSource": cfg.source,
        "lastValidationOk": cfg.last_validation_ok,
    }
