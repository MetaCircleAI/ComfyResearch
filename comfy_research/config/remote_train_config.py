from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class RemoteTrainConfig:
    host: str = ""
    user: str = "ubuntu"
    remote_path: str = ""
    python: str = "/root/miniconda3/bin/python3"
    identity: str = ""
    password: str = ""
    extra_opts: str = ""
    enabled: bool = False
    upload_dataset: bool = False
    source: str = "none"
    last_validation_ok: bool | None = None
    last_validation_error: str = ""

    def normalized(self) -> "RemoteTrainConfig":
        user = self.user.strip() or "ubuntu"
        python = self.python.strip() or "/root/miniconda3/bin/python3"
        return RemoteTrainConfig(
            host=self.host.strip(),
            user=user,
            remote_path=self.remote_path.strip(),
            python=python,
            identity=self.identity.strip(),
            password=self.password,
            extra_opts=self.extra_opts.strip(),
            enabled=bool(self.enabled),
            upload_dataset=bool(self.upload_dataset),
            source=self.source,
            last_validation_ok=self.last_validation_ok,
            last_validation_error=self.last_validation_error.strip(),
        )

    def is_active(self) -> bool:
        cfg = self.normalized()
        return bool(cfg.enabled and cfg.host and cfg.remote_path)


_LOCK = Lock()
_CACHE: RemoteTrainConfig | None = None


def _config_path() -> Path:
    return Path.cwd() / ".comfyresearch" / "remote_train_config.json"


def _from_env() -> RemoteTrainConfig:
    host = os.environ.get("COMFYRESEARCH_TRAIN_REMOTE_HOST", "").strip()
    return RemoteTrainConfig(
        host=host,
        user=os.environ.get("COMFYRESEARCH_TRAIN_REMOTE_USER", "ubuntu").strip() or "ubuntu",
        remote_path=os.environ.get("COMFYRESEARCH_TRAIN_REMOTE_PATH", "").strip(),
        python=os.environ.get("COMFYRESEARCH_TRAIN_REMOTE_PYTHON", "/root/miniconda3/bin/python3").strip()
        or "/root/miniconda3/bin/python3",
        identity=os.environ.get("COMFYRESEARCH_TRAIN_REMOTE_IDENTITY", "").strip(),
        password=os.environ.get("COMFYRESEARCH_TRAIN_REMOTE_PASSWORD", ""),
        extra_opts=os.environ.get("COMFYRESEARCH_TRAIN_REMOTE_EXTRA_OPTS", "").strip(),
        enabled=bool(host),
        upload_dataset=os.environ.get("COMFYRESEARCH_TRAIN_REMOTE_UPLOAD_DATASET", "").strip().lower()
        in {"1", "true", "yes", "on"},
        source="env",
    ).normalized()


def _parse_saved(raw: object) -> RemoteTrainConfig:
    if not isinstance(raw, dict):
        return RemoteTrainConfig()
    py = str(raw.get("python", "/root/miniconda3/bin/python3")).strip()
    if py == "python3":
        py = "/root/miniconda3/bin/python3"
    return RemoteTrainConfig(
        host=str(raw.get("host", "")),
        user=str(raw.get("user", "ubuntu")),
        remote_path=str(raw.get("remote_path", "")),
        python=py,
        identity=str(raw.get("identity", "")),
        password=str(raw.get("password", "")),
        extra_opts=str(raw.get("extra_opts", "")),
        enabled=bool(raw.get("enabled", False)),
        upload_dataset=bool(raw.get("upload_dataset", False)),
        source="stored",
        last_validation_ok=(
            bool(raw["last_validation_ok"]) if isinstance(raw.get("last_validation_ok"), bool) else None
        ),
        last_validation_error=str(raw.get("last_validation_error", "")),
    ).normalized()


def load_saved_remote_train_config() -> RemoteTrainConfig:
    global _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        path = _config_path()
        if not path.exists():
            _CACHE = RemoteTrainConfig(source="none")
            return _CACHE
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _CACHE = RemoteTrainConfig(source="none")
            return _CACHE
        _CACHE = _parse_saved(payload)
        return _CACHE


def save_remote_train_config(cfg: RemoteTrainConfig) -> RemoteTrainConfig:
    global _CACHE
    normalized = cfg.normalized()
    payload = asdict(normalized)
    payload.pop("source", None)
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _CACHE = normalized
        return normalized


def get_effective_remote_train_config() -> RemoteTrainConfig:
    stored = load_saved_remote_train_config().normalized()
    if stored.is_active():
        return RemoteTrainConfig(**{**asdict(stored), "source": "stored"}).normalized()
    env_cfg = _from_env().normalized()
    if env_cfg.is_active():
        return env_cfg
    return RemoteTrainConfig(source="none")


def set_last_validation_result(ok: bool, error: str = "") -> RemoteTrainConfig:
    stored = load_saved_remote_train_config().normalized()
    if stored.source != "stored" and not stored.host and not stored.remote_path:
        return stored
    updated = RemoteTrainConfig(
        **{
            **asdict(stored),
            "last_validation_ok": bool(ok),
            "last_validation_error": "" if ok else error.strip(),
            "source": "stored",
        }
    ).normalized()
    return save_remote_train_config(updated)
