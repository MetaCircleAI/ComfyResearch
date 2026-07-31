"""CIFAR-10 下载物的摘要与安全解包守卫。"""
from __future__ import annotations

import io
import os
import tarfile

import pytest
from fastapi import HTTPException

from comfy_research.engine.datasets.vision_datasets_runtime import (
    _cifar10_expected_sha256,
    _extract_cifar10_tar,
    _safe_tar_members,
    _sha256_of_file,
)


def _tar_with(name: str, tmp_path) -> str:
    p = tmp_path / "evil.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        data = b"x"
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    return str(p)


def test_traversal_member_rejected(tmp_path) -> None:
    tar = _tar_with("cifar-10-batches-py/../../evil.txt", tmp_path)
    with pytest.raises(HTTPException, match="escapes cache dir"):
        _extract_cifar10_tar(tar, str(tmp_path / "cache"))
    assert not (tmp_path / "evil.txt").exists()


def test_unexpected_top_level_rejected(tmp_path) -> None:
    tar = _tar_with("not-cifar/evil.txt", tmp_path)
    with pytest.raises(HTTPException, match="unexpected top-level"):
        _extract_cifar10_tar(tar, str(tmp_path / "cache"))


def test_symlink_member_rejected(tmp_path) -> None:
    p = tmp_path / "link.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        ti = tarfile.TarInfo("cifar-10-batches-py/link")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)
    with pytest.raises(HTTPException, match="non-file member"):
        with tarfile.open(p, "r:gz") as tf:
            _safe_tar_members(tf, str(tmp_path / "cache"))


def test_digest_env_override_and_mismatch(tmp_path, monkeypatch) -> None:
    f = tmp_path / "blob"
    f.write_bytes(b"hello")
    got = _sha256_of_file(str(f))
    assert got == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    monkeypatch.setenv("COMFYRESEARCH_CIFAR10_SHA256", "deadbeef")
    assert _cifar10_expected_sha256() == "deadbeef"
    monkeypatch.delenv("COMFYRESEARCH_CIFAR10_SHA256")
    assert len(_cifar10_expected_sha256()) == 64


def test_cached_tar_digest_mismatch_never_reaches_extract(tmp_path, monkeypatch) -> None:
    """缓存 tar 摘要不符 → 清缓存重下(mock 拒绝),绝不进 extract。"""
    from comfy_research.engine.datasets import vision_datasets_runtime as v

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cifar-10-python.tar.gz").write_bytes(b"malicious-cached-bytes")
    called = {"extract": 0, "download": 0}
    monkeypatch.setattr(v, "_extract_cifar10_tar", lambda *a, **k: called.__setitem__("extract", called["extract"] + 1))

    def _fake_download(tar_path):
        called["download"] += 1
        raise HTTPException(status_code=502, detail="download refused by test")

    monkeypatch.setattr(v, "_download_cifar10_tar", _fake_download)
    monkeypatch.setattr(v, "_bundled_cifar10_batches_root", lambda: None)
    with pytest.raises(HTTPException, match="download refused"):
        v._ensure_cifar10_extracted(str(cache))
    assert called["extract"] == 0 and called["download"] == 1
    assert not (cache / "cifar-10-python.tar.gz").exists()  # 恶意缓存已被 wipe


def test_truncated_extracted_batch_is_wiped_before_redownload(tmp_path, monkeypatch) -> None:
    from comfy_research.engine.datasets import vision_datasets_runtime as v

    cache = tmp_path / "cache"
    root = cache / "cifar-10-batches-py"
    root.mkdir(parents=True)
    for name in [*(f"data_batch_{i}" for i in range(1, 6)), "test_batch"]:
        with (root / name).open("wb") as f:
            f.truncate(31_000_000)
    (root / "data_batch_5").write_bytes(b"truncated")

    called = {"download": 0}

    def _fake_download(_tar_path: str) -> None:
        called["download"] += 1
        raise HTTPException(status_code=502, detail="redownload attempted")

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("COMFYRESEARCH_CIFAR10_BUNDLED", str(cache))
    monkeypatch.setattr(v, "_repo_root", lambda: tmp_path / "missing-repo")
    monkeypatch.setattr(v, "_download_cifar10_tar", _fake_download)

    with pytest.raises(HTTPException, match="redownload attempted"):
        v._ensure_cifar10_extracted(str(cache))

    assert called["download"] == 1
    assert not root.exists()


def test_sjtu_is_the_default_first_cifar10_source(monkeypatch) -> None:
    from comfy_research.engine.datasets import vision_datasets_runtime as v

    monkeypatch.delenv("COMFYRESEARCH_CIFAR10_URL", raising=False)
    monkeypatch.delenv("COMFYRESEARCH_CIFAR10_URLS", raising=False)

    assert v._cifar10_download_urls()[0] == (
        "https://scidata.sjtu.edu.cn/records/p4t8m-rbe26/files/"
        "cifar-10-python.tar.gz?download=1"
    )


def test_sjtu_gets_a_longer_download_timeout() -> None:
    from comfy_research.engine.datasets import vision_datasets_runtime as v

    sjtu = "https://scidata.sjtu.edu.cn/records/example/cifar-10-python.tar.gz"
    github = "https://github.com/example/cifar-10-python.tar.gz"

    assert v._cifar10_download_timeout_s(sjtu) == 300
    assert v._cifar10_download_timeout_s(github) == 120
