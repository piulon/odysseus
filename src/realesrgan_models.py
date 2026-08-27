"""Pinned local Real-ESRGAN checkpoint management.

Runtime inference never downloads checkpoints.  Models are provisioned only by
an explicit administrative Cookbook installation and are verified again before
they are handed to RealESRGANer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import secrets
import tempfile
from typing import Callable
import urllib.request

from src.constants import (
    FACEXLIB_MODELS_DIR,
    GFPGAN_MODELS_DIR,
    REALESRGAN_MODELS_DIR,
)


@dataclass(frozen=True)
class RealESRGANModelSpec:
    name: str
    url: str
    size: int
    sha256: str


GFPGAN_MODEL_SPEC = RealESRGANModelSpec(
    name="GFPGANv1.4.pth",
    url=(
        "https://github.com/TencentARC/GFPGAN/releases/"
        "download/v1.3.0/GFPGANv1.4.pth"
    ),
    size=348_632_874,
    sha256=(
        "e2cd4703ab14f4d01fd1383a8a8b266f"
        "9a5833dacee8e6a79d3bf21a1b6be5ad"
    ),
)



FACEXLIB_MODEL_SPECS = {
    "detection_Resnet50_Final.pth": RealESRGANModelSpec(
        name="detection_Resnet50_Final.pth",
        url=(
            "https://github.com/xinntao/facexlib/releases/"
            "download/v0.1.0/detection_Resnet50_Final.pth"
        ),
        size=109_497_761,
        sha256=(
            "6d1de9c2944f2ccddca5f5e010ea5ae6"
            "4a39845a86311af6fdf30841b0a5a16d"
        ),
    ),
    "parsing_parsenet.pth": RealESRGANModelSpec(
        name="parsing_parsenet.pth",
        url=(
            "https://github.com/xinntao/facexlib/releases/"
            "download/v0.2.2/parsing_parsenet.pth"
        ),
        size=85_331_193,
        sha256=(
            "3d558d8d0e42c20224f13cf5a29c79e"
            "ba2d59913419f945545d8cf7b72920de2"
        ),
    ),
}


REALESRGAN_MODEL_SPECS = {
    "RealESRGAN_x4plus.pth": RealESRGANModelSpec(
        name="RealESRGAN_x4plus.pth",
        url=(
            "https://github.com/xinntao/Real-ESRGAN/releases/"
            "download/v0.1.0/RealESRGAN_x4plus.pth"
        ),
        size=67_040_989,
        sha256=(
            "4fa0d38905f75ac06eb49a7951b4266"
            "70021be3018265fd191d2125df9d682f1"
        ),
    ),
    "realesr-general-x4v3.pth": RealESRGANModelSpec(
        name="realesr-general-x4v3.pth",
        url=(
            "https://github.com/xinntao/Real-ESRGAN/releases/"
            "download/v0.2.5.0/realesr-general-x4v3.pth"
        ),
        size=4_885_111,
        sha256=(
            "8dc7edb9ac80ccdc30c3a5dca6616509"
            "367f05fbc184ad95b731f05bece96292"
        ),
    ),
    "realesr-general-wdn-x4v3.pth": RealESRGANModelSpec(
        name="realesr-general-wdn-x4v3.pth",
        url=(
            "https://github.com/xinntao/Real-ESRGAN/releases/"
            "download/v0.2.5.0/realesr-general-wdn-x4v3.pth"
        ),
        size=4_885_111,
        sha256=(
            "1641f8c4464b9f097c9fdda558927371"
            "3f67cf59f3d909e0bd688f0cee269dca"
        ),
    ),
}


class RealESRGANModelError(RuntimeError):
    """A required checkpoint is missing, corrupt, or could not be provisioned."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _verify_path(
    path: Path,
    spec: RealESRGANModelSpec,
) -> None:
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise RealESRGANModelError(
            f"Required Real-ESRGAN model is missing: {spec.name}"
        ) from exc

    if not path.is_file():
        raise RealESRGANModelError(
            f"Real-ESRGAN model is not a regular file: {spec.name}"
        )

    if stat.st_size != spec.size:
        raise RealESRGANModelError(
            f"Real-ESRGAN model size mismatch: {spec.name}"
        )

    actual = _sha256_path(path)

    if not secrets.compare_digest(
        actual.lower(),
        spec.sha256.lower(),
    ):
        raise RealESRGANModelError(
            f"Real-ESRGAN model checksum mismatch: {spec.name}"
        )


def verified_realesrgan_model(
    name: str,
    *,
    model_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return a local model path only after exact size + SHA-256 verification."""

    try:
        spec = REALESRGAN_MODEL_SPECS[name]
    except KeyError as exc:
        raise RealESRGANModelError(
            f"Unknown Real-ESRGAN model: {name}"
        ) from exc

    root = Path(
        model_dir
        if model_dir is not None
        else REALESRGAN_MODELS_DIR
    )

    path = root / spec.name

    _verify_path(path, spec)

    return path


def provision_realesrgan_models(
    *,
    model_dir: str | os.PathLike[str] | None = None,
    opener: Callable[..., object] | None = None,
) -> dict[str, str]:
    """Download all pinned checkpoints atomically and verify before publication.

    This is an explicit provisioning operation for the admin-only Cookbook
    installer.  Runtime gallery requests must never call it.
    """

    root = Path(
        model_dir
        if model_dir is not None
        else REALESRGAN_MODELS_DIR
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    open_url = (
        opener
        if opener is not None
        else urllib.request.urlopen
    )

    result: dict[str, str] = {}

    for name, spec in REALESRGAN_MODEL_SPECS.items():
        target = root / name

        try:
            _verify_path(target, spec)
        except RealESRGANModelError:
            pass
        else:
            result[name] = "verified-existing"
            continue

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".tmp",
            dir=root,
        )

        tmp_path = Path(tmp_name)

        try:
            total = 0

            request = urllib.request.Request(
                spec.url,
                headers={
                    "User-Agent": (
                        "Odysseus/"
                        "RealESRGAN-checkpoint-provisioner"
                    )
                },
            )

            with os.fdopen(fd, "wb") as output:
                with open_url(
                    request,
                    timeout=120,
                ) as response:
                    while True:
                        chunk = response.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        total += len(chunk)

                        if total > spec.size:
                            raise RealESRGANModelError(
                                "Real-ESRGAN model exceeded "
                                f"expected size: {name}"
                            )

                        output.write(chunk)

                output.flush()
                os.fsync(output.fileno())

            _verify_path(
                tmp_path,
                spec,
            )

            os.replace(
                tmp_path,
                target,
            )

            _verify_path(
                target,
                spec,
            )

            result[name] = "downloaded-verified"

        except Exception as exc:
            if isinstance(
                exc,
                RealESRGANModelError,
            ):
                raise

            raise RealESRGANModelError(
                f"Failed to provision Real-ESRGAN model: {name}"
            ) from exc

        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    return result

def verified_gfpgan_model(
    *,
    model_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return GFPGANv1.4 only after exact size + SHA-256 verification."""

    root = Path(
        model_dir
        if model_dir is not None
        else GFPGAN_MODELS_DIR
    )

    path = root / GFPGAN_MODEL_SPEC.name

    _verify_path(
        path,
        GFPGAN_MODEL_SPEC,
    )

    return path


def provision_gfpgan_model(
    *,
    model_dir: str | os.PathLike[str] | None = None,
    opener: Callable[..., object] | None = None,
) -> dict[str, str]:
    """Explicitly provision the pinned GFPGAN checkpoint.

    Gallery requests never call this function.
    """

    root = Path(
        model_dir
        if model_dir is not None
        else GFPGAN_MODELS_DIR
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    spec = GFPGAN_MODEL_SPEC
    target = root / spec.name

    try:
        _verify_path(
            target,
            spec,
        )
    except RealESRGANModelError:
        pass
    else:
        return {
            spec.name: "verified-existing"
        }

    open_url = (
        opener
        if opener is not None
        else urllib.request.urlopen
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{spec.name}.",
        suffix=".tmp",
        dir=root,
    )

    tmp_path = Path(tmp_name)

    try:
        total = 0

        request = urllib.request.Request(
            spec.url,
            headers={
                "User-Agent": (
                    "Odysseus/"
                    "GFPGAN-checkpoint-provisioner"
                )
            },
        )

        with os.fdopen(fd, "wb") as output:
            with open_url(
                request,
                timeout=120,
            ) as response:
                while True:
                    chunk = response.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    total += len(chunk)

                    if total > spec.size:
                        raise RealESRGANModelError(
                            "GFPGAN model exceeded "
                            "expected size"
                        )

                    output.write(chunk)

            output.flush()
            os.fsync(output.fileno())

        _verify_path(
            tmp_path,
            spec,
        )

        os.replace(
            tmp_path,
            target,
        )

        _verify_path(
            target,
            spec,
        )

        return {
            spec.name: "downloaded-verified"
        }

    except Exception as exc:
        if isinstance(
            exc,
            RealESRGANModelError,
        ):
            raise

        raise RealESRGANModelError(
            "Failed to provision GFPGAN model"
        ) from exc

    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

def verified_facexlib_model_root(
    *,
    model_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the FaceXlib model root only after all required assets verify."""

    root = Path(
        model_dir
        if model_dir is not None
        else FACEXLIB_MODELS_DIR
    )

    for spec in FACEXLIB_MODEL_SPECS.values():
        _verify_path(
            root / spec.name,
            spec,
        )

    return root


def provision_facexlib_models(
    *,
    model_dir: str | os.PathLike[str] | None = None,
    opener: Callable[..., object] | None = None,
) -> dict[str, str]:
    """Explicitly provision the FaceXlib models required by GFPGAN.

    FaceXlib itself is patched to fail closed if an expected local model is
    absent. Gallery requests never call this provisioner.
    """

    root = Path(
        model_dir
        if model_dir is not None
        else FACEXLIB_MODELS_DIR
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    open_url = (
        opener
        if opener is not None
        else urllib.request.urlopen
    )

    result: dict[str, str] = {}

    for name, spec in FACEXLIB_MODEL_SPECS.items():
        target = root / name

        try:
            _verify_path(
                target,
                spec,
            )
        except RealESRGANModelError:
            pass
        else:
            result[name] = "verified-existing"
            continue

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".tmp",
            dir=root,
        )

        tmp_path = Path(
            tmp_name
        )

        try:
            total = 0

            request = urllib.request.Request(
                spec.url,
                headers={
                    "User-Agent": (
                        "Odysseus/"
                        "FaceXlib-checkpoint-provisioner"
                    )
                },
            )

            with os.fdopen(
                fd,
                "wb",
            ) as output:

                with open_url(
                    request,
                    timeout=120,
                ) as response:

                    while True:
                        chunk = response.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        total += len(
                            chunk
                        )

                        if total > spec.size:
                            raise RealESRGANModelError(
                                "FaceXlib model exceeded "
                                f"expected size: {name}"
                            )

                        output.write(
                            chunk
                        )

                output.flush()
                os.fsync(
                    output.fileno()
                )

            _verify_path(
                tmp_path,
                spec,
            )

            os.replace(
                tmp_path,
                target,
            )

            _verify_path(
                target,
                spec,
            )

            result[name] = (
                "downloaded-verified"
            )

        except Exception as exc:
            if isinstance(
                exc,
                RealESRGANModelError,
            ):
                raise

            raise RealESRGANModelError(
                "Failed to provision "
                f"FaceXlib model: {name}"
            ) from exc

        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    return result
