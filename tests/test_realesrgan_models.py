import hashlib
import io
from pathlib import Path

import pytest

import src.realesrgan_models as models


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        self.close()
        return False


def _tiny_spec(payload=b"known-checkpoint"):
    return models.RealESRGANModelSpec(
        name="tiny.pth",
        url=(
            "https://github.com/example/project/"
            "releases/download/v1/tiny.pth"
        ),
        size=len(payload),
        sha256=hashlib.sha256(
            payload
        ).hexdigest(),
    )


def test_production_model_specs_match_audited_manifest():
    expected = {
        "RealESRGAN_x4plus.pth": (
            67_040_989,
            "4fa0d38905f75ac06eb49a7951b4266"
            "70021be3018265fd191d2125df9d682f1",
        ),
        "realesr-general-x4v3.pth": (
            4_885_111,
            "8dc7edb9ac80ccdc30c3a5dca6616509"
            "367f05fbc184ad95b731f05bece96292",
        ),
        "realesr-general-wdn-x4v3.pth": (
            4_885_111,
            "1641f8c4464b9f097c9fdda558927371"
            "3f67cf59f3d909e0bd688f0cee269dca",
        ),
    }

    assert set(
        models.REALESRGAN_MODEL_SPECS
    ) == set(expected)

    for name, (
        size,
        sha256,
    ) in expected.items():
        spec = (
            models.REALESRGAN_MODEL_SPECS[
                name
            ]
        )

        assert spec.size == size
        assert spec.sha256 == sha256
        assert spec.url.startswith(
            "https://github.com/xinntao/"
            "Real-ESRGAN/releases/"
        )


def test_verified_model_accepts_exact_file(
    tmp_path,
    monkeypatch,
):
    payload = b"known-checkpoint"
    spec = _tiny_spec(payload)

    monkeypatch.setattr(
        models,
        "REALESRGAN_MODEL_SPECS",
        {spec.name: spec},
    )

    path = tmp_path / spec.name
    path.write_bytes(payload)

    assert (
        models.verified_realesrgan_model(
            spec.name,
            model_dir=tmp_path,
        )
        == path
    )


def test_verified_model_rejects_tampering(
    tmp_path,
    monkeypatch,
):
    payload = b"known-checkpoint"
    spec = _tiny_spec(payload)

    monkeypatch.setattr(
        models,
        "REALESRGAN_MODEL_SPECS",
        {spec.name: spec},
    )

    path = tmp_path / spec.name
    path.write_bytes(
        b"tampered-checkpt"
    )

    assert len(
        path.read_bytes()
    ) == spec.size

    with pytest.raises(
        models.RealESRGANModelError,
        match="checksum mismatch",
    ):
        models.verified_realesrgan_model(
            spec.name,
            model_dir=tmp_path,
        )


def test_verified_model_rejects_missing(
    tmp_path,
    monkeypatch,
):
    spec = _tiny_spec()

    monkeypatch.setattr(
        models,
        "REALESRGAN_MODEL_SPECS",
        {spec.name: spec},
    )

    with pytest.raises(
        models.RealESRGANModelError,
        match="missing",
    ):
        models.verified_realesrgan_model(
            spec.name,
            model_dir=tmp_path,
        )


def test_provision_downloads_verifies_and_publishes_atomically(
    tmp_path,
    monkeypatch,
):
    payload = b"known-checkpoint"
    spec = _tiny_spec(payload)

    monkeypatch.setattr(
        models,
        "REALESRGAN_MODEL_SPECS",
        {spec.name: spec},
    )

    seen = []

    def opener(
        request,
        timeout,
    ):
        seen.append(
            (
                request.full_url,
                timeout,
            )
        )

        return _FakeResponse(
            payload
        )

    result = models.provision_realesrgan_models(
        model_dir=tmp_path,
        opener=opener,
    )

    assert result == {
        spec.name: "downloaded-verified"
    }

    assert (
        tmp_path / spec.name
    ).read_bytes() == payload

    assert seen == [
        (
            spec.url,
            120,
        )
    ]

    assert not list(
        tmp_path.glob("*.tmp")
    )

    assert not list(
        tmp_path.glob(".*.tmp")
    )


def test_provision_rejects_wrong_download(
    tmp_path,
    monkeypatch,
):
    payload = b"known-checkpoint"
    spec = _tiny_spec(payload)

    monkeypatch.setattr(
        models,
        "REALESRGAN_MODEL_SPECS",
        {spec.name: spec},
    )

    def opener(
        request,
        timeout,
    ):
        return _FakeResponse(
            b"wrong-checkpoint"
        )

    with pytest.raises(
        models.RealESRGANModelError,
    ):
        models.provision_realesrgan_models(
            model_dir=tmp_path,
            opener=opener,
        )

    assert not (
        tmp_path / spec.name
    ).exists()
