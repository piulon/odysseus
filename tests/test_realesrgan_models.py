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


def test_gfpgan_production_spec_matches_audited_checkpoint():
    spec = models.GFPGAN_MODEL_SPEC

    assert spec.name == "GFPGANv1.4.pth"

    assert spec.size == 348_632_874

    assert spec.sha256 == (
        "e2cd4703ab14f4d01fd1383a8a8b266f"
        "9a5833dacee8e6a79d3bf21a1b6be5ad"
    )

    assert spec.url == (
        "https://github.com/TencentARC/GFPGAN/"
        "releases/download/v1.3.0/GFPGANv1.4.pth"
    )


def test_verified_gfpgan_model_accepts_exact_file(
    tmp_path,
    monkeypatch,
):
    payload = b"known-gfpgan-checkpoint"

    spec = models.RealESRGANModelSpec(
        name="GFPGANv1.4.pth",
        url=(
            "https://github.com/example/project/"
            "releases/download/v1/GFPGANv1.4.pth"
        ),
        size=len(payload),
        sha256=hashlib.sha256(
            payload
        ).hexdigest(),
    )

    monkeypatch.setattr(
        models,
        "GFPGAN_MODEL_SPEC",
        spec,
    )

    path = tmp_path / spec.name

    path.write_bytes(payload)

    assert (
        models.verified_gfpgan_model(
            model_dir=tmp_path,
        )
        == path
    )


def test_verified_gfpgan_model_rejects_tampering(
    tmp_path,
    monkeypatch,
):
    payload = b"known-gfpgan-checkpoint"

    spec = models.RealESRGANModelSpec(
        name="GFPGANv1.4.pth",
        url="https://example.invalid/GFPGANv1.4.pth",
        size=len(payload),
        sha256=hashlib.sha256(
            payload
        ).hexdigest(),
    )

    monkeypatch.setattr(
        models,
        "GFPGAN_MODEL_SPEC",
        spec,
    )

    path = tmp_path / spec.name

    tampered = bytearray(payload)
    tampered[0] ^= 1

    path.write_bytes(tampered)

    with pytest.raises(
        models.RealESRGANModelError,
        match="checksum mismatch",
    ):
        models.verified_gfpgan_model(
            model_dir=tmp_path,
        )


def test_provision_gfpgan_downloads_verifies_and_reuses(
    tmp_path,
    monkeypatch,
):
    payload = b"known-gfpgan-checkpoint"

    spec = models.RealESRGANModelSpec(
        name="GFPGANv1.4.pth",
        url=(
            "https://github.com/example/project/"
            "releases/download/v1/GFPGANv1.4.pth"
        ),
        size=len(payload),
        sha256=hashlib.sha256(
            payload
        ).hexdigest(),
    )

    monkeypatch.setattr(
        models,
        "GFPGAN_MODEL_SPEC",
        spec,
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

    first = models.provision_gfpgan_model(
        model_dir=tmp_path,
        opener=opener,
    )

    assert first == {
        spec.name: "downloaded-verified"
    }

    assert seen == [
        (
            spec.url,
            120,
        )
    ]

    def forbidden(
        request,
        timeout,
    ):
        raise AssertionError(
            "verified checkpoint was downloaded again"
        )

    second = models.provision_gfpgan_model(
        model_dir=tmp_path,
        opener=forbidden,
    )

    assert second == {
        spec.name: "verified-existing"
    }


def test_provision_gfpgan_rejects_wrong_download(
    tmp_path,
    monkeypatch,
):
    payload = b"known-gfpgan-checkpoint"

    spec = models.RealESRGANModelSpec(
        name="GFPGANv1.4.pth",
        url="https://example.invalid/GFPGANv1.4.pth",
        size=len(payload),
        sha256=hashlib.sha256(
            payload
        ).hexdigest(),
    )

    monkeypatch.setattr(
        models,
        "GFPGAN_MODEL_SPEC",
        spec,
    )

    def opener(
        request,
        timeout,
    ):
        return _FakeResponse(
            b"wrong-gfpgan-checkpoint"
        )

    with pytest.raises(
        models.RealESRGANModelError,
    ):
        models.provision_gfpgan_model(
            model_dir=tmp_path,
            opener=opener,
        )

    assert not (
        tmp_path / spec.name
    ).exists()

def test_facexlib_production_specs_match_audited_checkpoints():
    from src.realesrgan_models import (
        FACEXLIB_MODEL_SPECS,
    )

    assert set(FACEXLIB_MODEL_SPECS) == {
        "detection_Resnet50_Final.pth",
        "parsing_parsenet.pth",
    }

    detection = FACEXLIB_MODEL_SPECS[
        "detection_Resnet50_Final.pth"
    ]

    assert detection.size == 109_497_761
    assert detection.sha256 == (
        "6d1de9c2944f2ccddca5f5e010ea5ae6"
        "4a39845a86311af6fdf30841b0a5a16d"
    )

    parsing = FACEXLIB_MODEL_SPECS[
        "parsing_parsenet.pth"
    ]

    assert parsing.size == 85_331_193
    assert parsing.sha256 == (
        "3d558d8d0e42c20224f13cf5a29c79e"
        "ba2d59913419f945545d8cf7b72920de2"
    )


def test_verified_facexlib_model_root_accepts_exact_files_and_rejects_tampering(
    tmp_path,
    monkeypatch,
):
    import hashlib
    import pytest
    import src.realesrgan_models as models

    payloads = {
        "det.pth": b"verified-detector",
        "parse.pth": b"verified-parser",
    }

    specs = {
        name: models.RealESRGANModelSpec(
            name=name,
            url=f"https://example.invalid/{name}",
            size=len(payload),
            sha256=hashlib.sha256(
                payload
            ).hexdigest(),
        )
        for name, payload in payloads.items()
    }

    monkeypatch.setattr(
        models,
        "FACEXLIB_MODEL_SPECS",
        specs,
    )

    for name, payload in payloads.items():
        (
            tmp_path
            / name
        ).write_bytes(
            payload
        )

    assert (
        models.verified_facexlib_model_root(
            model_dir=tmp_path
        )
        == tmp_path
    )

    (
        tmp_path
        / "parse.pth"
    ).write_bytes(
        b"tampered"
    )

    with pytest.raises(
        models.RealESRGANModelError
    ):
        models.verified_facexlib_model_root(
            model_dir=tmp_path
        )


def test_provision_facexlib_models_downloads_verifies_reuses_and_repairs(
    tmp_path,
    monkeypatch,
):
    import hashlib
    import io
    import src.realesrgan_models as models

    payloads = {
        "det.pth": b"detector-payload",
        "parse.pth": b"parser-payload",
    }

    specs = {
        name: models.RealESRGANModelSpec(
            name=name,
            url=f"https://example.invalid/{name}",
            size=len(payload),
            sha256=hashlib.sha256(
                payload
            ).hexdigest(),
        )
        for name, payload in payloads.items()
    }

    monkeypatch.setattr(
        models,
        "FACEXLIB_MODEL_SPECS",
        specs,
    )

    calls = []

    class Response:
        def __init__(
            self,
            payload,
        ):
            self._stream = io.BytesIO(
                payload
            )

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            tb,
        ):
            return False

        def read(
            self,
            size=-1,
        ):
            return self._stream.read(
                size
            )

    def opener(
        request,
        timeout,
    ):
        name = request.full_url.rsplit(
            "/",
            1,
        )[-1]

        calls.append(
            name
        )

        return Response(
            payloads[name]
        )

    first = models.provision_facexlib_models(
        model_dir=tmp_path,
        opener=opener,
    )

    assert first == {
        "det.pth": "downloaded-verified",
        "parse.pth": "downloaded-verified",
    }

    assert calls == [
        "det.pth",
        "parse.pth",
    ]

    calls.clear()

    second = models.provision_facexlib_models(
        model_dir=tmp_path,
        opener=opener,
    )

    assert second == {
        "det.pth": "verified-existing",
        "parse.pth": "verified-existing",
    }

    assert calls == []

    (
        tmp_path
        / "parse.pth"
    ).write_bytes(
        b"tampered"
    )

    repaired = models.provision_facexlib_models(
        model_dir=tmp_path,
        opener=opener,
    )

    assert repaired == {
        "det.pth": "verified-existing",
        "parse.pth": "downloaded-verified",
    }

    assert calls == [
        "parse.pth",
    ]

    assert (
        tmp_path
        / "parse.pth"
    ).read_bytes() == payloads[
        "parse.pth"
    ]
