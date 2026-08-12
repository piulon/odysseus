from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PICKER = (ROOT / "static/js/modelPicker.js").read_text(encoding="utf-8")
SESSIONS = (ROOT / "static/js/sessions.js").read_text(encoding="utf-8")


def test_auto_is_state_not_fake_model():
    assert "autoRoute: true" in PICKER
    assert "fd.append('auto_route', 'true')" in PICKER

    assert "modelId: 'Auto'" not in PICKER
    assert 'modelId: "Auto"' not in PICKER
    assert "fd.append('model', 'Auto')" not in PICKER
    assert 'fd.append("model", "Auto")' not in PICKER


def test_existing_session_auto_patch_preserves_real_route():
    start = PICKER.index("async function _pickAuto()")
    end = PICKER.index("async function _pick(m)", start)
    body = PICKER[start:end]

    assert "fd.append('auto_route', 'true')" in body
    assert "fd.append('model'" not in body
    assert "fd.append('endpoint_url'" not in body

    assert "session.auto_route = true" in body
    assert "session.model =" not in body
    assert "session.endpoint_url =" not in body


def test_manual_model_pick_disables_auto_locally():
    assert "s.auto_route = false;" in PICKER
    assert "autoRoute: false" in PICKER


def test_pending_auto_preserves_real_model_and_materializes_flag():
    assert "...pending," in PICKER
    assert "autoRoute: true" in PICKER

    assert "_pendingChat = { url, modelId, endpointId, autoRoute: false };" in SESSIONS
    assert "if (pending.autoRoute)" in SESSIONS
    assert "fd.append('auto_route', 'true')" in SESSIONS

    # The real persistent route is still materialized normally.
    assert "fd.append('endpoint_url', pending.url || '')" in SESSIONS
    assert "fd.append('model', pending.modelId || '')" in SESSIONS


def test_picker_displays_auto_without_overwriting_model_id():
    start = PICKER.index("export function updateModelPicker()")
    body = PICKER[start:]

    assert "(s && s.auto_route)" in body
    assert "_pendingChat.autoRoute" in body
    assert "? 'Auto'" in body
    assert "const logo = autoRoute ? null" in body


def test_pending_auto_is_not_replaced_by_default_resolution():
    assert "(pending.source === 'manual' || pending.autoRoute)" in PICKER


def test_auto_row_requires_real_base_route():
    assert "const _autoBaseReady" in PICKER
    assert "_autoSession.model && _autoSession.endpoint_url" in PICKER
    assert "_autoPending.modelId && _autoPending.url" in PICKER
