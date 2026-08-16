from src.adaptive_routing_refresh import _native_ollama_root


def test_remote_ollama_v1_on_standard_port_maps_to_native_root():
    assert _native_ollama_root(
        "http://100.87.190.46:11434/v1"
    ) == "http://100.87.190.46:11434"


def test_pathless_ollama_standard_port_is_preserved():
    assert _native_ollama_root(
        "http://host.docker.internal:11434"
    ) == "http://host.docker.internal:11434"


def test_ollama_hostname_v1_maps_to_native_root():
    assert _native_ollama_root(
        "https://ollama.example/v1"
    ) == "https://ollama.example"


def test_generic_v1_endpoint_is_not_assumed_to_be_ollama():
    assert _native_ollama_root(
        "https://api.example/v1"
    ) is None


def test_custom_path_on_ollama_port_is_not_guessed():
    assert _native_ollama_root(
        "http://10.0.0.5:11434/custom"
    ) is None
