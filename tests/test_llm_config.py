from multiagent.llm import get_llm_settings, llm_metadata


def test_deepseek_defaults_remain_backward_compatible(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-test")

    settings = get_llm_settings()

    assert settings.provider == "deepseek"
    assert settings.api_key == "test-key"
    assert settings.model == "deepseek-test"


def test_generic_provider_uses_llm_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "generic")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "generic-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")

    settings = get_llm_settings()

    assert settings.provider == "generic"
    assert settings.base_url == "https://example.invalid/v1"
    assert settings.model == "generic-model"
    assert settings.temperature == 0.1


def test_llm_metadata_never_contains_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "generic")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_MODEL", "model-x")

    metadata = llm_metadata()

    assert "api_key" not in metadata
    assert metadata["provider"] == "generic"
    assert metadata["model"] == "model-x"
