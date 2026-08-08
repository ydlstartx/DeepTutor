from deeptutor.services.provider_registry import (
    effort_options_for_model,
    find_by_name,
    find_gateway,
)


def test_nvidia_nim_gateway_detection_by_key_and_base() -> None:
    spec = find_by_name("nvidia_nim")

    assert spec is not None
    assert spec.supports_stream_options is False
    assert find_gateway(api_key="nvapi-test-key") == spec
    assert find_gateway(api_base="https://integrate.api.nvidia.com/v1") == spec


def test_atlascloud_provider_aliases_and_base_detection() -> None:
    spec = find_by_name("atlascloud")

    assert spec is not None
    assert spec.display_name == "Atlas Cloud"
    assert spec.env_key == "ATLASCLOUD_API_KEY"
    assert spec.backend == "openai_compat"
    assert spec.mode == "gateway"
    assert spec.default_api_base == "https://api.atlascloud.ai/v1"
    assert find_by_name("atlas-cloud") == spec
    assert find_by_name("atlas_cloud") == spec
    assert find_by_name("atlas") == spec
    assert find_gateway(api_base="https://api.atlascloud.ai/v1") == spec


def test_edenai_provider_aliases_and_base_detection() -> None:
    spec = find_by_name("edenai")

    assert spec is not None
    assert spec.display_name == "Eden AI"
    assert spec.env_key == "EDENAI_API_KEY"
    assert spec.backend == "openai_compat"
    assert spec.mode == "gateway"
    assert spec.default_api_base == "https://api.edenai.run/v3"
    assert find_by_name("eden-ai") == spec
    assert find_by_name("eden_ai") == spec
    assert find_gateway(api_base="https://api.edenai.run/v3") == spec


def test_novita_provider_aliases_and_base_detection() -> None:
    spec = find_by_name("novita")

    assert spec is not None
    assert spec.display_name == "Novita AI"
    assert spec.env_key == "NOVITA_API_KEY"
    assert spec.backend == "openai_compat"
    assert spec.mode == "gateway"
    assert spec.default_api_base == "https://api.novita.ai/openai"
    assert find_by_name("novita-ai") == spec
    assert find_by_name("novita_ai") == spec
    assert find_gateway(api_base="https://api.novita.ai/openai") == spec


def test_openai_codex_is_not_detected_from_api_base() -> None:
    assert find_gateway(api_base="https://codex.example.com/v1") is None


def test_openai_codex_provider_is_oauth_backed() -> None:
    spec = find_by_name("openai_codex")

    assert spec is not None
    assert spec.auth_mode == "oauth"
    assert spec.env_key == ""


def test_github_copilot_is_oauth_backed() -> None:
    spec = find_by_name("github_copilot")

    assert spec is not None
    assert spec.auth_mode == "oauth"
    assert spec.env_key == ""


def test_kimi_coding_plan_aliases_and_base_detection() -> None:
    spec = find_by_name("kimi_coding_plan")

    assert spec is not None
    assert spec.display_name == "Kimi Coding Plan"
    assert spec.backend == "openai_compat"
    assert spec.mode == "gateway"
    assert spec.default_api_base == "https://api.kimi.com/coding/v1"
    assert find_by_name("kimi-coding-plan") == spec
    assert find_by_name("KimiCodingPlan") == spec
    assert find_gateway(api_base="https://api.kimi.com/coding/v1") == spec


def test_effort_options_kimi_coding_plan_k3_variants() -> None:
    spec = find_by_name("kimi_coding_plan")

    assert spec is not None
    for model in ("k3", "k3-256k", "kimi-k3"):
        assert effort_options_for_model(spec, model) == (("low", "high", "max"), "high")


def test_effort_options_kimi_for_coding_has_no_selector() -> None:
    spec = find_by_name("kimi_coding_plan")

    assert spec is not None
    assert effort_options_for_model(spec, "kimi-for-coding") is None
    assert effort_options_for_model(spec, "kimi-for-coding-highspeed") is None


def test_effort_options_moonshot_public_k3_defaults_to_max() -> None:
    spec = find_by_name("moonshot")

    assert spec is not None
    assert effort_options_for_model(spec, "kimi-k3") == (("low", "high", "max"), "max")


def test_effort_options_deepseek_v4_levels() -> None:
    spec = find_by_name("deepseek")

    assert spec is not None
    for model in ("deepseek-v4-pro", "deepseek-v4-flash"):
        assert effort_options_for_model(spec, model) == (("low", "high", "max"), "high")


def test_effort_options_dashscope_is_on_off_toggle() -> None:
    spec = find_by_name("dashscope")

    assert spec is not None
    assert effort_options_for_model(spec, "qwen3.7-plus") == (("on", "off"), "on")
    assert effort_options_for_model(spec, "qwen3.7-flash") == (("on", "off"), "on")


def test_effort_options_fall_back_to_toggle_for_thinking_style_providers() -> None:
    spec = find_by_name("volcengine")

    assert spec is not None
    assert effort_options_for_model(spec, "doubao-seed-1.6") == (("on", "off"), "on")


def test_effort_options_none_for_providers_without_thinking_support() -> None:
    spec = find_by_name("openai")

    assert spec is not None
    assert effort_options_for_model(spec, "gpt-4o") is None
    assert effort_options_for_model(None, "k3") is None
