"""GraphRAG + LightRAG engine knobs stored in RuntimeSettingsService."""

from __future__ import annotations

from pathlib import Path

from deeptutor.services.config.runtime_settings import RuntimeSettingsService


def test_graphrag_defaults_and_clamp(tmp_path: Path) -> None:
    svc = RuntimeSettingsService(tmp_path, process_env={})
    defaults = svc.load_graphrag()
    assert defaults["response_type"] == "Multiple Paragraphs"
    assert defaults["community_level"] == 2
    assert defaults["dynamic_community_selection"] is False

    saved = svc.save_graphrag(
        {
            "community_level": 99,
            "dynamic_community_selection": "yes",
            "response_type": "  Single Paragraph  ",
        }
    )
    assert saved["community_level"] == 5  # clamped to max
    assert saved["dynamic_community_selection"] is True
    assert saved["response_type"] == "Single Paragraph"
    assert (tmp_path / "graphrag.json").exists()


def test_lightrag_defaults_and_clamp(tmp_path: Path) -> None:
    svc = RuntimeSettingsService(tmp_path, process_env={})
    defaults = svc.load_lightrag()
    assert defaults["top_k"] == 60
    assert defaults["response_type"] == "Multiple Paragraphs"
    assert defaults["vector_storage"] == "nano"
    assert defaults["llm_concurrency"] == 8
    assert defaults["embedding_concurrency"] == 2
    assert defaults["multimodal_concurrency"] == 8
    assert defaults["entity_extract_max_gleaning"] == 0
    assert defaults["chunk_token_size"] == 1400
    assert defaults["chunk_overlap_token_size"] == 80
    assert defaults["embedding_batch_num"] == 20
    assert defaults["force_llm_summary_on_merge"] == 16

    saved = svc.save_lightrag({"top_k": 9999})
    assert saved["top_k"] == 200  # clamped to max
    assert (tmp_path / "lightrag.json").exists()

    floored = svc.save_lightrag({"top_k": 0})
    assert floored["top_k"] == 1  # clamped to min

    throughput = svc.save_lightrag(
        {
            "llm_concurrency": 100,
            "embedding_concurrency": 0,
            "multimodal_concurrency": 100,
            "entity_extract_max_gleaning": 9,
            "chunk_token_size": 128,
            "chunk_overlap_token_size": 9999,
            "embedding_batch_num": 999,
            "force_llm_summary_on_merge": 0,
        }
    )
    assert throughput["llm_concurrency"] == 32
    assert throughput["embedding_concurrency"] == 1
    assert throughput["multimodal_concurrency"] == 16
    assert throughput["entity_extract_max_gleaning"] == 3
    assert throughput["chunk_token_size"] == 256
    assert throughput["chunk_overlap_token_size"] == 255
    assert throughput["embedding_batch_num"] == 256
    assert throughput["force_llm_summary_on_merge"] == 3


def test_lightrag_vector_storage_normalization(tmp_path: Path) -> None:
    svc = RuntimeSettingsService(tmp_path, process_env={})
    assert svc.save_lightrag({"vector_storage": "faiss"})["vector_storage"] == "faiss"
    assert svc.save_lightrag({"vector_storage": " Faiss "})["vector_storage"] == "faiss"
    # Unknown / empty values never break indexing — they fall back to nano.
    assert svc.save_lightrag({"vector_storage": "qdrant"})["vector_storage"] == "nano"
    assert svc.save_lightrag({"vector_storage": ""})["vector_storage"] == "nano"


def test_response_type_capped(tmp_path: Path) -> None:
    svc = RuntimeSettingsService(tmp_path, process_env={})
    saved = svc.save_graphrag({"response_type": "x" * 500})
    assert len(saved["response_type"]) == 80


def test_preflight_shape_for_all_engines() -> None:
    from deeptutor.services.rag.preflight import engine_preflight

    for provider in ("llamaindex", "pageindex", "graphrag", "lightrag"):
        report = engine_preflight(provider)
        assert set(report) == {"ok", "checks"}
        assert isinstance(report["ok"], bool)
        assert report["checks"], f"{provider} should report at least one check"
        for check in report["checks"]:
            assert set(check) == {"key", "label", "ok", "detail", "optional"}
        # Overall ok ignores optional checks.
        required_ok = all(c["ok"] for c in report["checks"] if not c["optional"])
        assert report["ok"] == required_ok


def test_graphrag_static_preflight_does_not_guess_structured_output_support(
    monkeypatch,
) -> None:
    from deeptutor.services.rag import preflight
    from deeptutor.services.rag.pipelines.graphrag import config as graphrag_config

    monkeypatch.setattr(graphrag_config, "is_graphrag_available", lambda: True)
    monkeypatch.setattr(preflight, "_active_chat_model", lambda: ("deepseek-v4-flash", "deepseek"))
    monkeypatch.setattr(preflight, "_active_embedding", lambda: ("text-embedding", 1024))

    report = preflight.engine_preflight("graphrag")
    assert all(check["key"] != "structured_output" for check in report["checks"])
    assert report["ok"] is True


def test_preflight_unknown_provider_falls_back_to_default() -> None:
    from deeptutor.services.rag.preflight import engine_preflight

    # Unknown providers normalize to the default (llamaindex) engine.
    report = engine_preflight("does-not-exist")
    assert any(c["key"] == "embedding" for c in report["checks"])
