"""Unit tests for the LightRAG RAG pipeline + provider routing.

RAG-Anything / LightRAG is an optional dependency that is NOT installed in CI,
so these tests exercise everything that does not require the package (factory
routing, config bridge, storage, lifecycle gating, parse-layer consumption)
directly, and stub the thin ``engine`` adapter + the parse service to cover the
index/search orchestration without the heavy deps.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
from pathlib import Path
import sys
import threading
import time
import types

import pytest

from deeptutor.services.rag.factory import (
    LIGHTRAG_PROVIDER,
    get_pipeline,
    list_pipelines,
    normalize_provider_name,
)
from deeptutor.services.rag.index_versioning import resolve_storage_dir_for_read
from deeptutor.services.rag.pipelines.lightrag import (
    block_policy,
    engine,
    storage,
)
from deeptutor.services.rag.pipelines.lightrag import (
    config as lr_config,
)
from deeptutor.services.rag.pipelines.lightrag.pipeline import LightRagPipeline
from deeptutor.services.rag.pipelines.lightrag.worker import (
    run_in_shared_worker_loop,
    run_in_worker_loop,
)

# --------------------------------------------------------------------------- #
# factory routing + config
# --------------------------------------------------------------------------- #


def test_factory_dispatches_lightrag_lazily(tmp_path) -> None:
    pipe = get_pipeline("lightrag", kb_base_dir=str(tmp_path))
    assert type(pipe).__name__ == "LightRagPipeline"
    # Building the pipeline must NOT import the heavy optional dependency.
    assert "raganything" not in sys.modules


def test_list_pipelines_includes_lightrag(monkeypatch) -> None:
    monkeypatch.setattr(lr_config, "is_lightrag_available", lambda: False)
    entry = next(p for p in list_pipelines() if p["id"] == LIGHTRAG_PROVIDER)
    assert entry["requires_api_key"] is False
    assert entry["configured"] is False


def test_indexing_kwargs_bridge_runtime_throughput_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        "deeptutor.services.config.load_lightrag_settings",
        lambda: {
            "llm_concurrency": 12,
            "embedding_concurrency": 3,
            "multimodal_concurrency": 7,
            "entity_extract_max_gleaning": 1,
            "chunk_token_size": 1600,
            "chunk_overlap_token_size": 64,
            "embedding_batch_num": 24,
            "force_llm_summary_on_merge": 20,
        },
    )

    assert lr_config.lightrag_kwargs_from_settings() == {
        "llm_model_max_async": 12,
        "embedding_func_max_async": 3,
        "max_parallel_insert": 7,
        "entity_extract_max_gleaning": 1,
        "chunk_token_size": 1600,
        "chunk_overlap_token_size": 64,
        "embedding_batch_num": 24,
        "force_llm_summary_on_merge": 20,
    }


def test_normalize_provider_keeps_lightrag() -> None:
    assert normalize_provider_name("lightrag") == "lightrag"
    assert normalize_provider_name("LightRAG") == "lightrag"


@pytest.mark.parametrize(
    "given,expected",
    [
        ("hybrid", "hybrid"),
        ("MIX", "mix"),
        ("naive", "naive"),
        ("local", "local"),
        ("global", "global"),
        ("", "hybrid"),
        (None, "hybrid"),
        ("bogus", "hybrid"),
    ],
)
def test_normalize_mode(given, expected) -> None:
    assert lr_config.normalize_mode(given) == expected


@pytest.mark.parametrize(
    ("query_only", "required_package"),
    [(False, "raganything"), (True, "lightrag")],
)
def test_is_lightrag_available_checks_deployment_dependency(
    monkeypatch, query_only, required_package
) -> None:
    looked_up: list[str] = []

    def fake_find_spec(name):
        looked_up.append(name)
        return None

    monkeypatch.setattr(lr_config.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr("deeptutor.knowledge.policy.is_kb_query_only", lambda: query_only)

    assert lr_config.is_lightrag_available() is False
    assert looked_up == [required_package]


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #


def test_storage_meta_and_has_output(tmp_path) -> None:
    root = tmp_path / "version-1"
    root.mkdir()
    assert storage.has_output(root) is False
    assert storage.has_output(None) is False

    (root / "vdb_chunks.json").write_text("{}", encoding="utf-8")
    assert storage.has_output(root) is False

    (root / "graph_chunk_entity_relation.graphml").write_text("<graph/>", encoding="utf-8")
    assert storage.has_output(root) is False

    (root / "kv_store_doc_status.json").write_text(
        json.dumps(
            {
                "doc-1": {
                    "status": "failed",
                    "file_path": "bad.docx",
                    "error_msg": "embedding failed",
                    "chunks_list": [],
                }
            }
        ),
        encoding="utf-8",
    )
    assert storage.has_output(root) is False
    assert storage.failure_summary(root) == "bad.docx: embedding failed"
    assert storage.document_error(root, "doc-1") == "embedding failed"

    (root / "kv_store_doc_status.json").write_text(
        json.dumps(
            {
                "doc-1": {
                    "status": "processed",
                    "file_path": "good.docx",
                    "chunks_list": ["chunk-1"],
                }
            }
        ),
        encoding="utf-8",
    )
    assert storage.has_output(root) is True

    storage.write_meta(root)
    meta = json.loads((root / storage.META_FILENAME).read_text())
    assert meta["signature"] == "lightrag"
    assert meta["provider"] == "lightrag"


def test_storage_reports_truncated_or_missing_graph_for_ready_index(tmp_path) -> None:
    root = tmp_path / "version-1"
    root.mkdir()
    (root / "kv_store_doc_status.json").write_text(
        json.dumps({"doc-1": {"status": "processed", "chunks_list": ["chunk-1"]}}),
        encoding="utf-8",
    )

    assert "missing GraphML" in storage.graph_integrity_error(root)

    graph = root / storage.GRAPH_FILENAME
    graph.write_text("<graphml><graph>", encoding="utf-8")
    assert "invalid GraphML" in storage.graph_integrity_error(root)

    graph.write_text("<graphml><graph/></graphml>", encoding="utf-8")
    assert storage.graph_integrity_error(root) is None

    graph.write_text("<graphml/>", encoding="utf-8")
    assert "no <graph>" in storage.graph_integrity_error(root)


class _FakeEmbeddingFunc:
    """Stands in for ``lightrag.utils.EmbeddingFunc``.

    Its signature is deliberately limited to the real dataclass's fields, so a
    constructor kwarg the pinned dependency does not accept fails here too.
    ``test_fake_embedding_func_matches_the_real_dataclass`` pins the two
    together whenever LightRAG is installed.
    """

    def __init__(
        self,
        *,
        embedding_dim,
        func,
        max_token_size=8192,
        send_dimensions=None,
        model_name=None,
        supports_asymmetric=False,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.func = func
        self.max_token_size = max_token_size
        self.send_dimensions = send_dimensions
        self.model_name = model_name
        self.supports_asymmetric = supports_asymmetric


class _RecordingBridge:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, factory):
        self.calls += 1
        return await factory()


def _install_fake_lightrag(monkeypatch) -> None:
    fake_lightrag = types.ModuleType("lightrag")
    fake_utils = types.ModuleType("lightrag.utils")
    fake_utils.EmbeddingFunc = _FakeEmbeddingFunc
    monkeypatch.setitem(sys.modules, "lightrag", fake_lightrag)
    monkeypatch.setitem(sys.modules, "lightrag.utils", fake_utils)


def test_fake_embedding_func_matches_the_real_dataclass() -> None:
    """Guard against the stub drifting from the dependency it stands in for."""
    import dataclasses

    lightrag_utils = pytest.importorskip("lightrag.utils")

    real_fields = {field.name for field in dataclasses.fields(lightrag_utils.EmbeddingFunc)}
    stub_fields = set(inspect.signature(_FakeEmbeddingFunc.__init__).parameters.keys() - {"self"})
    assert stub_fields == real_fields


def test_embedding_func_returns_numpy_array(monkeypatch) -> None:
    _install_fake_lightrag(monkeypatch)

    class _Config:
        dim = 3
        max_tokens = 99

    class _Client:
        async def embed(self, texts, *, input_type=None):
            del input_type
            return [[1, 2, 3] for _ in texts]

    monkeypatch.setattr("deeptutor.services.embedding.get_embedding_config", lambda: _Config())
    monkeypatch.setattr("deeptutor.services.embedding.get_embedding_client", lambda: _Client())

    bridge = _RecordingBridge()
    embedding = lr_config.build_embedding_func(io_bridge=bridge)
    vectors = asyncio.run(embedding.func(["a", "b"]))
    assert embedding.embedding_dim == 3
    assert embedding.max_token_size == 99
    assert vectors.shape == (2, 3)
    assert hasattr(vectors, "size")
    assert bridge.calls == 1


def test_embedding_func_maps_lightrag_query_and_document_context(monkeypatch) -> None:
    calls: list[tuple[list[str], str | None]] = []
    _install_fake_lightrag(monkeypatch)

    class _Config:
        dim = 3
        max_tokens = 99

    class _Client:
        async def embed(self, texts, *, input_type=None):
            calls.append((list(texts), input_type))
            return [[1, 2, 3] for _ in texts]

    monkeypatch.setattr("deeptutor.services.embedding.get_embedding_config", lambda: _Config())
    monkeypatch.setattr("deeptutor.services.embedding.get_embedding_client", lambda: _Client())

    embedding = lr_config.build_embedding_func()
    asyncio.run(embedding.func(["question"], context="query", _priority=1))
    asyncio.run(embedding.func(["passage"], context="document"))
    # The pinned LightRAG passes no context at all; that must mean "no role",
    # not "document", or every query would be embedded as a passage.
    asyncio.run(embedding.func(["unlabelled"]))

    assert calls == [
        (["question"], "search_query"),
        (["passage"], "search_document"),
        (["unlabelled"], None),
    ]


def test_lightrag_llm_adapter_preserves_messages_and_drops_extra_kwargs(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def get_model_func(self):
            async def model_func(prompt, **kwargs):
                captured["prompt"] = prompt
                captured.update(kwargs)
                return "ok"

            return model_func

    monkeypatch.setattr("deeptutor.services.llm.get_llm_client", lambda: _Client())

    bridge = _RecordingBridge()
    func = lr_config.build_llm_model_func(io_bridge=bridge)
    result = asyncio.run(
        func(
            "",
            system_prompt="sys",
            messages=[{"role": "user", "content": "from messages"}],
            response_format={"type": "json_object"},
            hashing_kv=object(),
            keyword_extraction=True,
        )
    )

    assert result == "ok"
    assert captured["prompt"] == ""
    assert captured["system_prompt"] == "sys"
    assert captured["history_messages"] == []
    assert captured["messages"] == [{"role": "user", "content": "from messages"}]
    assert "response_format" not in captured
    assert "hashing_kv" not in captured
    assert "keyword_extraction" not in captured
    assert bridge.calls == 1


def test_lightrag_llm_adapter_retries_only_malformed_entity_extraction(monkeypatch) -> None:
    responses = [
        "entity<|#|>Broken<|#|>person",
        "entity<|#|>Fixed<|#|>person<|#|>A complete description.\n<|COMPLETE|>",
    ]
    calls: list[dict[str, object]] = []

    class _Client:
        def get_model_func(self):
            async def model_func(prompt, **kwargs):
                calls.append({"prompt": prompt, **kwargs})
                return responses[len(calls) - 1]

            return model_func

    monkeypatch.setattr("deeptutor.services.llm.get_llm_client", lambda: _Client())

    bridge = _RecordingBridge()
    func = lr_config.build_llm_model_func(io_bridge=bridge)
    result = asyncio.run(
        func(
            "extract this source",
            system_prompt=(
                "You are a Knowledge Graph Specialist responsible for extracting "
                "entities and relationships."
            ),
        )
    )

    assert result == responses[1]
    assert bridge.calls == 2
    assert calls[0]["prompt"] == "extract this source"
    assert "previous extraction output" in str(calls[1]["prompt"])
    assert calls[1]["history_messages"] == [
        {"role": "user", "content": "extract this source"},
        {"role": "assistant", "content": responses[0]},
    ]


def test_lightrag_llm_adapter_does_not_retry_parseable_empty_extraction(monkeypatch) -> None:
    calls = 0

    class _Client:
        def get_model_func(self):
            async def model_func(prompt, **kwargs):
                nonlocal calls
                calls += 1
                return "<|COMPLETE|>"

            return model_func

    monkeypatch.setattr("deeptutor.services.llm.get_llm_client", lambda: _Client())

    func = lr_config.build_llm_model_func()
    result = asyncio.run(
        func(
            "no entities here",
            system_prompt=(
                "You are a Knowledge Graph Specialist responsible for extracting "
                "entities and relationships from the input text."
            ),
        )
    )

    assert result == "<|COMPLETE|>"
    assert calls == 1


def test_lightrag_llm_adapter_does_not_treat_description_summary_as_extraction(
    monkeypatch,
) -> None:
    calls = 0

    class _Client:
        def get_model_func(self):
            async def model_func(prompt, **kwargs):
                nonlocal calls
                calls += 1
                return "A concise entity description without extraction delimiters."

            return model_func

    monkeypatch.setattr("deeptutor.services.llm.get_llm_client", lambda: _Client())

    func = lr_config.build_llm_model_func()
    result = asyncio.run(
        func(
            "summarize these descriptions",
            system_prompt=(
                "You are a Knowledge Graph Specialist, proficient in data curation and synthesis."
            ),
        )
    )

    assert result == "A concise entity description without extraction delimiters."
    assert calls == 1


def test_lightrag_vision_adapter_preserves_messages(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def get_vision_model_func(self):
            async def model_func(prompt, **kwargs):
                captured["prompt"] = prompt
                captured.update(kwargs)
                return "ok"

            return model_func

    monkeypatch.setattr("deeptutor.services.llm.get_llm_client", lambda: _Client())

    bridge = _RecordingBridge()
    func = lr_config.build_vision_model_func(io_bridge=bridge)
    result = asyncio.run(
        func(
            "",
            image_data="abc123",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        )
    )

    assert result == "ok"
    assert captured["prompt"] == ""
    assert captured["image_data"] == "abc123"
    assert captured["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert bridge.calls == 1


def test_build_rag_skips_raganything_parser_install_check(monkeypatch) -> None:
    """Regression for issue #594.

    RAG-Anything validates its *default* parser (``mineru``) at LightRAG-init
    time, even though DeepTutor only ever inserts a pre-parsed ``content_list``
    and never uses RAG-Anything's parser. ``build_rag`` must pre-satisfy that
    check so indexing with a different parse engine (e.g. pymupdf4llm) doesn't
    hard-fail when MinerU is absent.
    """
    captured: dict[str, object] = {}

    class _FakeConfig:
        def __init__(self, *, working_dir) -> None:
            self.working_dir = working_dir
            self.parser = "mineru"  # RAG-Anything's default

    class _FakeRagAnything:
        def __init__(
            self, *, config, llm_model_func, vision_model_func, embedding_func, lightrag_kwargs=None
        ) -> None:
            # Mirror the real constructor: the install check starts unsatisfied.
            self._parser_installation_checked = False
            captured["config"] = config
            captured["lightrag_kwargs"] = lightrag_kwargs

    fake_module = types.ModuleType("raganything")
    fake_module.RAGAnything = _FakeRagAnything
    fake_module.RAGAnythingConfig = _FakeConfig
    monkeypatch.setitem(sys.modules, "raganything", fake_module)
    monkeypatch.setattr(engine, "build_llm_model_func", lambda: "llm")
    monkeypatch.setattr(engine, "build_vision_model_func", lambda: "vision")
    monkeypatch.setattr(engine, "build_embedding_func", lambda: "embed")

    rag = engine.build_rag(Path("/tmp/kb-wd"))  # noqa: S108

    assert rag._parser_installation_checked is True
    assert captured["config"].working_dir == "/tmp/kb-wd"


def test_insert_repairs_xml_invalid_latex_control_characters() -> None:
    captured: dict[str, object] = {}

    class _Rag:
        async def insert_content_list(self, **kwargs):
            captured.update(kwargs)

    original = [
        {
            "type": "text",
            "text": "i_\x08eta and f=\x0crac{a}{b}",
            "metadata": {"label": "bad\x00value"},
        }
    ]
    asyncio.run(engine.insert(_Rag(), original, file_name="chapter.pdf", doc_id="chapter"))

    cleaned = captured["content_list"]
    assert cleaned[0]["text"] == r"i_\beta and f=\frac{a}{b}"
    assert cleaned[0]["metadata"]["label"] == "bad\ufffdvalue"
    assert original[0]["text"] == "i_\x08eta and f=\x0crac{a}{b}"


def test_atomic_graphml_write_preserves_previous_file_on_failure(tmp_path, monkeypatch) -> None:
    nx = pytest.importorskip("networkx")
    target = tmp_path / storage.GRAPH_FILENAME
    original = b"<graphml><graph/></graphml>"
    target.write_bytes(original)

    def fail_after_partial_write(_graph, path):
        Path(path).write_text("<graphml><graph>", encoding="utf-8")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(nx, "write_graphml", fail_after_partial_write)
    with pytest.raises(OSError, match="interrupted write"):
        engine._atomic_write_nx_graph(nx.Graph(), str(target), "test")

    assert target.read_bytes() == original
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_atomic_graphml_write_sanitizes_graph_and_produces_parseable_file(tmp_path) -> None:
    nx = pytest.importorskip("networkx")
    target = tmp_path / storage.GRAPH_FILENAME
    target.write_text("<graphml><graph/></graphml>", encoding="utf-8")
    target.chmod(0o664)
    graph = nx.Graph()
    graph.add_node("\x08eta", description="i_\x08eta")

    engine._atomic_write_nx_graph(graph, str(target), "test")

    loaded = nx.read_graphml(target)
    assert list(loaded.nodes) == [r"\beta"]
    assert loaded.nodes[r"\beta"]["description"] == r"i_\beta"
    assert target.stat().st_mode & 0o777 == 0o664


def test_networkx_storage_patch_propagates_upstream_save_failure(monkeypatch) -> None:
    fake_lightrag = types.ModuleType("lightrag")
    fake_lightrag.__path__ = []
    fake_kg = types.ModuleType("lightrag.kg")
    fake_kg.__path__ = []
    fake_impl = types.ModuleType("lightrag.kg.networkx_impl")

    class _Storage:
        _graphml_xml_file = "/tmp/graph.graphml"  # noqa: S108

        @staticmethod
        def write_nx_graph(*_args):
            raise AssertionError("old writer should be replaced")

        async def index_done_callback(self):
            return False

    fake_impl.NetworkXStorage = _Storage
    monkeypatch.setitem(sys.modules, "lightrag", fake_lightrag)
    monkeypatch.setitem(sys.modules, "lightrag.kg", fake_kg)
    monkeypatch.setitem(sys.modules, "lightrag.kg.networkx_impl", fake_impl)

    engine._install_atomic_networkx_storage()

    assert _Storage.write_nx_graph is engine._atomic_write_nx_graph
    with pytest.raises(RuntimeError, match="failed to persist GraphML"):
        asyncio.run(_Storage().index_done_callback())


def test_lightrag_query_initializes_raganything_before_aquery(monkeypatch) -> None:
    calls: list[str] = []

    class _Rag:
        lightrag = None

        async def _ensure_lightrag_initialized(self):
            calls.append("ensure")
            self.lightrag = object()
            return {"success": True}

        async def aquery(self, question, mode=None, **kwargs):
            calls.append("aquery")
            assert self.lightrag is not None
            assert question == "hello"
            assert mode == "hybrid"
            assert kwargs == {}
            return "answer"

    monkeypatch.setattr(engine, "query_kwargs_from_settings", lambda: {})

    result = asyncio.run(engine.query(_Rag(), "hello", "hybrid"))

    assert result == "answer"
    assert calls == ["ensure", "aquery"]


def test_lightrag_query_surfaces_raganything_initialization_failure() -> None:
    class _Rag:
        lightrag = None

        async def _ensure_lightrag_initialized(self):
            return {"success": False, "error": "storage failed"}

        async def aquery(self, question, mode=None, **kwargs):  # pragma: no cover
            raise AssertionError("aquery should not run")

    with pytest.raises(RuntimeError, match="storage failed"):
        asyncio.run(engine.query(_Rag(), "hello", "hybrid"))


def test_lightrag_query_with_sources_keeps_answer_and_exposes_provenance(monkeypatch) -> None:
    """The answer path remains RAG-Anything while citations use LightRAG data."""
    captured: dict[str, object] = {}

    class _QueryParam:
        def __init__(self, **kwargs) -> None:
            captured["query_param"] = kwargs

    fake_lightrag = types.ModuleType("lightrag")
    fake_lightrag.QueryParam = _QueryParam
    monkeypatch.setitem(sys.modules, "lightrag", fake_lightrag)
    monkeypatch.setattr(engine, "query_kwargs_from_settings", lambda: {"top_k": 3})

    class _LightRag:
        async def aquery_data(self, question, *, param):
            captured["provenance_query"] = question
            captured["provenance_param"] = param
            return {
                "status": "success",
                "data": {
                    "references": [{"reference_id": "ref-1", "file_path": "/kb/book.pdf"}],
                    "chunks": [
                        {
                            "chunk_id": "chunk-1",
                            "content": "The retrieved passage.",
                            "reference_id": "ref-1",
                        }
                    ],
                    "entities": [
                        {
                            "entity_name": "Newton's laws",
                            "entity_type": "concept",
                            "description": "A mechanics foundation.",
                            "source_id": "chunk-1",
                            "reference_id": "ref-1",
                        }
                    ],
                    "relationships": [
                        {
                            "src_id": "force",
                            "tgt_id": "acceleration",
                            "description": "Force produces acceleration.",
                            "source_id": "chunk-1",
                            "reference_id": "ref-1",
                        }
                    ],
                },
            }

    class _Rag:
        lightrag = _LightRag()

        async def aquery(self, question, mode=None, **kwargs):
            captured["answer_query"] = (question, mode, kwargs)
            return "Grounded answer"

    answer, sources = asyncio.run(engine.query_with_sources(_Rag(), "What is force?", "hybrid"))

    assert answer == "Grounded answer"
    assert captured["answer_query"] == ("What is force?", "hybrid", {"top_k": 3})
    assert captured["provenance_query"] == "What is force?"
    assert captured["query_param"] == {"mode": "hybrid", "top_k": 3}
    assert sources == [
        {
            "title": "book.pdf",
            "content": "The retrieved passage.",
            "source": "/kb/book.pdf",
            "page": "",
            "chunk_id": "chunk-1",
            "reference_id": "ref-1",
        },
        {
            "title": "Newton's laws",
            "content": "A mechanics foundation.",
            "source": "/kb/book.pdf",
            "page": "",
            "entity_id": "Newton's laws",
            "entity_type": "concept",
            "source_id": "chunk-1",
            "reference_id": "ref-1",
        },
        {
            "title": "force->acceleration",
            "content": "Force produces acceleration.",
            "source": "/kb/book.pdf",
            "page": "",
            "relation_id": "force->acceleration",
            "source_entity_id": "force",
            "target_entity_id": "acceleration",
            "source_id": "chunk-1",
            "reference_id": "ref-1",
        },
    ]


def test_lightrag_query_sources_falls_back_when_structured_api_is_unavailable() -> None:
    class _Rag:
        lightrag = object()

    assert asyncio.run(engine.query_sources(_Rag(), "hello", "hybrid")) == []


# --------------------------------------------------------------------------- #
# pipeline lifecycle (engine + parse service stubbed)
# --------------------------------------------------------------------------- #


class _FakeRag:
    def __init__(self, working_dir) -> None:
        self.working_dir = Path(working_dir)


def _force_available(monkeypatch, available: bool = True) -> None:
    monkeypatch.setattr(lr_config, "is_lightrag_available", lambda: available)


def _stub_engine(monkeypatch, answer: str = "ANSWER") -> list[dict]:
    """Stub the engine so insert writes a readiness marker and query echoes."""
    inserts: list[dict] = []
    monkeypatch.setattr(engine, "build_rag", lambda wd, *_a, **_: _FakeRag(wd))

    async def fake_insert(rag, content_list, *, file_name, doc_id):
        inserts.append({"file": file_name, "doc_id": doc_id, "blocks": content_list})
        (rag.working_dir / "vdb_chunks.json").write_text(
            json.dumps({"vectors": [[1.0]]}), encoding="utf-8"
        )
        (rag.working_dir / "kv_store_doc_status.json").write_text(
            json.dumps(
                {
                    doc_id: {
                        "status": "processed",
                        "file_path": file_name,
                        "chunks_list": ["chunk-1"],
                    }
                }
            ),
            encoding="utf-8",
        )
        (rag.working_dir / storage.GRAPH_FILENAME).write_text(
            "<graphml><graph/></graphml>", encoding="utf-8"
        )

    async def fake_query_with_sources(rag, question, mode):
        return f"{answer}|{mode}", []

    monkeypatch.setattr(engine, "insert", fake_insert)
    monkeypatch.setattr(engine, "query_with_sources", fake_query_with_sources)
    return inserts


def _stub_parse(
    monkeypatch,
    *,
    blocks=None,
    markdown: str = "# md",
    engine_name: str = "fake",
    parser_signature: str = "",
) -> None:
    from deeptutor.services.parsing.types import ParsedDocument

    class _Service:
        def parse(self, path, **_):
            return ParsedDocument(
                markdown=markdown,
                blocks=blocks,
                source_hash="h_" + Path(path).stem,
                parser_signature=parser_signature,
                engine=engine_name,
            )

    monkeypatch.setattr("deeptutor.services.parsing.get_parse_service", lambda: _Service())


def test_indexing_isolated_from_owner_loop_with_context_and_progress(tmp_path, monkeypatch) -> None:
    """Regression for #761: local JSON work must not stall service I/O."""
    from deeptutor.services.parsing.types import ParsedDocument

    request_scope = contextvars.ContextVar("lightrag_test_scope", default="missing")
    captured: dict[str, object] = {"inserts": [], "progress": [], "parse_threads": []}

    class _ParseService:
        def parse(self, path, **_):
            captured["parse_threads"].append(threading.get_ident())
            source = Path(path)
            return ParsedDocument(
                markdown="",
                blocks=[{"type": "text", "text": source.stem, "page_idx": 0}],
                source_hash=f"hash-{source.stem}",
                engine="fake",
            )

    class _BlockingRag:
        def __init__(self, working_dir, io_bridge) -> None:
            self.working_dir = Path(working_dir)
            self.io_bridge = io_bridge

        async def insert_content_list(self, *, content_list, file_path, doc_id):
            captured["worker_thread"] = threading.get_ident()
            captured["worker_context"] = request_scope.get()
            captured["block_started_at"] = time.monotonic()
            time.sleep(0.15)

            async def fake_network_io():
                captured["io_thread"] = threading.get_ident()
                captured["io_context"] = request_scope.get()
                return "io-ok"

            captured["io_result"] = await self.io_bridge.run(fake_network_io)
            captured["inserts"].append(
                {"content_list": content_list, "file_path": file_path, "doc_id": doc_id}
            )
            self.working_dir.mkdir(parents=True, exist_ok=True)
            (self.working_dir / "kv_store_doc_status.json").write_text(
                json.dumps(
                    {
                        doc_id: {
                            "status": "processed",
                            "file_path": file_path,
                            "chunks_list": ["chunk-1"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (self.working_dir / storage.GRAPH_FILENAME).write_text(
                "<graphml><graph/></graphml>", encoding="utf-8"
            )

    def fake_build_rag(working_dir, *_a, io_bridge=None, **_):
        captured["build_thread"] = threading.get_ident()
        return _BlockingRag(working_dir, io_bridge)

    monkeypatch.setattr("deeptutor.services.parsing.get_parse_service", lambda: _ParseService())
    monkeypatch.setattr(engine, "build_rag", fake_build_rag)
    _force_available(monkeypatch, True)

    docs = [tmp_path / "one.pdf", tmp_path / "two.pdf"]
    for doc in docs:
        doc.write_bytes(b"%PDF")

    async def scenario() -> bool:
        owner_thread = threading.get_ident()
        captured["owner_thread"] = owner_thread
        request_scope.set("user-761")

        async def on_progress(current: int, total: int) -> None:
            await asyncio.sleep(0)
            captured["progress"].append(
                (current, total, threading.get_ident(), request_scope.get())
            )

        async def heartbeat() -> None:
            while "block_started_at" not in captured:
                await asyncio.sleep(0)
            await asyncio.sleep(0.01)
            captured["heartbeat_at"] = time.monotonic()

        pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
        indexing = asyncio.create_task(
            pipe.initialize("kb", [str(doc) for doc in docs], progress_callback=on_progress)
        )
        pulse = asyncio.create_task(heartbeat())
        result = await indexing
        await pulse
        return result

    assert asyncio.run(scenario()) is True
    owner_thread = captured["owner_thread"]
    assert captured["build_thread"] != owner_thread
    assert captured["worker_thread"] != owner_thread
    assert owner_thread not in captured["parse_threads"]
    assert captured["worker_thread"] not in captured["parse_threads"]
    assert captured["io_thread"] == owner_thread
    assert captured["worker_context"] == "user-761"
    assert captured["io_context"] == "user-761"
    assert captured["io_result"] == "io-ok"
    assert captured["heartbeat_at"] - captured["block_started_at"] < 0.1
    assert captured["progress"] == [
        (1, 2, owner_thread, "user-761"),
        (2, 2, owner_thread, "user-761"),
    ]
    assert captured["inserts"] == [
        {
            "content_list": [{"type": "text", "text": "one", "page_idx": 0}],
            "file_path": "one.pdf",
            "doc_id": "hash-one",
        },
        {
            "content_list": [{"type": "text", "text": "two", "page_idx": 0}],
            "file_path": "two.pdf",
            "doc_id": "hash-two",
        },
    ]


def test_indexing_parses_files_with_bounded_concurrency(tmp_path, monkeypatch) -> None:
    from deeptutor.services.parsing.types import ParsedDocument

    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    class _ParseService:
        def parse(self, path, **_):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                time.sleep(0.05)
                source = Path(path)
                return ParsedDocument(
                    markdown=source.stem,
                    blocks=[],
                    source_hash=f"hash-{source.stem}",
                    engine="fake",
                )
            finally:
                with lock:
                    state["active"] -= 1

    monkeypatch.setattr("deeptutor.services.parsing.get_parse_service", lambda: _ParseService())
    monkeypatch.setattr(
        "deeptutor.services.rag.pipelines.lightrag.config.indexing_kwargs_from_settings",
        lambda: {"max_concurrent_files": 2},
    )
    inserts = _stub_engine(monkeypatch)
    _force_available(monkeypatch, True)
    docs = [tmp_path / f"doc-{index}.pdf" for index in range(4)]
    for doc in docs:
        doc.write_bytes(b"%PDF")

    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    assert asyncio.run(pipe.initialize("kb", [str(doc) for doc in docs])) is True

    assert state["peak"] == 2
    assert [insert["file"] for insert in inserts] == [doc.name for doc in docs]


def test_indexing_worker_exception_propagates_unchanged(tmp_path, monkeypatch) -> None:
    class _IndexingFailure(RuntimeError):
        pass

    class _FailingRag:
        def __init__(self, working_dir) -> None:
            self.working_dir = Path(working_dir)

        async def insert_content_list(self, **_):
            raise _IndexingFailure("nano-vdb merge failed")

    monkeypatch.setattr(engine, "build_rag", lambda wd, *_a, **_: _FailingRag(wd))
    _stub_parse(monkeypatch, blocks=[{"type": "text", "text": "x", "page_idx": 0}])
    _force_available(monkeypatch, True)
    document = tmp_path / "bad.pdf"
    document.write_bytes(b"%PDF")

    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    with pytest.raises(_IndexingFailure, match="nano-vdb merge failed"):
        asyncio.run(pipe.initialize("kb", [str(document)]))


def test_indexing_cancellation_waits_for_worker_loop_to_close() -> None:
    started = threading.Event()
    stopped = threading.Event()
    owner_callback_called = False
    worker_loop: asyncio.AbstractEventLoop | None = None

    async def scenario() -> None:
        async def job(io_bridge) -> None:
            nonlocal owner_callback_called, worker_loop
            worker_loop = asyncio.get_running_loop()
            started.set()
            try:
                # Stand in for an uninterruptible synchronous NanoVectorDB
                # flush. Cancellation is observed at the next bridge call.
                time.sleep(0.05)

                def owner_callback() -> None:
                    nonlocal owner_callback_called
                    owner_callback_called = True

                await io_bridge.call(owner_callback)
            finally:
                stopped.set()

        task = asyncio.create_task(run_in_worker_loop(job))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert stopped.is_set()
    assert worker_loop is not None
    assert worker_loop.is_closed()
    assert owner_callback_called is False


def test_shared_worker_loop_survives_cancellation() -> None:
    """Cancelling one job must leave the persistent loop usable — LightRAG's
    process-global locks stay bound to it for the process lifetime."""

    async def scenario() -> None:
        started = threading.Event()

        async def slow_job(_bridge) -> None:
            started.set()
            await asyncio.sleep(60)

        task = asyncio.create_task(run_in_shared_worker_loop(slow_job))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        async def ping(_bridge) -> str:
            return "alive"

        assert await run_in_shared_worker_loop(ping) == "alive"

    asyncio.run(scenario())


def test_shared_worker_loop_honors_cancellation_before_submit() -> None:
    """A queued job must not start after its owner has already cancelled.

    The shared loop can be busy in synchronous GraphML/FAISS work, leaving the
    thread-safe ``submit`` callback queued for a while. Cancellation during
    that window used to miss the not-yet-created Task; it then ran anyway and
    tried to publish into an already-cancelled concurrent Future.
    """

    async def scenario() -> None:
        from deeptutor.services.rag.pipelines.lightrag import worker

        shared_loop = worker._ensure_shared_loop()
        release_loop = threading.Event()
        job_started = threading.Event()
        shared_loop.call_soon_threadsafe(lambda: release_loop.wait(1.0))

        async def job(_bridge) -> None:
            job_started.set()

        task = asyncio.create_task(run_in_shared_worker_loop(job))
        await asyncio.sleep(0)  # queue submit behind the blocking callback
        task.cancel()
        try:
            await asyncio.sleep(0.05)
            completed_before_release = task.done()
        finally:
            release_loop.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.01)
        assert completed_before_release is True
        assert job_started.is_set() is False

    asyncio.run(scenario())


def test_indexing_jobs_share_one_persistent_worker_loop(tmp_path, monkeypatch) -> None:
    """Regression: LightRAG keeps asyncio locks in process-global registries
    (``pipeline_status:*``). A fresh worker loop per indexing run leaves those
    locks bound to a dead loop, so every later run fails with "bound to a
    different event loop". All indexing jobs must ride one persistent loop,
    even across separate requests (separate owner loops)."""
    _force_available(monkeypatch, True)
    _stub_engine_counting(monkeypatch)  # provides the insert/query fakes
    build_threads: list[int] = []

    def fake_build(wd, *_a, **_):
        build_threads.append(threading.get_ident())
        return _FakeRag(wd)

    monkeypatch.setattr(engine, "build_rag", fake_build)
    _stub_parse(monkeypatch)
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    one = tmp_path / "one.pdf"
    one.write_bytes(b"%PDF")
    two = tmp_path / "two.pdf"
    two.write_bytes(b"%PDF")

    asyncio.run(pipe.initialize("kb", [str(one)]))
    asyncio.run(pipe.add_documents("kb", [str(two)]))

    assert len(build_threads) == 2
    assert build_threads[0] == build_threads[1]


def test_initialize_requires_lightrag(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, False)
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    with pytest.raises(lr_config.LightRagNotAvailableError):
        asyncio.run(pipe.initialize("kb", [str(pdf)]))


def test_initialize_orchestrates_index_and_uses_blocks(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, True)
    inserts = _stub_engine(monkeypatch)
    _stub_parse(monkeypatch, blocks=[{"type": "text", "text": "hi", "page_idx": 0}])
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "exam.pdf"
    pdf.write_bytes(b"%PDF")

    ok = asyncio.run(pipe.initialize("kb", [str(pdf)]))
    assert ok is True
    assert len(inserts) == 1
    assert inserts[0]["file"] == "exam.pdf"
    # blocks from the parse layer are passed through verbatim (multimodal path).
    assert inserts[0]["blocks"] == [{"type": "text", "text": "hi", "page_idx": 0}]
    # version dir is marked ready.
    root = resolve_storage_dir_for_read(tmp_path / "kb", None)
    assert storage.has_output(root) is True


def test_initialize_filters_only_mineru_layout_blocks(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, True)
    inserts = _stub_engine(monkeypatch)
    raw_blocks = [
        {"type": "header", "text": "chapter", "page_idx": 0},
        {"type": "text", "text": "body", "page_idx": 0},
        {"type": "image", "img_path": "/tmp/image.png", "page_idx": 0},  # noqa: S108
        {"type": "footer", "text": "publisher", "page_idx": 0},
        {"type": "page_number", "text": "1", "page_idx": 0},
    ]
    original = json.loads(json.dumps(raw_blocks))
    _stub_parse(
        monkeypatch,
        blocks=raw_blocks,
        engine_name="mineru",
        parser_signature="mineru-signature",
    )
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "exam.pdf"
    pdf.write_bytes(b"%PDF")

    assert asyncio.run(pipe.initialize("kb", [str(pdf)])) is True

    assert [item["type"] for item in inserts[0]["blocks"]] == ["text", "image"]
    assert raw_blocks == original
    root = resolve_storage_dir_for_read(tmp_path / "kb", None)
    assert root is not None
    ledgers = list((root / block_policy.LEDGER_DIRNAME).glob("*.json"))
    assert len(ledgers) == 1
    ledger = json.loads(ledgers[0].read_text(encoding="utf-8"))
    assert ledger["counts"]["raw_total"] == 5
    assert ledger["counts"]["filtered_total"] == 3
    assert ledger["counts"]["eligible_multimodal_total"] == 1
    assert ledger["counts"]["unknown_total"] == 0
    assert ledger["decision"]["ledger_role"] == "current-index"
    assert ledger["decision"]["policy_outcome"] == "accepted"
    attempts = list((tmp_path / "kb" / block_policy.ATTEMPT_LEDGER_DIRNAME).glob("*.json"))
    assert len(attempts) == 1
    accepted_attempt = json.loads(attempts[0].read_text(encoding="utf-8"))
    assert accepted_attempt["decision"]["policy_outcome"] == "accepted"
    assert accepted_attempt["decision"]["attempt_id"] == ledger["decision"]["attempt_id"]


def test_initialize_indexes_unknown_mineru_types_and_records_them(tmp_path, monkeypatch) -> None:
    """A new MinerU block type must not take the whole ingest down.

    The type is unrecognized, not unwanted: index it, record the count so the
    policy can be extended, and keep the block's own text out of the audit
    file.
    """
    _force_available(monkeypatch, True)
    inserts = _stub_engine(monkeypatch)
    _stub_parse(
        monkeypatch,
        blocks=[{"type": "future_widget", "text": "raw-block-secret", "page_idx": 0}],
        engine_name="mineru",
    )
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "exam.pdf"
    pdf.write_bytes(b"%PDF")

    assert asyncio.run(pipe.initialize("kb", [str(pdf)])) is True

    assert len(inserts) == 1
    assert resolve_storage_dir_for_read(tmp_path / "kb", None) is not None
    attempts = list((tmp_path / "kb" / block_policy.ATTEMPT_LEDGER_DIRNAME).glob("*.json"))
    assert len(attempts) == 1
    recorded = json.loads(attempts[0].read_text(encoding="utf-8"))
    assert recorded["counts"]["unknown_by_type"] == {"future_widget": 1}
    assert recorded["decision"]["ledger_role"] == "attempt"
    assert recorded["decision"]["policy_outcome"] == "unknown_types"
    assert "raw-block-secret" not in attempts[0].read_text(encoding="utf-8")


def test_add_documents_records_unknown_types_without_blocking_ingest(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, True)
    inserts = _stub_engine(monkeypatch)
    _stub_parse(
        monkeypatch,
        blocks=[{"type": "text", "text": "accepted", "page_idx": 0}],
        engine_name="mineru",
        parser_signature="accepted-signature",
    )
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "exam.pdf"
    pdf.write_bytes(b"%PDF")

    assert asyncio.run(pipe.initialize("kb", [str(pdf)])) is True
    root = resolve_storage_dir_for_read(tmp_path / "kb", None)
    assert root is not None
    current_path = next((root / block_policy.LEDGER_DIRNAME).glob("*.json"))
    accepted_payload = current_path.read_text(encoding="utf-8")

    _stub_parse(
        monkeypatch,
        blocks=[{"type": "future_widget", "text": "later", "page_idx": 0}],
        engine_name="mineru",
        parser_signature="unknown-type-signature",
    )
    assert asyncio.run(pipe.add_documents("kb", [str(pdf)])) is True

    assert len(inserts) == 2
    attempts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "kb" / block_policy.ATTEMPT_LEDGER_DIRNAME).glob("*.json")
    ]
    assert len(attempts) == 2
    recorded = next(
        item for item in attempts if item["decision"]["policy_outcome"] == "unknown_types"
    )
    assert recorded["parser"]["parser_signature"] == "unknown-type-signature"
    assert recorded["counts"]["unknown_total"] == 1


def test_add_documents_insert_failure_keeps_current_accepted_ledger(
    tmp_path,
    monkeypatch,
) -> None:
    _force_available(monkeypatch, True)
    _stub_engine(monkeypatch)
    _stub_parse(
        monkeypatch,
        blocks=[{"type": "text", "text": "accepted", "page_idx": 0}],
        engine_name="mineru",
        parser_signature="accepted-signature",
    )
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "exam.pdf"
    pdf.write_bytes(b"%PDF")

    assert asyncio.run(pipe.initialize("kb", [str(pdf)])) is True
    root = resolve_storage_dir_for_read(tmp_path / "kb", None)
    assert root is not None
    current_path = next((root / block_policy.LEDGER_DIRNAME).glob("*.json"))
    accepted_payload = current_path.read_text(encoding="utf-8")

    _stub_parse(
        monkeypatch,
        blocks=[{"type": "text", "text": "new attempt", "page_idx": 0}],
        engine_name="mineru",
        parser_signature="new-signature",
    )

    async def fail_insert(*_args, **_kwargs) -> None:
        raise RuntimeError("insert failed")

    monkeypatch.setattr(engine, "insert", fail_insert)
    with pytest.raises(RuntimeError, match="insert failed"):
        asyncio.run(pipe.add_documents("kb", [str(pdf)]))

    assert current_path.read_text(encoding="utf-8") == accepted_payload
    attempts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "kb" / block_policy.ATTEMPT_LEDGER_DIRNAME).glob("*.json")
    ]
    assert len(attempts) == 2
    latest_attempt = next(
        item for item in attempts if item["parser"]["parser_signature"] == "new-signature"
    )
    assert latest_attempt["decision"]["policy_outcome"] == "accepted"
    assert latest_attempt["counts"]["unknown_total"] == 0


def test_ingest_falls_back_to_markdown_when_no_blocks(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, True)
    inserts = _stub_engine(monkeypatch)
    _stub_parse(monkeypatch, blocks=None, markdown="# only markdown")
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "notes.pdf"
    pdf.write_bytes(b"%PDF")

    asyncio.run(pipe.initialize("kb", [str(pdf)]))
    assert inserts[0]["blocks"] == [{"type": "text", "text": "# only markdown", "page_idx": 0}]


def test_initialize_no_content_returns_false(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, True)
    inserts = _stub_engine(monkeypatch)
    _stub_parse(monkeypatch, blocks=None, markdown="")  # empty parse
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(b"%PDF")

    ok = asyncio.run(pipe.initialize("kb", [str(pdf)]))
    assert ok is False
    assert inserts == []


def test_initialize_fails_when_lightrag_records_doc_failure(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, True)
    monkeypatch.setattr(engine, "build_rag", lambda wd, *_a, **_: _FakeRag(wd))

    async def fake_insert(rag, content_list, *, file_name, doc_id):
        (rag.working_dir / "kv_store_doc_status.json").write_text(
            json.dumps(
                {
                    doc_id: {
                        "status": "failed",
                        "file_path": file_name,
                        "error_msg": "'list' object has no attribute 'size'",
                        "chunks_list": [],
                    }
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(engine, "insert", fake_insert)
    _stub_parse(monkeypatch, blocks=[{"type": "text", "text": "hi", "page_idx": 0}])
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    docx = tmp_path / "bad.docx"
    docx.write_bytes(b"docx")

    with pytest.raises(RuntimeError, match="list.*size"):
        asyncio.run(pipe.initialize("kb", [str(docx)]))

    assert resolve_storage_dir_for_read(tmp_path / "kb", None) is None


def test_search_needs_reindex_without_output(tmp_path) -> None:
    res = asyncio.run(LightRagPipeline(kb_base_dir=str(tmp_path)).search("q", "missing"))
    assert res["needs_reindex"] is True
    assert res["provider"] == "lightrag"


def _write_ready_lightrag_store(root: Path, *, graph: str) -> None:
    root.mkdir(parents=True)
    (root / "kv_store_doc_status.json").write_text(
        json.dumps(
            {
                "doc-1": {
                    "status": "processed",
                    "file_path": "chapter2.pdf",
                    "chunks_list": ["chunk-1"],
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "vdb_chunks.json").write_text(json.dumps({"vectors": [[1.0]]}), encoding="utf-8")
    (root / storage.GRAPH_FILENAME).write_text(graph, encoding="utf-8")
    (root / storage.META_FILENAME).write_text(
        json.dumps({"provider": "lightrag", "vector_storage": "nano"}),
        encoding="utf-8",
    )


def test_add_documents_rejects_corrupt_existing_graph_before_loading_engine(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "kb" / "version-1"
    _write_ready_lightrag_store(root, graph="<graphml><graph>")
    _force_available(monkeypatch, True)
    monkeypatch.setattr(
        engine,
        "build_rag",
        lambda *_a, **_k: pytest.fail("corrupt graph must be rejected before engine load"),
    )
    pdf = tmp_path / "chapter3.pdf"
    pdf.write_bytes(b"%PDF")

    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="corrupted.*Rebuild"):
        asyncio.run(pipe.add_documents("kb", [str(pdf)]))


def test_search_reports_corrupt_graph_as_needing_reindex(tmp_path, monkeypatch) -> None:
    root = tmp_path / "kb" / "version-1"
    _write_ready_lightrag_store(root, graph="<graphml><graph>")
    monkeypatch.setattr(
        engine,
        "build_rag",
        lambda *_a, **_k: pytest.fail("corrupt graph must not be queried"),
    )

    result = asyncio.run(LightRagPipeline(kb_base_dir=str(tmp_path)).search("q", "kb"))

    assert result["error_type"] == "corrupt_index"
    assert result["needs_reindex"] is True
    assert "Rebuild" in result["answer"]


def test_search_not_configured_when_unavailable(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, True)
    _stub_engine(monkeypatch)
    _stub_parse(monkeypatch, blocks=[{"type": "text", "text": "x", "page_idx": 0}])
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    asyncio.run(pipe.initialize("kb", [str(pdf)]))

    _force_available(monkeypatch, False)
    res = asyncio.run(pipe.search("q", "kb"))
    assert res["error_type"] == "not_configured"


def test_search_happy_path_resolves_mode(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, True)
    _stub_engine(monkeypatch, answer="GROUNDED")
    _stub_parse(monkeypatch, blocks=[{"type": "text", "text": "x", "page_idx": 0}])
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    asyncio.run(pipe.initialize("kb", [str(pdf)]))

    # Per-KB search_mode is read from kb_config.json next to the store.
    (tmp_path / "kb_config.json").write_text(
        json.dumps({"knowledge_bases": {"kb": {"search_mode": "local"}}}), encoding="utf-8"
    )
    res = asyncio.run(pipe.search("question?", "kb"))
    assert res["answer"] == "GROUNDED|local"
    assert res["mode"] == "local"
    assert res["provider"] == "lightrag"


def test_search_returns_lightrag_provenance_sources(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, True)
    _stub_engine(monkeypatch)
    _stub_parse(monkeypatch, blocks=[{"type": "text", "text": "x", "page_idx": 0}])
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    asyncio.run(pipe.initialize("kb", [str(pdf)]))

    async def fake_query_with_sources(rag, question, mode):
        return "Grounded", [{"title": "a.pdf", "chunk_id": "chunk-1"}]

    monkeypatch.setattr(engine, "query_with_sources", fake_query_with_sources)

    res = asyncio.run(pipe.search("question?", "kb"))

    assert res["answer"] == "Grounded"
    assert res["sources"] == [{"title": "a.pdf", "chunk_id": "chunk-1"}]


def test_explicit_mode_overrides_kb_config(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, True)
    _stub_engine(monkeypatch, answer="A")
    _stub_parse(monkeypatch, blocks=[{"type": "text", "text": "x", "page_idx": 0}])
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    asyncio.run(pipe.initialize("kb", [str(pdf)]))

    res = asyncio.run(pipe.search("q", "kb", mode="global"))
    assert res["mode"] == "global"


def test_global_provider_mode_used_when_kb_has_none(tmp_path, monkeypatch) -> None:
    _force_available(monkeypatch, True)
    _stub_engine(monkeypatch, answer="A")
    _stub_parse(monkeypatch, blocks=[{"type": "text", "text": "x", "page_idx": 0}])
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    asyncio.run(pipe.initialize("kb", [str(pdf)]))

    # No per-KB search_mode, but a global default mode set from the engine card.
    (tmp_path / "kb_config.json").write_text(
        json.dumps({"defaults": {"provider_modes": {"lightrag": "naive"}}}), encoding="utf-8"
    )
    res = asyncio.run(pipe.search("q", "kb"))
    assert res["mode"] == "naive"


# --------------------------------------------------------------------------- #
# vector-storage engine selection (nano / faiss)
# --------------------------------------------------------------------------- #


def _fake_raganything_module(captured: dict) -> types.ModuleType:
    class _FakeConfig:
        def __init__(self, *, working_dir) -> None:
            self.working_dir = working_dir
            self.parser = "mineru"

    class _FakeRagAnything:
        def __init__(
            self, *, config, llm_model_func, vision_model_func, embedding_func, lightrag_kwargs=None
        ) -> None:
            self._parser_installation_checked = True
            captured["lightrag_kwargs"] = lightrag_kwargs

    fake_module = types.ModuleType("raganything")
    fake_module.RAGAnything = _FakeRagAnything
    fake_module.RAGAnythingConfig = _FakeConfig
    return fake_module


def _stub_raganything(monkeypatch) -> dict:
    captured: dict[str, object] = {}
    monkeypatch.setitem(sys.modules, "raganything", _fake_raganything_module(captured))
    monkeypatch.setattr(engine, "build_llm_model_func", lambda: "llm")
    monkeypatch.setattr(engine, "build_vision_model_func", lambda: "vision")
    monkeypatch.setattr(engine, "build_embedding_func", lambda: "embed")
    monkeypatch.setattr(
        engine,
        "indexing_kwargs_from_settings",
        lambda: {
            "llm_model_max_async": 8,
            "embedding_func_max_async": 2,
            "max_parallel_insert": 8,
            "entity_extract_max_gleaning": 0,
            "chunk_token_size": 1400,
            "chunk_overlap_token_size": 80,
            "embedding_batch_num": 20,
            "force_llm_summary_on_merge": 16,
        },
    )
    return captured


def _stub_native_lightrag(monkeypatch) -> dict:
    captured: dict[str, object] = {"pipeline_status_calls": 0}

    class _FakeQueryParam:
        def __init__(
            self,
            mode="mix",
            top_k=40,
            response_type="Multiple Paragraphs",
        ) -> None:
            self.mode = mode
            self.top_k = top_k
            self.response_type = response_type

    class _FakeLightRAG:
        def __init__(
            self,
            working_dir,
            llm_model_func,
            embedding_func,
            vector_storage=None,
            enable_llm_cache=True,
            enable_llm_cache_for_entity_extract=True,
            default_llm_timeout=180,
            default_embedding_timeout=30,
            llm_model_max_async=4,
            embedding_func_max_async=8,
            max_parallel_insert=2,
            entity_extract_max_gleaning=1,
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            embedding_batch_num=10,
            force_llm_summary_on_merge=8,
        ) -> None:
            self.working_dir = working_dir
            self.role_llm_funcs = {}
            self.embedding_func = embedding_func
            captured["constructor"] = {
                "working_dir": working_dir,
                "llm_model_func": llm_model_func,
                "embedding_func": embedding_func,
                "vector_storage": vector_storage,
                "enable_llm_cache": enable_llm_cache,
                "enable_llm_cache_for_entity_extract": enable_llm_cache_for_entity_extract,
                "default_llm_timeout": default_llm_timeout,
                "default_embedding_timeout": default_embedding_timeout,
                "llm_model_max_async": llm_model_max_async,
                "embedding_func_max_async": embedding_func_max_async,
                "max_parallel_insert": max_parallel_insert,
                "entity_extract_max_gleaning": entity_extract_max_gleaning,
                "chunk_token_size": chunk_token_size,
                "chunk_overlap_token_size": chunk_overlap_token_size,
                "embedding_batch_num": embedding_batch_num,
                "force_llm_summary_on_merge": force_llm_summary_on_merge,
            }

        async def initialize_storages(self) -> None:
            captured["initialized"] = True

        async def aquery(self, question, param=None):
            captured["query"] = (question, param)
            return "native answer"

        async def finalize_storages(self) -> None:
            captured["finalized"] = True

    async def initialize_pipeline_status() -> None:
        captured["pipeline_status_calls"] = int(captured["pipeline_status_calls"]) + 1

    fake_lightrag = types.ModuleType("lightrag")
    fake_lightrag.__path__ = []
    fake_lightrag.LightRAG = _FakeLightRAG
    fake_lightrag.QueryParam = _FakeQueryParam
    fake_kg = types.ModuleType("lightrag.kg")
    fake_kg.__path__ = []
    fake_shared = types.ModuleType("lightrag.kg.shared_storage")
    fake_shared.initialize_pipeline_status = initialize_pipeline_status
    monkeypatch.setitem(sys.modules, "lightrag", fake_lightrag)
    monkeypatch.setitem(sys.modules, "lightrag.kg", fake_kg)
    monkeypatch.setitem(sys.modules, "lightrag.kg.shared_storage", fake_shared)
    monkeypatch.setattr(engine, "_is_query_only", lambda: True)
    monkeypatch.setattr(engine, "build_llm_model_func", lambda **_: "llm")
    monkeypatch.setattr(engine, "build_embedding_func", lambda **_: "embed")
    monkeypatch.setattr(
        engine,
        "build_vision_model_func",
        lambda **_: (_ for _ in ()).throw(AssertionError("query runtime must not build vision")),
    )
    monkeypatch.setattr(engine, "_install_lean_faiss_storage", lambda: None)
    monkeypatch.setattr(engine.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        engine,
        "indexing_kwargs_from_settings",
        lambda: {"max_concurrent_files": 8},
    )
    monkeypatch.setattr(
        engine,
        "lightrag_kwargs_from_settings",
        lambda: {
            "llm_model_max_async": 8,
            "embedding_func_max_async": 2,
            "max_parallel_insert": 8,
            "entity_extract_max_gleaning": 0,
            "chunk_token_size": 1400,
            "chunk_overlap_token_size": 80,
            "embedding_batch_num": 20,
            "force_llm_summary_on_merge": 16,
        },
    )
    return captured


def test_build_rag_forwards_faiss_vector_storage(monkeypatch) -> None:
    captured = _stub_raganything(monkeypatch)
    # The lean-storage install is a separate concern (covered by
    # test_lean_faiss_*) and requires the real lightrag/faiss packages.
    monkeypatch.setattr(engine, "_install_lean_faiss_storage", lambda: None)
    monkeypatch.setattr(engine.importlib.util, "find_spec", lambda name: object())
    engine.build_rag(Path("/tmp/kb-wd"), "faiss")  # noqa: S108
    assert captured["lightrag_kwargs"] == {
        "vector_storage": "FaissVectorDBStorage",
        "default_llm_timeout": engine._LIGHTRAG_LLM_TIMEOUT_S,
        "default_embedding_timeout": engine._LIGHTRAG_EMBEDDING_TIMEOUT_S,
        "llm_model_max_async": 8,
        "embedding_func_max_async": 2,
        "max_parallel_insert": 8,
        "entity_extract_max_gleaning": 0,
        "chunk_token_size": 1400,
        "chunk_overlap_token_size": 80,
        "embedding_batch_num": 20,
        "force_llm_summary_on_merge": 16,
    }


def test_build_rag_nano_and_unknown_pass_no_storage_kwarg(monkeypatch) -> None:
    captured = _stub_raganything(monkeypatch)
    expected = {
        "default_llm_timeout": engine._LIGHTRAG_LLM_TIMEOUT_S,
        "default_embedding_timeout": engine._LIGHTRAG_EMBEDDING_TIMEOUT_S,
        "llm_model_max_async": 8,
        "embedding_func_max_async": 2,
        "max_parallel_insert": 8,
        "entity_extract_max_gleaning": 0,
        "chunk_token_size": 1400,
        "chunk_overlap_token_size": 80,
        "embedding_batch_num": 20,
        "force_llm_summary_on_merge": 16,
    }

    engine.build_rag(Path("/tmp/kb-wd"))  # noqa: S108  # default == nano
    assert captured["lightrag_kwargs"] == expected

    engine.build_rag(Path("/tmp/kb-wd"), "nano")  # noqa: S108
    assert captured["lightrag_kwargs"] == expected

    # Unknown ids fall back to nano instead of breaking indexing.
    engine.build_rag(Path("/tmp/kb-wd"), "qdrant-whatever")  # noqa: S108
    assert captured["lightrag_kwargs"] == expected


def test_query_only_build_uses_native_lightrag_without_insertion_surface(monkeypatch) -> None:
    captured = _stub_native_lightrag(monkeypatch)
    monkeypatch.setattr(
        engine,
        "query_kwargs_from_settings",
        lambda: {"top_k": 60, "response_type": "Multiple Paragraphs"},
    )

    rag = engine.build_rag(Path("/tmp/kb-wd"), "faiss")  # noqa: S108
    answer = asyncio.run(engine.query(rag, "hello", "hybrid"))
    asyncio.run(engine.finalize(rag, cancel_pending=False))

    assert not hasattr(rag, "insert_content_list")
    assert captured["constructor"] == {
        "working_dir": "/tmp/kb-wd",
        "llm_model_func": "llm",
        "embedding_func": "embed",
        "vector_storage": "FaissVectorDBStorage",
        "enable_llm_cache": False,
        "enable_llm_cache_for_entity_extract": False,
        "default_llm_timeout": engine._LIGHTRAG_LLM_TIMEOUT_S,
        "default_embedding_timeout": engine._LIGHTRAG_EMBEDDING_TIMEOUT_S,
        "llm_model_max_async": 8,
        "embedding_func_max_async": 2,
        "max_parallel_insert": 8,
        "entity_extract_max_gleaning": 0,
        "chunk_token_size": 1400,
        "chunk_overlap_token_size": 80,
        "embedding_batch_num": 20,
        "force_llm_summary_on_merge": 16,
    }
    question, param = captured["query"]
    assert answer == "native answer"
    assert question == "hello"
    assert param.mode == "hybrid"
    assert param.top_k == 60
    assert param.response_type == "Multiple Paragraphs"
    assert captured["initialized"] is True
    assert captured["pipeline_status_calls"] == 1
    assert captured["finalized"] is True


def test_native_query_facade_matches_installed_lightrag_lifecycle(tmp_path) -> None:
    """Exercise the real optional dependency when it is present locally."""
    np = pytest.importorskip("numpy")
    lightrag = pytest.importorskip("lightrag")
    lightrag_utils = pytest.importorskip("lightrag.utils")

    async def fake_llm(_prompt, **_kwargs):
        return "unused"

    async def fake_embedding(texts):
        return np.ones((len(texts), 2), dtype=np.float32)

    async def scenario() -> None:
        native = lightrag.LightRAG(
            working_dir=str(tmp_path),
            llm_model_func=fake_llm,
            embedding_func=lightrag_utils.EmbeddingFunc(
                embedding_dim=2,
                max_token_size=16,
                func=fake_embedding,
            ),
            enable_llm_cache=False,
            enable_llm_cache_for_entity_extract=False,
        )
        rag = engine._NativeQueryRag(native)
        await engine.ensure_ready(rag)
        assert rag._ready is True
        await engine.finalize(rag, cancel_pending=False)
        assert rag._ready is False

    asyncio.run(scenario())


def test_lightrag_queue_shutdown_supports_old_and_new_signatures() -> None:
    calls: list[object] = []

    async def old_shutdown() -> None:
        calls.append("old")

    async def new_shutdown(*, graceful: bool, timeout: float) -> None:
        calls.append((graceful, timeout))

    async def old_func():
        return None

    async def new_func():
        return None

    old_func.shutdown = old_shutdown
    new_func.shutdown = new_shutdown
    fake_lightrag = types.SimpleNamespace(
        role_llm_funcs={"old": old_func, "new": new_func},
        embedding_func=None,
        rerank_model_func=None,
    )

    asyncio.run(engine._shutdown_queues(fake_lightrag, cancel_pending=True))

    assert calls == ["old", (False, 5.0)]


def test_build_rag_lightrag_watchdog_stays_a_backstop(monkeypatch) -> None:
    """LightRAG hard-cancels injected calls running past 2× default_*_timeout.

    Those outer caps must stay looser than DeepTutor's own per-attempt
    wall-clock cap (900s, DEEPTUTOR_LLM_ATTEMPT_TIMEOUT_S default) and the
    embedding client's full retry budget (~390s), or a slow/hung-but-
    recoverable call becomes a hard document failure with no retry.
    """
    captured = _stub_raganything(monkeypatch)
    engine.build_rag(Path("/tmp/kb-wd"))  # noqa: S108
    kwargs = captured["lightrag_kwargs"]
    assert kwargs["default_llm_timeout"] * 2 > 900
    assert kwargs["default_embedding_timeout"] * 2 > 390


def test_write_meta_records_and_pins_vector_storage(tmp_path) -> None:
    root = tmp_path / "version-1"
    root.mkdir()
    storage.write_meta(root, "faiss")
    meta = json.loads((root / storage.META_FILENAME).read_text())
    assert meta["vector_storage"] == "faiss"
    # A pinned version must reopen with its own engine regardless of defaults.
    assert storage.read_vector_storage(root) == "faiss"


def test_read_vector_storage_legacy_version_defaults_nano(tmp_path, monkeypatch) -> None:
    # Pre-feature version: has LightRAG output but meta.json lacks the field.
    def _boom():
        raise AssertionError("global settings must not be consulted for legacy versions")

    monkeypatch.setattr("deeptutor.services.config.load_lightrag_settings", _boom)
    root = tmp_path / "version-1"
    root.mkdir()
    (root / "vdb_chunks.json").write_text(json.dumps({"vectors": [[1.0]]}), encoding="utf-8")
    (root / "kv_store_doc_status.json").write_text(
        json.dumps({"d1": {"status": "processed", "file_path": "a.pdf", "chunks_list": ["c1"]}}),
        encoding="utf-8",
    )
    (root / storage.META_FILENAME).write_text(
        json.dumps({"version": "version-1"}), encoding="utf-8"
    )
    assert storage.read_vector_storage(root) == "nano"


def test_read_vector_storage_fresh_dir_uses_global_setting(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "deeptutor.services.config.load_lightrag_settings",
        lambda: {"vector_storage": "faiss"},
    )
    assert storage.read_vector_storage(tmp_path / "version-1") == "faiss"
    assert storage.read_vector_storage(None) == "faiss"


# --------------------------------------------------------------------------- #
# RAG instance caching
# --------------------------------------------------------------------------- #


def _stub_engine_counting(monkeypatch) -> list:
    """Like _stub_engine but records every build_rag call."""
    builds: list = []

    def fake_build(wd, *_a, **_):
        builds.append(wd)
        return _FakeRag(wd)

    monkeypatch.setattr(engine, "build_rag", fake_build)

    async def fake_insert(rag, content_list, *, file_name, doc_id):
        (rag.working_dir / "vdb_chunks.json").write_text(json.dumps({"vectors": [[1.0]]}))
        (rag.working_dir / "kv_store_doc_status.json").write_text(
            json.dumps(
                {doc_id: {"status": "processed", "file_path": file_name, "chunks_list": ["c1"]}}
            )
        )
        (rag.working_dir / storage.GRAPH_FILENAME).write_text("<graphml><graph/></graphml>")

    async def fake_query(rag, question, mode):
        return "A"

    monkeypatch.setattr(engine, "insert", fake_insert)
    monkeypatch.setattr(engine, "query", fake_query)
    return builds


def test_rag_instance_built_once_per_version(tmp_path, monkeypatch) -> None:
    """LightRAG reloads every store on construction — a fresh instance per
    query is what made large-KB queries block the loop (or OOM with nano).
    Indexing builds its own worker-confined instance (loop isolation); the
    searches that follow must share one cached instance. Cached instances
    are pinned to their owner loop, so the scenario runs on one loop like
    production's uvicorn loop."""
    _force_available(monkeypatch, True)
    builds = _stub_engine_counting(monkeypatch)
    _stub_parse(monkeypatch)
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")

    async def scenario() -> None:
        await pipe.initialize("kb", [str(pdf)])
        assert len(builds) == 1  # worker-confined indexing instance
        await pipe.search("q1", "kb")
        await pipe.search("q2", "kb")
        assert len(builds) == 2  # both searches share one cached instance

    asyncio.run(scenario())


def test_embedding_signature_change_rebuilds_instance(tmp_path, monkeypatch) -> None:
    """Switching the embedding model mid-process must not reuse a cached
    instance that still embeds with the old model."""
    _force_available(monkeypatch, True)
    builds = _stub_engine_counting(monkeypatch)
    _stub_parse(monkeypatch)
    pipe = LightRagPipeline(kb_base_dir=str(tmp_path))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")

    class _Sig:
        model = "other-embed"
        dim = 1024

    async def scenario() -> None:
        await pipe.initialize("kb", [str(pdf)])
        await pipe.search("q1", "kb")
        assert len(builds) == 2  # indexing worker + first search
        monkeypatch.setattr(
            "deeptutor.services.rag.embedding_signature.signature_from_embedding_config",
            lambda: _Sig(),
        )
        await pipe.search("q2", "kb")
        assert len(builds) == 3

    asyncio.run(scenario())


def test_lean_faiss_load_skips_vector_reification(tmp_path, monkeypatch) -> None:
    """Upstream FaissVectorDBStorage reifies every vector as a Python float
    list on load (~80 KB per 2560-dim record, held forever). The lean subclass
    must skip that while keeping on-demand reconstruction working."""
    pytest.importorskip("faiss")
    pytest.importorskip("lightrag")
    import faiss
    import numpy as np

    engine._install_lean_faiss_storage()
    from lightrag.kg.faiss_impl import FaissVectorDBStorage

    dim = 8
    vecs = np.random.rand(3, dim).astype(np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / "faiss_index_chunks.index"))
    (tmp_path / "faiss_index_chunks.index.meta.json").write_text(
        json.dumps(
            {
                str(i): {"__id__": f"chunk-{i}", "__created_at__": 1, "content": f"doc{i}"}
                for i in range(3)
            }
        ),
        encoding="utf-8",
    )

    class _Emb:
        embedding_dim = dim

    global_config = {
        "working_dir": str(tmp_path),
        "embedding_batch_num": 4,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }
    store = FaissVectorDBStorage(
        namespace="chunks", workspace="", global_config=global_config, embedding_func=_Emb()
    )
    assert store._index.ntotal == 3
    assert all("__vector__" not in meta for meta in store._id_to_meta.values())

    out = asyncio.run(store.get_vectors_by_ids(["chunk-1", "missing"]))
    assert list(out) == ["chunk-1"]
    assert np.allclose(out["chunk-1"], vecs[1], atol=1e-5)
