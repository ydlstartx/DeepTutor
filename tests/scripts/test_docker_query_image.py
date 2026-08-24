"""Static contracts for the full and slim Docker image profiles."""

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_query_extra_does_not_include_indexing_or_gpu_stack() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = project["project"]["optional-dependencies"]["rag-lightrag-query"]
    normalized = "\n".join(deps).lower()

    assert "lightrag-hku" in normalized
    assert "raganything" not in normalized
    assert "mineru" not in normalized
    assert "torch" not in normalized


def test_dockerfile_keeps_full_and_query_profiles_separate() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python-common AS python-query" in dockerfile
    assert "FROM python-common AS python-full" in dockerfile
    assert "FROM production-runtime AS production-query" in dockerfile
    assert "FROM production-runtime AS production\n" in dockerfile
    assert "COPY --from=python-query" in dockerfile
    assert "COPY --from=python-full" in dockerfile
    assert "DEEPTUTOR_KB_QUERY_ONLY=true" in dockerfile
    assert "find_spec('raganything') is None" in dockerfile
    assert "find_spec('mineru') is None" in dockerfile
    assert "find_spec('torch') is None" in dockerfile
    assert "n.startswith(('nvidia-','cupy'))" in dockerfile
    assert "FROM production AS development" in dockerfile


def test_dev_workflow_publishes_both_profiles() -> None:
    workflow = (ROOT / ".github/workflows/build-dev-image.yml").read_text(encoding="utf-8")

    assert "target: production\n" in workflow
    assert "tags: ${{ env.IMAGE_NAME }}:dev\n" in workflow
    assert "target: production-query\n" in workflow
    assert "tags: ${{ env.IMAGE_NAME }}:dev-query\n" in workflow
