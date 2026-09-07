"""The v2 schema upgrade preserves the dev branch's existing folder fields."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("key", ["fields", "schema"])
def test_existing_collection_migration_succeeds_and_keeps_existing_fields(key):
    spec = spec_from_file_location(
        "pb_setup", Path(__file__).resolve().parents[2] / "scripts" / "pb_setup.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    original = [{"name": "folder_id", "type": "text"}, {"name": "custom", "type": "text"}]
    record = SimpleNamespace(id="sessions-id", name="sessions", indexes=[], **{key: original})
    collections = Mock()
    collections.get_full_list.return_value = [record]
    pb = SimpleNamespace(collections=collections)
    new_field = {"name": "session_updated_at", "type": "number"}
    desired = {"name": "sessions", "schema": [original[0], new_field]}

    assert module._create_if_missing(pb, "sessions", desired, {"sessions"}) is True
    collections.create.assert_not_called()
    collections.update.assert_called_once_with(
        "sessions-id", {key: [*original, new_field], "indexes": []}
    )
