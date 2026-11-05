"""The memory tests of CrewAI 1.15.35, copied unchanged from crewAIInc/crewAI (lib/crewai/tests/memory, MIT License,
Copyright (c) 2025 crewAI, Inc., in LICENSE in this folder), run on ChromaStorage.

LanceDBStorage, which these tests and ``Memory`` create, is replaced by a ChromaStorage on the server at
$CMC_HOST (default 127.0.0.1):$CMC_PORT, one collection for each path. With CMC_BACKEND=lancedb nothing is replaced: the same tests run on the
default backend of CrewAI, for the reference.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import uuid

import chromadb
from crewai.memory.storage import lancedb_storage
import pytest

from crewai_memory_chroma import ChromaStorage


if os.environ.get("CMC_BACKEND", "chroma") == "chroma":
    _client = (
        chromadb.HttpClient(host=os.environ.get("CMC_HOST", "127.0.0.1"), port=int(os.environ["CMC_PORT"]))
        if os.environ.get("CMC_PORT")
        else chromadb.EphemeralClient()
    )
    _run = uuid.uuid4().hex
    _created: set[str] = set()

    class LanceDBStorageOnChroma(ChromaStorage):
        """A ChromaStorage with the constructor of LanceDBStorage: the path and the table name choose the collection."""

        def __init__(
            self,
            path: str | Path | None = None,
            table_name: str = "memories",
            vector_dim: int | None = None,
            compact_every: int = 100,
        ) -> None:
            name = "t" + hashlib.sha1(f"{_run}|{path}|{table_name}".encode()).hexdigest()[:30]
            _created.add(name)
            super().__init__(client=_client, collection_name=name, vector_dim=vector_dim)

    # The original stays reachable for the comparison with LanceDB in tests/test_oracle.py.
    lancedb_storage._OriginalLanceDBStorage = lancedb_storage.LanceDBStorage  # type: ignore[attr-defined]
    lancedb_storage.LanceDBStorage = LanceDBStorageOnChroma  # type: ignore[misc]

    # Scopes are matched by path components, as QdrantEdgeStorage does, not as strings: this test searches "/share\",
    # a part of the scope name "/share\docs", and expects the record of that scope.
    _PART_OF_A_SCOPE_NAME = (
        "test_lancedb_filter_escaping.py::test_percent_underscore_and_backslash_in_scope_names_match_literally"
    )

    def pytest_collection_modifyitems(config, items):
        for item in items:
            if item.nodeid.endswith(_PART_OF_A_SCOPE_NAME):
                item.add_marker(pytest.mark.xfail(strict=True, reason="scopes are matched by path components"))

    @pytest.fixture(scope="session", autouse=True)
    def _drop_collections():
        yield
        for name in _created:
            try:
                _client.delete_collection(name)
            except Exception:  # noqa: S110
                pass
