"""The server for the tests: a Chroma server, or a server with the API of Chroma, at $CMC_HOST (default 127.0.0.1):$CMC_PORT."""

from __future__ import annotations

from collections.abc import Iterator
import os
import uuid

import chromadb
from chromadb.api import ClientAPI
import pytest

from crewai_memory_chroma import ChromaStorage


@pytest.fixture(scope="session")
def chroma_client() -> ClientAPI:
    return chromadb.HttpClient(host=os.environ.get("CMC_HOST", "127.0.0.1"), port=int(os.environ["CMC_PORT"]))


@pytest.fixture
def new_storage(chroma_client: ClientAPI) -> Iterator:
    """A function that makes a ChromaStorage on a new collection; the collections are deleted after the test."""
    names: list[str] = []

    def make(vector_dim: int | None = None) -> ChromaStorage:
        name = "cmc_" + uuid.uuid4().hex
        names.append(name)
        return ChromaStorage(client=chroma_client, collection_name=name, vector_dim=vector_dim)

    yield make
    for name in names:
        try:
            chroma_client.delete_collection(name)
        except Exception:  # noqa: S110
            pass
