"""The cap on scans and the dimension checks, as in LanceDBStorage.

Scans for the scope info, the counts in a scope, the categories and the deletes by category or metadata read at most
``_SCAN_ROWS_LIMIT`` records (LanceDBStorage: 50,000); the list of records, the delete of a scope and the count of everything
are not capped. A dimension that the server refuses, because another client wrote to the collection first, is raised as
the EmbeddingDimensionMismatchError of CrewAI, on save and on search.
"""

from __future__ import annotations

from crewai.memory.storage.backend import EmbeddingDimensionMismatchError
from crewai.memory.types import MemoryRecord
import pytest

from crewai_memory_chroma import storage as storage_module


def test_scans_are_capped_where_lancedb_caps_them(new_storage, monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "_SCAN_ROWS_LIMIT", 3)
    storage = new_storage(vector_dim=2)
    storage.save(
        [MemoryRecord(id=f"r{i}", content=f"r{i}", scope="/s", categories=["c"], embedding=[1.0, float(i)]) for i in range(5)]
    )
    assert storage.count("/s") == 3
    assert storage.get_scope_info("/s").record_count == 3
    assert storage.list_categories("/s") == {"c": 3}
    assert storage.count() == 5
    assert len(storage.list_records("/s")) == 5
    assert storage.delete(scope_prefix="/s", categories=["c"]) == 3
    assert storage.count() == 2
    assert storage.delete(scope_prefix="/s") == 2


def test_a_dimension_written_first_by_another_client_is_a_mismatch_on_save(new_storage, chroma_client) -> None:
    storage = new_storage()
    chroma_client.create_collection(storage._collection_name, embedding_function=None).add(
        ids=["other"], embeddings=[[1.0, 0.0, 0.0]], documents=["other"], metadatas=[{"k": 1}]
    )
    with pytest.raises(EmbeddingDimensionMismatchError) as raised:
        storage.save([MemoryRecord(id="mine", content="mine", embedding=[1.0, 0.0, 0.0, 0.0])])
    assert (raised.value.stored_dim, raised.value.new_dim) == (3, 4)


def test_a_dimension_written_first_by_another_client_is_a_mismatch_on_search(new_storage, chroma_client) -> None:
    storage = new_storage()
    assert storage.search([1.0, 0.0, 0.0, 0.0]) == []
    chroma_client.create_collection(storage._collection_name, embedding_function=None).add(
        ids=["other"], embeddings=[[1.0, 0.0, 0.0]], documents=["other"], metadatas=[{"k": 1}]
    )
    with pytest.raises(EmbeddingDimensionMismatchError) as raised:
        storage.search([1.0, 0.0, 0.0, 0.0])
    assert (raised.value.stored_dim, raised.value.new_dim) == (3, 4)
