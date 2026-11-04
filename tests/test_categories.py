"""Categories are matched by the server before the nearest records are taken, as QdrantEdgeStorage does.

Records 0..39 on a line, only the farthest one in category "rare". LanceDBStorage (CrewAI 1.15.35) takes the nearest
limit*3 records and keeps those with the category, so a search with limit 13 or less, and Memory.recall with limit 5,
return nothing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from crewai.memory.types import MemoryRecord
from crewai.memory.unified_memory import Memory


def _line(new_storage):
    storage = new_storage(vector_dim=2)
    storage.save(
        [
            MemoryRecord(
                id=f"r{i}",
                content=f"memory {i}",
                scope="/crew/a",
                categories=["rare"] if i == 39 else ["common", "line"],
                metadata={"even": i % 2 == 0},
                embedding=[1.0, i / 10],
            )
            for i in range(40)
        ]
    )
    return storage


def test_search_finds_the_only_record_of_a_category_far_away(new_storage) -> None:
    storage = _line(new_storage)
    for limit in (1, 5, 13):
        found = storage.search([1.0, 0.0], scope_prefix="/crew/a", categories=["rare"], limit=limit)
        assert [r.id for r, _ in found] == ["r39"]


def test_search_with_several_categories_finds_a_record_of_any_of_them(new_storage) -> None:
    storage = _line(new_storage)
    found = storage.search([1.0, 0.0], categories=["missing", "rare"], limit=5)
    assert [r.id for r, _ in found] == ["r39"]
    found = storage.search([1.0, 0.0], categories=["rare", "line"], limit=3)
    assert [r.id for r, _ in found] == ["r0", "r1", "r2"]


def test_search_with_a_category_stays_in_the_scope(new_storage) -> None:
    storage = _line(new_storage)
    storage.save([MemoryRecord(id="other", content="other", scope="/crew/b", categories=["rare"], embedding=[1.0, 0.0])])
    found = storage.search([1.0, 0.0], scope_prefix="/crew/a", categories=["rare"], limit=5)
    assert [r.id for r, _ in found] == ["r39"]


def test_a_record_without_categories_is_found_without_a_category_filter(new_storage) -> None:
    storage = new_storage(vector_dim=2)
    storage.save([MemoryRecord(id="plain", content="plain", embedding=[1.0, 0.0])])
    assert storage.get_record("plain").categories == []
    assert [r.id for r, _ in storage.search([1.0, 0.0], limit=5)] == ["plain"]
    assert storage.search([1.0, 0.0], categories=["rare"], limit=5) == []


def test_memory_recall_finds_the_only_record_of_a_category_far_away(new_storage) -> None:
    storage = _line(new_storage)
    memory = Memory(
        storage=storage, llm=MagicMock(), embedder=MagicMock(side_effect=lambda texts: [[1.0, 0.0] for _ in texts])
    )
    found = memory.recall("anything", scope="/crew/a", categories=["rare"], limit=5, depth="shallow")
    assert [m.record.id for m in found] == ["r39"]
