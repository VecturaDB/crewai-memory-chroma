"""Scopes are matched by path components, as QdrantEdgeStorage does, not as strings, as LanceDBStorage does.

Records in "/crew/research", in its subscope, in "/crew/research-2", in "/crew/researcher" and in the root scope: every
operation on "/crew/research" reaches the first two only. On LanceDBStorage (CrewAI 1.15.35) the search, the list and the
count also return "/crew/research-2" and "/crew/researcher", a delete of the scope also deletes the root scope, a delete
with older_than deletes the whole scope, and a reset deletes "/crew/research-2".
"""

from __future__ import annotations

from datetime import datetime, timedelta

from crewai.memory.types import MemoryRecord

DIM = 4
NOW = datetime(2026, 11, 4, 12, 0, 0, 123456)
SCOPES = {
    "own": "/crew/research",
    "own-child": "/crew/research/agent",
    "dash": "/crew/research-2",
    "longer": "/crew/researcher",
    "root": "/",
}


def _filled(new_storage):
    storage = new_storage(vector_dim=DIM)
    storage.save(
        [
            MemoryRecord(id=k, content=k, scope=v, categories=["note"], embedding=[0.1] * DIM, created_at=NOW)
            for k, v in SCOPES.items()
        ]
    )
    return storage


def _ids(records) -> set[str]:
    return {r.id for r in records}


def test_search_list_and_count_stay_in_the_scope(new_storage) -> None:
    storage = _filled(new_storage)
    assert {r.id for r, _ in storage.search([0.1] * DIM, scope_prefix="/crew/research", limit=10)} == {"own", "own-child"}
    assert _ids(storage.list_records("/crew/research")) == {"own", "own-child"}
    assert storage.count("/crew/research") == 2
    assert storage.get_scope_info("/crew/research").child_scopes == ["/crew/research/agent"]
    assert storage.list_categories("/crew/research") == {"note": 2}
    assert storage.list_scopes("/crew") == ["/crew/research", "/crew/research-2", "/crew/researcher"]


def test_a_scope_with_a_trailing_slash_or_without_the_leading_one_is_the_same_scope(new_storage) -> None:
    storage = _filled(new_storage)
    for spelling in ("/crew/research/", "crew/research", "crew/research/"):
        assert {r.id for r, _ in storage.search([0.1] * DIM, scope_prefix=spelling, limit=10)} == {"own", "own-child"}


def test_a_part_of_a_name_is_not_a_scope(new_storage) -> None:
    storage = _filled(new_storage)
    assert storage.search([0.1] * DIM, scope_prefix="/crew/res", limit=10) == []
    assert storage.count("/crew/res") == 0


def test_delete_of_a_scope_keeps_the_root_scope_and_the_other_scopes(new_storage) -> None:
    storage = _filled(new_storage)
    assert storage.delete(scope_prefix="/crew/research") == 2
    assert _ids(storage.list_records()) == {"dash", "longer", "root"}


def test_delete_of_a_scope_with_older_than_deletes_only_the_older_records(new_storage) -> None:
    storage = _filled(new_storage)
    storage.save([MemoryRecord(id="old", content="old", scope="/crew/research", embedding=[0.1] * DIM,
                               created_at=NOW - timedelta(days=30))])
    assert storage.delete(scope_prefix="/crew/research", older_than=NOW - timedelta(days=1)) == 1
    assert _ids(storage.list_records()) == set(SCOPES)


def test_reset_of_a_scope_keeps_the_scopes_with_a_longer_name(new_storage) -> None:
    storage = _filled(new_storage)
    storage.reset(scope_prefix="/crew/research")
    assert _ids(storage.list_records()) == {"dash", "longer", "root"}


def test_the_root_scope_is_everything(new_storage) -> None:
    storage = _filled(new_storage)
    assert storage.count("/") == 5
    assert len(storage.search([0.1] * DIM, scope_prefix="/", limit=10)) == 5
    storage.reset(scope_prefix="/")
    assert storage.count() == 0
